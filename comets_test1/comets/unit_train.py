# comets/unit_train.py
from typing import Optional
import numpy as np
import torch
from tqdm import trange

from .configs import SamplingConfig, UnitTrainingConfig
from .core import CoMETsModel, _device, _to_tensor
from .sampler import UnitSampler
from .unit_encoder import UnitEncoder, UnitEncoderConfig
from .unit_loss import UnitLoss


def train_unit(latent_model: CoMETsModel,
               spikes,
               sampling_cfg: SamplingConfig,
               cfg: UnitTrainingConfig,
               lfp=None,
               device: Optional[torch.device] = None,
               z_spikes=None,
               z_lfp=None) -> UnitEncoder:
    """
    Stage 2: train a stationary unit (neuron) embedding given a frozen latent encoder.

    Args:
        latent_model:  Trained CoMETsModel from Stage 1 (weights are frozen here).
        spikes:        [T, N] spike count array — column j = neuron j (fixed order).
        sampling_cfg:  Temporal sampling config (batch_size used as K ctx windows).
        cfg:           UnitTrainingConfig.
        lfp:           [T, C] LFP array (optional).  If provided, LFP latent is
                       included in context similarity weighted by lambda_lfp_ctx.
        device:        Torch device; defaults to CUDA if available.
        z_spikes:      [T, D] pre-computed spike latent (optional). When provided,
                       skips latent_model.embed_latent_spikes() — useful for
                       ablations (e.g. PCA baseline).
        z_lfp:         [T, D] pre-computed LFP latent (optional). When provided,
                       skips latent_model.embed_latent_lfp().

    Returns:
        Trained UnitEncoder. Call .embeddings() -> [N, d_u] for all unit vectors.
    """
    device = device or _device()

    spikes_t = _to_tensor(spikes, device)   # [T, N]
    T, N = spikes_t.shape

    # --- precompute frozen latent trajectories ---
    # Use pre-computed arrays when provided (e.g. PCA ablation); otherwise
    # run the latent model's forward pass.
    if z_spikes is not None:
        Z_spikes = _to_tensor(np.asarray(z_spikes, dtype=np.float32), device)
    else:
        Z_spikes = torch.from_numpy(
            latent_model.embed_latent_spikes(spikes)
        ).float().to(device)                    # [T, D_spike]

    if z_lfp is not None:
        Z_lfp = _to_tensor(np.asarray(z_lfp, dtype=np.float32), device)
    elif lfp is not None:
        Z_lfp = torch.from_numpy(
            latent_model.embed_latent_lfp(lfp)
        ).float().to(device)               # [T, D_lfp]
    else:
        Z_lfp = None

    # --- unit encoder + optimizer ---
    unit_enc = UnitEncoder(UnitEncoderConfig(n_units=N, d_u=cfg.d_u)).to(device)

    opt = torch.optim.AdamW(unit_enc.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, T // sampling_cfg.batch_size)
    total_steps     = cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # --- loss + unit sampler ---
    loss_fn = UnitLoss(
        temperature      = cfg.temperature,
        lambda_coact     = cfg.lambda_coact,
        lambda_ctx       = cfg.lambda_ctx,
        lambda_spike_ctx = cfg.lambda_spike_ctx,
        lambda_lfp_ctx   = cfg.lambda_lfp_ctx,
    )
    unit_sampler = UnitSampler(n_units=N, m_units=cfg.m_units, device=device)

    K          = sampling_cfg.batch_size   # context bank size
    pos_offset = sampling_cfg.pos_offset
    M          = cfg.m_units

    pbar = trange(cfg.epochs, dynamic_ncols=True, desc="Stage 2: Unit Encoder", leave=True)
    for ep in pbar:
        unit_enc.train()
        sum_loss = 0.0

        for _ in range(steps_per_epoch):

            # ---- temporal anchor / pos ----
            max_start = max(1, T - pos_offset)
            t_anchor  = int(torch.randint(0, max_start, (1,)).item())
            t_pos     = min(t_anchor + pos_offset, T - 1)

            # ================================================================
            # L_coact — 2M pool: same M units at t_anchor and t_pos
            # ================================================================
            uid_coact = torch.randperm(N, device=device)[:M]          # [M]
            uid_pool  = torch.cat([uid_coact, uid_coact])              # [2M]

            with torch.no_grad():
                r_anchor = spikes_t[t_anchor, uid_coact]               # [M]
                r_pos    = spikes_t[t_pos,    uid_coact]               # [M]
                r_pool   = torch.cat([r_anchor, r_pos])                # [2M]

            e_pool = unit_enc(uid_pool)                                # [2M, d_u]

            # ================================================================
            # L_ctx — two independent K-banks drawn from the full time range
            #         (decoupled from t_anchor / t_pos)
            # ================================================================
            ctx_A = torch.randint(0, T, (K,), device=device)          # [K]
            ctx_B = torch.randint(0, T, (K,), device=device)          # [K]

            uid_ctx_A, uid_ctx_B = unit_sampler.sample()              # [M], [M]

            with torch.no_grad():
                # firing rates: [K, M] -> .T -> [M, K]
                r_ctx_A = spikes_t[ctx_A[:, None],
                                   uid_ctx_A[None, :]].T.contiguous()  # [M, K]
                r_ctx_B = spikes_t[ctx_B[:, None],
                                   uid_ctx_B[None, :]].T.contiguous()  # [M, K]

                z_spike_A = Z_spikes[ctx_A]                            # [K, D]
                z_spike_B = Z_spikes[ctx_B]                            # [K, D]
                z_lfp_A   = Z_lfp[ctx_A] if Z_lfp is not None else None
                z_lfp_B   = Z_lfp[ctx_B] if Z_lfp is not None else None

            e_ctx_A = unit_enc(uid_ctx_A)                             # [M, d_u]
            e_ctx_B = unit_enc(uid_ctx_B)                             # [M, d_u]

            # ================================================================
            # Combined loss + update
            # ================================================================
            loss = loss_fn(
                e_pool, r_pool, uid_pool,
                e_ctx_A, e_ctx_B, r_ctx_A, r_ctx_B,
                z_spike_A, z_spike_B, z_lfp_A, z_lfp_B,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unit_enc.parameters(), 1.0)
            opt.step()
            scheduler.step()

            sum_loss += loss.item()

        avg = sum_loss / steps_per_epoch
        pbar.set_postfix_str(f"loss={avg:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    return unit_enc
