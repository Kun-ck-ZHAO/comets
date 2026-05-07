# comets/unit_train.py
from typing import Optional
import numpy as np
import torch
from tqdm import trange

from .configs import SamplingConfig, UnitTrainingConfig
from .core import CoMETsModel, _device, _to_tensor
from .sampler import LatentSampler, UnitSampler, _gather_windows
from .unit_encoder import UnitEncoder, UnitEncoderConfig
from .unit_loss import UnitLoss


def train_unit(latent_model,
               spikes,
               sampling_cfg: SamplingConfig,
               cfg: UnitTrainingConfig,
               lfp=None,
               device: Optional[torch.device] = None,
               z_spikes=None,
               z_lfp=None,
               epoch_callback=None,
               eval_every: int = 20) -> UnitEncoder:
    """
    Stage 2: train a stationary unit (neuron) embedding given a frozen latent encoder.

    Args:
        latent_model:  Trained CoMETsModel from Stage 1 (weights are frozen here).
        spikes:        [T, N] spike count array — column j = neuron j (fixed order).
        sampling_cfg:  Temporal sampling config (batch_size used as K ctx windows).
        cfg:           UnitTrainingConfig.
        lfp:           [T, C] LFP array (optional).
        device:        Torch device; defaults to CUDA if available.
        z_spikes:      [T, D] pre-computed spike latent (optional).
        z_lfp:         [T, D] pre-computed LFP latent (optional).

    Returns:
        Trained UnitEncoder. Call .embeddings() -> [N, d_u] for all unit vectors.
    """
    device = device or _device()

    spikes_t = _to_tensor(spikes, device)   # [T, N]
    T, N = spikes_t.shape

    # --- precompute frozen latent trajectories ---
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
        ).float().to(device)                    # [T, D_lfp]
    else:
        Z_lfp = None

    # --- unit encoder + optimizer ---
    unit_enc = UnitEncoder(UnitEncoderConfig(n_units=N, d_u=cfg.d_u)).to(device)

    opt = torch.optim.AdamW(unit_enc.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, T // sampling_cfg.batch_size)
    total_steps     = cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # --- loss + samplers ---
    loss_fn = UnitLoss(
        temperature  = cfg.temperature,
        lambda_coact = cfg.lambda_coact,
        lambda_ctx   = cfg.lambda_ctx,
    )
    latent_sampler = LatentSampler(sampling_cfg, device)
    unit_sampler   = UnitSampler(n_units=N, m_units=cfg.m_units, device=device)

    K       = sampling_cfg.batch_size   # context bank size (= batch size)
    M       = cfg.m_units
    win_len = cfg.win_len

    pbar = trange(cfg.epochs, dynamic_ncols=True, desc="Stage 2: Unit Encoder", leave=True)
    for ep in pbar:
        unit_enc.train()
        sum_loss = 0.0

        for _ in range(steps_per_epoch):

            # ---- temporal sampling (shared backbone) ----
            # ref_idx [K]: anchor time points — used as ctx_A bank for L_ctx
            # pos_idx [K]: anchor + pos_offset — first pair used for L_coact
            ref_idx, pos_idx = latent_sampler.sample(T)   # [K], [K]

            # ================================================================
            # L_coact — 2M pool: same M units at t_anchor and t_pos
            # ================================================================
            uid_coact = torch.randperm(N, device=device)[:M]   # [M]
            uid_pool  = torch.cat([uid_coact, uid_coact])       # [2M]

            with torch.no_grad():
                t_anc_idx = ref_idx[0:1]   # [1] — first pair from sampler
                t_pos_idx = pos_idx[0:1]   # [1]
                spk_anc  = _gather_windows(spikes_t, t_anc_idx, win_len)[0][:, uid_coact]  # [win_len, M]
                spk_pos  = _gather_windows(spikes_t, t_pos_idx, win_len)[0][:, uid_coact]  # [win_len, M]
                spk_pool = torch.cat([spk_anc, spk_pos], dim=1)                            # [win_len, 2M]

            e_pool = unit_enc(uid_pool)                                        # [2M, d_u]

            # ================================================================
            # L_ctx — single neg bank (ref_idx), two independent unit pools
            # ================================================================
            uid_ctx_A, uid_ctx_B = unit_sampler.sample()   # [M], [M]

            with torch.no_grad():
                # spike windows: [K, win_len, M] — same K neg bank for both pools
                spk_ctx_A = _gather_windows(spikes_t, ref_idx, win_len)[:, :, uid_ctx_A]  # [K, win_len, M]
                spk_ctx_B = _gather_windows(spikes_t, ref_idx, win_len)[:, :, uid_ctx_B]  # [K, win_len, M]

                # latent windows: [K, win_len, D] — not mean-pooled, kept for flatten+cosine
                z_ctx = _gather_windows(Z_spikes, ref_idx, win_len)   # [K, win_len, D]
                if Z_lfp is not None:
                    z_lfp_ctx = _gather_windows(Z_lfp, ref_idx, win_len)  # [K, win_len, D_lfp]
                else:
                    z_lfp_ctx = None

            e_ctx_A = unit_enc(uid_ctx_A)   # [M, d_u]
            e_ctx_B = unit_enc(uid_ctx_B)   # [M, d_u]

            # ================================================================
            # Combined loss + update
            # ================================================================
            loss = loss_fn(
                e_pool, spk_pool, uid_pool,
                e_ctx_A, e_ctx_B,
                spk_ctx_A, spk_ctx_B,
                z_ctx, z_lfp_ctx,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unit_enc.parameters(), 1.0)
            opt.step()
            scheduler.step()

            sum_loss += loss.item()

        avg = sum_loss / steps_per_epoch
        pbar.set_postfix_str(f"loss={avg:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch_callback is not None and (ep + 1) % eval_every == 0:
            epoch_callback(ep + 1, unit_enc)
            unit_enc.train()

    return unit_enc
