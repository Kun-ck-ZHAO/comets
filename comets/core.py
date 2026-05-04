# comets/core.py
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from .configs import ModalityConfig, SamplingConfig, TrainingConfig
from .infonce import InfoNCELoss
from .heads import make_head, TwoLayerMLP
from .sampler import InBatchSampler, _gather_windows


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float().to(device)


class CoMETs:
    def __init__(self, encA: nn.Module, encB: nn.Module,
                 projA: nn.Module, projB: nn.Module,
                 device=None, win_len_A=5, win_len_B=5):
        self.encA, self.encB = encA, encB
        self.projA, self.projB = projA, projB
        self.device = device or next(encA.parameters()).device
        self.win_len_A = win_len_A
        self.win_len_B = win_len_B

    @torch.no_grad()
    def _embed(self, X, encoder, projector, win_len, batch_size=512):
        X = _to_tensor(X, self.device)
        T = X.shape[0]
        encoder.eval(); projector.eval()

        out = []
        for start in range(0, T, batch_size):
            stop = min(start + batch_size, T)
            idx = torch.arange(start, stop, device=self.device)
            batch = _gather_windows(X, idx, win_len)
            z = projector(encoder(batch))
            out.append(z)
        Z = torch.cat(out, dim=0)
        return Z.detach().cpu().numpy()

    def embedA(self, X, batch_size=512):
        return self._embed(X, self.encA, self.projA, self.win_len_A, batch_size)

    def embedB(self, X, batch_size=512):
        return self._embed(X, self.encB, self.projB, self.win_len_B, batch_size)

    def embed_all(self, A, B, batch_size=512):
        ZA = self.embedA(A, batch_size)
        ZB = self.embedB(B, batch_size)
        return ZA, ZB


def train(A, modA: ModalityConfig,
          B, modB: ModalityConfig,
          sampling: SamplingConfig,
          cfg: TrainingConfig,
          device: Optional[torch.device] = None) -> CoMETs:

    # --- device + data prep ---
    device = device or _device()
    A, B = _to_tensor(A, device), _to_tensor(B, device)
    T = A.shape[0]  # define once, reuse everywhere

    # --- model init ---
    encA = modA.encoder_cls(modA.encoder_cfg).to(device)
    encB = modB.encoder_cls(modB.encoder_cfg).to(device)
    head_cls = cfg.head_cls or TwoLayerMLP
    projA = make_head(modA.encoder_cfg.emb_dim, cfg.proj_dim, head_cls).to(device)
    projB = make_head(modB.encoder_cfg.emb_dim, cfg.proj_dim, head_cls).to(device)

    params = list(encA.parameters()) + list(encB.parameters()) \
           + list(projA.parameters()) + list(projB.parameters())

    # --- Optimizer: AdamW + Cosine Annealing + Gradient Clipping ---
    opt = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay if cfg.weight_decay > 0 else 1e-5
    )
    total_steps = (T // sampling.batch_size) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # --- Loss and Sampler ---
    loss_fn = InfoNCELoss(cfg.temperature)
    sampler = InBatchSampler(modA, modB, sampling, device)
    Bsize, Poff = sampling.batch_size, sampling.pos_offset
    total_epochs = cfg.epochs

    pbar = trange(total_epochs, dynamic_ncols=True, desc="Training", leave=True)
    for ep in pbar:
        encA.train(); encB.train(); projA.train(); projB.train()
        sum_total, sum_within, sum_cross = 0.0, 0.0, 0.0
        steps = max(1, T // Bsize)

        for _ in range(steps):
            # --- sample indices from uniform prior ---
            ref_idx, pos_idx = sampler.sample(T)

            # --- gather windows ---
            A_ref_w = _gather_windows(A, ref_idx, modA.win_len)
            A_pos_w = _gather_windows(A, pos_idx, modA.win_len)
            B_ref_w = _gather_windows(B, ref_idx, modB.win_len)
            B_pos_w = _gather_windows(B, pos_idx, modB.win_len)

            # --- forward passes ---
            A_ref = projA(encA(A_ref_w))
            A_pos = projA(encA(A_pos_w))
            B_ref = projB(encB(B_ref_w))
            B_pos = projB(encB(B_pos_w))

            # --- in-batch negatives ---
            A_neg, A_neg_idx = A_ref, ref_idx
            B_neg, B_neg_idx = B_ref, ref_idx

            # --- InfoNCE losses ---
            loss_AA = loss_fn(ref_idx, A_ref, A_pos, A_neg, A_neg_idx, Poff, sampling.negative_mask_dist)
            loss_BB = loss_fn(ref_idx, B_ref, B_pos, B_neg, B_neg_idx, Poff, sampling.negative_mask_dist)
            loss_AB = loss_fn(ref_idx, A_ref, B_ref, B_neg, B_neg_idx, Poff, sampling.negative_mask_dist)
            loss_BA = loss_fn(ref_idx, B_ref, A_ref, A_neg, A_neg_idx, Poff, sampling.negative_mask_dist)

            within_loss = cfg.beta * loss_AA + (1.0 - cfg.beta) * loss_BB
            cross_loss  = cfg.beta * loss_AB + (1.0 - cfg.beta) * loss_BA
            total = cfg.alpha_cross * cross_loss + (1.0 - cfg.alpha_cross) * within_loss

            # --- backprop ---
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

            # --- accumulate logs ---
            sum_total  += total.item()
            sum_within += within_loss.item()
            sum_cross  += cross_loss.item()

        avg_total  = sum_total  / steps
        avg_within = sum_within / steps
        avg_cross  = sum_cross  / steps
        pbar.set_postfix_str(
            f"total={avg_total:.4f}, within={avg_within:.4f}, cross={avg_cross:.4f}, lr={scheduler.get_last_lr()[0]:.2e}"
        )

    return CoMETs(encA, encB, projA, projB, device=device,
                  win_len_A=modA.win_len, win_len_B=modB.win_len)
