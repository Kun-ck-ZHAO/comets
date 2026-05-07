# comets/unit_loss.py
"""
Stage 2 unit loss: L_coact + L_ctx.

Co-firing weight:
  Dot product of spike windows over win_len sub-windows.
  Naturally zero when either neuron is silent or both fire in different
  sub-windows — no explicit gate needed.
  Users should normalize or transform the spike array before training if desired.

Context similarity:
  Element-wise cosine of mean-pooled latent windows from two independent
  K-banks. Spike and/or LFP latent, weighted.

Soft InfoNCE:
  loss_i = log Σ_j exp(s_ij/τ)  −  log Σ_j W[i,j]·exp(s_ij/τ)
  All W tensors are detached.  Rows with zero total weight are skipped.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Core soft InfoNCE
# ---------------------------------------------------------------------------

def _soft_infonce(e_A: Tensor, e_B: Tensor, W: Tensor,
                  temperature: float,
                  exclude_mask: Tensor | None = None) -> Tensor:
    """
    e_A: [N, d]  e_B: [M, d] — L2-normalized embeddings
    W:   [N, M]  — non-negative weights (detached)
    exclude_mask: [N, M] bool — pairs removed from both numerator and denominator
                               (use for same-unit pairs in the 2M coact pool)
    """
    sim = (e_A @ e_B.T) / temperature  # [N, M]

    # denominator
    sim_den = sim if exclude_mask is None else sim.masked_fill(exclude_mask, float("-inf"))
    log_den = torch.logsumexp(sim_den, dim=1)  # [N]

    # numerator
    log_num_terms = sim + W.clamp(min=1e-30).log()
    if exclude_mask is not None:
        log_num_terms = log_num_terms.masked_fill(exclude_mask, float("-inf"))
    log_num = torch.logsumexp(log_num_terms, dim=1)  # [N]

    loss = log_den - log_num  # [N]

    valid = W.max(dim=1).values > 0
    if valid.any():
        return loss[valid].mean()
    return (e_A * 0.0).mean()


# ---------------------------------------------------------------------------
# Co-activity loss  (L_coact)
# ---------------------------------------------------------------------------

def coact_loss(e_pool: Tensor, spk_pool: Tensor,
               uid_pool: Tensor, temperature: float) -> Tensor:
    """
    e_pool:   [2M, d_u]     — embeddings: first M at t_anchor, next M at t_pos
    spk_pool:  [win_len, 2M] — spike windows for the 2M pool
    uid_pool: [2M]          — unit indices [uid ; uid]
    """
    # dot product of spike windows: W[i,j] = spk_pool[:,i] · spk_pool[:,j]
    W = (spk_pool.T @ spk_pool).detach()   # [2M, 2M]

    # mask all same-unit pairs
    exclude = (uid_pool[:, None] == uid_pool[None, :])   # [2M, 2M]
    W = W.masked_fill(exclude, 0.0)

    return _soft_infonce(e_pool, e_pool, W, temperature, exclude_mask=exclude)


# ---------------------------------------------------------------------------
# Context-conditioned loss  (L_ctx)
# ---------------------------------------------------------------------------

def ctx_loss(e_A: Tensor, e_B: Tensor,
             spk_ctx_A: Tensor, spk_ctx_B: Tensor,
             z_ctx: Tensor,
             temperature: float,
             z_lfp_ctx: Tensor | None = None) -> Tensor:
    """
    e_A, e_B:     [M, d_u]        — embeddings for two unit pools
    spk_ctx_A/B:  [K, win_len, M] — spike windows for each pool, same neg bank K
    z_ctx:        [K, win_len, D]  — spike latent windows at K context windows
    z_lfp_ctx:    [K, win_len, D]  — LFP latent windows (optional, concatenated)

    Low-rank W factorization (ctx_sim [K,K] never materialized, rank <= win_len*D):
      Z_norm  [K, win_len*D]       = F.normalize(z_ctx.reshape(K,-1))
      C_A     [win_len*D, win_len, M] = einsum('kd,kti->dti', Z_norm, spk_ctx_A)
      C_B     [win_len*D, win_len, M] = einsum('kd,ktj->dtj', Z_norm, spk_ctx_B)
      W       [M, M]               = einsum('dti,dtj->ij', C_A, C_B)
    """
    K = spk_ctx_A.shape[0]

    # flatten latent windows — captures temporal pattern, not just mean: [K, win_len*D]
    z = z_ctx if z_lfp_ctx is None else torch.cat([z_ctx, z_lfp_ctx], dim=-1)
    Z_norm = F.normalize(z.reshape(K, -1), dim=-1).detach()        # [K, win_len*D]

    # low-rank W: route through win_len*D bottleneck instead of materializing [K,K]
    C_A = torch.einsum('kd,kti->dti', Z_norm, spk_ctx_A)           # [win_len*D, win_len, M]
    C_B = torch.einsum('kd,ktj->dtj', Z_norm, spk_ctx_B)           # [win_len*D, win_len, M]
    W   = torch.einsum('dti,dtj->ij', C_A, C_B).detach()           # [M, M]

    return _soft_infonce(e_A, e_B, W, temperature)


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class UnitLoss(torch.nn.Module):
    """lambda_coact * L_coact  +  lambda_ctx * L_ctx."""

    def __init__(self, temperature: float = 0.1,
                 lambda_coact: float = 1.0,
                 lambda_ctx: float = 1.0):
        super().__init__()
        self.temperature  = temperature
        self.lambda_coact = lambda_coact
        self.lambda_ctx   = lambda_ctx

    def forward(self,
                # coact inputs
                e_pool:    Tensor, spk_pool:   Tensor, uid_pool:  Tensor,
                # ctx inputs — single neg bank, two unit pools
                e_ctx_A:   Tensor, e_ctx_B:    Tensor,
                spk_ctx_A: Tensor, spk_ctx_B:  Tensor,
                z_ctx:     Tensor,
                z_lfp_ctx: Tensor | None = None) -> Tensor:

        l_coact = coact_loss(e_pool, spk_pool, uid_pool, self.temperature)
        l_ctx   = ctx_loss(e_ctx_A, e_ctx_B, spk_ctx_A, spk_ctx_B,
                           z_ctx, self.temperature, z_lfp_ctx)
        return self.lambda_coact * l_coact + self.lambda_ctx * l_ctx
