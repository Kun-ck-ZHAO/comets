# comets/unit_loss.py
"""
Stage 2 unit loss: L_coact + L_ctx.

Gates:
  - Co-firing weight:  log(r_i+1) * log(r_j+1)  — zero when either neuron silent,
                       scales with joint intensity.  No sigmoid, no tau_act.
  - Co-activity bias:  batch-centered relu on log-rates — only pairs where BOTH
                       neurons fire above the batch mean contribute.  Handles
                       exploding units automatically (they raise the mean, pushing
                       other units below the threshold).
  - Context sim:       element-wise cosine of frozen latent pairs from two
                       independent K-banks.  Spike and/or LFP latent, weighted.

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

    # skip rows where every weight is zero (no valid positive this step)
    valid = W.max(dim=1).values > 0
    if valid.any():
        return loss[valid].mean()
    # All weights zero — return a zero that stays in the computation graph so
    # .backward() doesn't fail when this is the only active loss term.
    return (e_A * 0.0).mean()


# ---------------------------------------------------------------------------
# Co-activity loss  (L_coact)
# ---------------------------------------------------------------------------

def coact_loss(e_pool: Tensor, r_pool: Tensor,
               uid_pool: Tensor, temperature: float) -> Tensor:
    """
    Pulls pairs of neurons that both fire above the batch mean at nearby windows.

    e_pool:   [2M, d_u] — embeddings: first M at t_anchor, next M at t_pos
                          (same M unit indices repeated, different rates)
    r_pool:   [2M]      — firing rates  [r at t_anchor ; r at t_pos]
    uid_pool: [2M]      — unit indices  [uid ; uid]  (first M == last M)
    """
    # batch-centered relu: only above-average firing contributes
    lr    = torch.log(r_pool + 1.0)           # [2M]
    delta = torch.relu(lr - lr.mean())         # [2M]  — zero below batch mean
    W     = (delta[:, None] * delta[None, :]).detach()  # [2M, 2M]

    # mask all same-unit pairs (self-pairs and cross-time same-unit pairs)
    exclude = (uid_pool[:, None] == uid_pool[None, :])  # [2M, 2M] bool
    W = W.masked_fill(exclude, 0.0)

    return _soft_infonce(e_pool, e_pool, W, temperature, exclude_mask=exclude)


# ---------------------------------------------------------------------------
# Context-conditioned loss  (L_ctx)
# ---------------------------------------------------------------------------

def ctx_loss(e_A: Tensor, e_B: Tensor,
             r_A: Tensor, r_B: Tensor,
             z_spike_A: Tensor, z_spike_B: Tensor,
             temperature: float,
             lambda_spike: float = 1.0,
             lambda_lfp: float = 0.0,
             z_lfp_A: Tensor | None = None,
             z_lfp_B: Tensor | None = None) -> Tensor:
    """
    Pulls pairs of neurons that co-fire in windows sharing similar latent context,
    across temporally diverse time points (the negative bank).

    e_A, e_B:    [M, d_u]  — embeddings for two independently sampled unit pools
    r_A:         [M, K]    — firing rates for pool A at K ctx_A window indices
    r_B:         [M, K]    — firing rates for pool B at K ctx_B window indices
    z_spike_A:   [K, D]    — frozen spike latent at ctx_A windows
    z_spike_B:   [K, D]    — frozen spike latent at ctx_B windows
    lambda_spike / lambda_lfp: relative context-modality weights
    z_lfp_A/B:  [K, D]    — frozen LFP latent at ctx_A/B windows (optional)
    """
    # log-compressed co-firing: [M, M, K]
    lr_A    = torch.log(r_A + 1.0)                       # [M, K]
    lr_B    = torch.log(r_B + 1.0)                       # [M, K]
    co_fire = lr_A[:, None, :] * lr_B[None, :, :]        # [M, M, K]

    # element-wise context similarity per K pair: cosine(ctx_A[k], ctx_B[k])
    def _cosine_elementwise(zA: Tensor, zB: Tensor) -> Tensor:
        return (F.normalize(zA, dim=-1) * F.normalize(zB, dim=-1)).sum(dim=-1)  # [K]

    ctx_sim = _cosine_elementwise(z_spike_A, z_spike_B) * lambda_spike  # [K]
    if z_lfp_A is not None and z_lfp_B is not None:
        ctx_sim = ctx_sim + _cosine_elementwise(z_lfp_A, z_lfp_B) * lambda_lfp
    ctx_sim = ctx_sim.clamp(min=0).detach()  # [K]  — only positive similarity counts

    # pair weight: sum co-firing over K context pairs, weighted by context sim
    W = (co_fire * ctx_sim[None, None, :]).sum(dim=-1).detach()  # [M, M]

    return _soft_infonce(e_A, e_B, W, temperature)


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class UnitLoss(torch.nn.Module):
    """lambda_coact * L_coact  +  lambda_ctx * L_ctx."""

    def __init__(self, temperature: float = 0.1,
                 lambda_coact: float = 1.0,
                 lambda_ctx: float = 1.0,
                 lambda_spike_ctx: float = 1.0,
                 lambda_lfp_ctx: float = 1.0):
        super().__init__()
        self.temperature      = temperature
        self.lambda_coact     = lambda_coact
        self.lambda_ctx       = lambda_ctx
        self.lambda_spike_ctx = lambda_spike_ctx
        self.lambda_lfp_ctx   = lambda_lfp_ctx

    def forward(self,
                # coact inputs
                e_pool:    Tensor, r_pool:   Tensor, uid_pool:  Tensor,
                # ctx unit inputs
                e_ctx_A:   Tensor, e_ctx_B:  Tensor,
                r_ctx_A:   Tensor, r_ctx_B:  Tensor,
                # context latents
                z_spike_A: Tensor, z_spike_B: Tensor,
                z_lfp_A:   Tensor | None = None,
                z_lfp_B:   Tensor | None = None) -> Tensor:

        l_coact = coact_loss(e_pool, r_pool, uid_pool, self.temperature)
        l_ctx   = ctx_loss(e_ctx_A, e_ctx_B, r_ctx_A, r_ctx_B,
                           z_spike_A, z_spike_B, self.temperature,
                           self.lambda_spike_ctx, self.lambda_lfp_ctx,
                           z_lfp_A, z_lfp_B)
        return self.lambda_coact * l_coact + self.lambda_ctx * l_ctx
