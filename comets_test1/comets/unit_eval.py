# comets/unit_eval.py
"""
Stage 2 evaluation metrics for the unit embedding space E [N, d_u].

Metrics
-------
ratemap_correlation     — Spearman ρ between pairwise embedding cosine similarity
                          and pairwise ratemap cosine similarity
si_gradient_correlation — Spearman ρ between pairwise |SI_i − SI_j| and pairwise
                          embedding cosine distance (1 − cos sim)
split_half_stability    — Procrustes-aligned mean cosine similarity between two E
                          matrices trained on disjoint temporal halves
"""

from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _cosine_sim_matrix(E: np.ndarray) -> np.ndarray:
    """[N, N] cosine similarity matrix from row-normalised E."""
    norm = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    E_n = E / norm
    return E_n @ E_n.T


def _upper_tri(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


# ---------------------------------------------------------------------------
# 1. Rate-map correlation
# ---------------------------------------------------------------------------

def ratemap_correlation(E, ratemaps) -> tuple[float, float]:
    """
    Spearman ρ between pairwise embedding cosine similarity and pairwise
    ratemap cosine similarity.

    Args:
        E:        [N, d_u] unit embeddings
        ratemaps: [N, ...] rate-map per neuron (flattened to [N, bins])

    Returns:
        (rho, p_value) — positive rho means neurons close in E have similar
        spatial tuning.  Returns (nan, nan) if fewer than 2 active neurons.
    """
    E = _to_np(E)
    R = _to_np(ratemaps).reshape(E.shape[0], -1)

    active = np.linalg.norm(R, axis=1) > 0
    if active.sum() < 2:
        return float("nan"), float("nan")
    E = E[active]
    R = R[active]

    emb_sim  = _upper_tri(_cosine_sim_matrix(E))
    rate_sim = _upper_tri(_cosine_sim_matrix(R))

    rho, p = spearmanr(emb_sim, rate_sim)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# 2. SI gradient correlation
# ---------------------------------------------------------------------------

def si_gradient_correlation(E, si_values) -> tuple[float, float]:
    """
    Spearman ρ between pairwise |SI_i − SI_j| and pairwise embedding cosine
    distance (1 − cosine similarity).

    Tests whether neurons with similar spatial information scores are
    geometrically close in the embedding space — without requiring a
    binary place-cell label.

    Args:
        E:         [N, d_u] unit embeddings
        si_values: [N] spatial information in bits/spike

    Returns:
        (rho, p_value) — positive rho means SI-similar neurons cluster together.
        Returns (nan, nan) if fewer than 2 neurons have finite SI.
    """
    E  = _to_np(E)
    si = _to_np(si_values).ravel()

    valid = np.isfinite(si)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    E  = E[valid]
    si = si[valid]

    N = len(si)
    si_diff  = np.abs(si[:, None] - si[None, :])   # [N, N]
    emb_dist = 1.0 - _cosine_sim_matrix(E)          # [N, N] cosine distance

    rho, p = spearmanr(_upper_tri(si_diff), _upper_tri(emb_dist))
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# 3. Split-half stability
# ---------------------------------------------------------------------------

def split_half_stability(E1, E2) -> float:
    """
    Procrustes-aligned mean cosine similarity between two unit embedding
    matrices trained on non-overlapping temporal halves.

    Args:
        E1, E2: [N, d_u] — matched unit ordering, same shape

    Returns:
        mean cosine similarity in [−1, 1]; higher = more stable.
    """
    E1 = _to_np(E1)
    E2 = _to_np(E2)
    assert E1.shape == E2.shape, "E1 and E2 must have matching shape [N, d_u]"

    R, _ = orthogonal_procrustes(E2, E1)
    E2_aligned = E2 @ R

    n1 = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-8)
    n2 = E2_aligned / (np.linalg.norm(E2_aligned, axis=1, keepdims=True) + 1e-8)
    return float((n1 * n2).sum(axis=1).mean())
