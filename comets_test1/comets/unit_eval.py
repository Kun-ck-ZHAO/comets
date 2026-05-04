# comets/unit_eval.py
"""
Stage 2 evaluation metrics for the unit embedding space E [N, d_u].

Primary:
  ratemap_correlation    — Spearman between pairwise embedding distance and ratemap similarity
  place_cell_enrichment  — permutation test: are place cells more clustered in E than chance?

Secondary:
  split_half_stability   — Procrustes alignment of two E matrices trained on disjoint halves
  nn_purity              — k-NN label purity given cell-type labels

Baselines (used externally):
  random nn.Embedding and PCA of mean firing rate profiles.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_np(x):
    """Accept ndarray, torch tensor, or list."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _pairwise_distances(E: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix [N, N], upper-triangle entries."""
    return cdist(E, E, metric="euclidean")


def _upper_tri(M: np.ndarray) -> np.ndarray:
    """Return upper-triangle (excluding diagonal) as a flat vector."""
    N = M.shape[0]
    idx = np.triu_indices(N, k=1)
    return M[idx]


# ---------------------------------------------------------------------------
# 1. Rate-map correlation
# ---------------------------------------------------------------------------

def ratemap_correlation(E, ratemaps) -> tuple[float, float]:
    """
    Spearman correlation between pairwise embedding distance (in E) and
    pairwise ratemap dissimilarity.

    Args:
        E:        [N, d_u] unit embeddings
        ratemaps: [N, ...] rate-map array per neuron (any shape after N;
                  will be flattened per neuron before computing cosine distance)

    Returns:
        (rho, p_value) — Spearman rho and two-tailed p-value.
        Positive rho is good: neurons close in E have similar rate maps.
        Returns (nan, nan) if fewer than 2 active neurons remain after filtering.
    """
    E = _to_np(E)
    R = _to_np(ratemaps).reshape(E.shape[0], -1)   # [N, bins]

    # Silent neurons (zero-norm rate maps) produce NaN cosine distances.
    # Exclude them from both E and R before computing pairwise metrics.
    active = np.linalg.norm(R, axis=1) > 0          # [N] bool
    n_active = active.sum()
    if n_active < 2:
        return float("nan"), float("nan")
    E = E[active]
    R = R[active]

    emb_dist  = _upper_tri(_pairwise_distances(E))
    rate_dist = _upper_tri(cdist(R, R, metric="cosine"))

    rho, p = spearmanr(emb_dist, rate_dist)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# 2. Place-cell enrichment
# ---------------------------------------------------------------------------

def place_cell_enrichment(E, is_place_cell, n_perms: int = 1000,
                          seed: int = 0) -> tuple[float, float]:
    """
    Permutation test: are place cells more tightly clustered in E than chance?

    Statistic: mean pairwise Euclidean distance among place cells (lower = more clustered).

    Args:
        E:             [N, d_u] unit embeddings
        is_place_cell: [N] bool array
        n_perms:       number of permutations
        seed:          numpy RNG seed for reproducibility

    Returns:
        (observed_stat, p_value)
        p_value is the fraction of permutations with stat <= observed (one-tailed).
    """
    E  = _to_np(E)
    pc = np.asarray(is_place_cell, dtype=bool)
    n_pc = pc.sum()

    if n_pc < 2:
        raise ValueError("Need at least 2 place cells for enrichment test.")

    D = _pairwise_distances(E)

    def _cluster_stat(mask):
        idx = np.where(mask)[0]
        sub = D[np.ix_(idx, idx)]
        return _upper_tri(sub).mean()

    observed = _cluster_stat(pc)

    rng = np.random.default_rng(seed)
    null = np.array([
        _cluster_stat(rng.permutation(pc))
        for _ in range(n_perms)
    ])

    p_value = float((null <= observed).mean())
    return float(observed), p_value


# ---------------------------------------------------------------------------
# 3. Split-half stability
# ---------------------------------------------------------------------------

def split_half_stability(E1, E2) -> float:
    """
    Procrustes-aligned similarity between two unit embedding matrices trained
    on non-overlapping temporal halves.

    Applies orthogonal Procrustes (no reflection) to align E2 onto E1,
    then returns mean cosine similarity between aligned pairs.

    Args:
        E1, E2: [N, d_u] — embeddings from two independent training runs
                           (must have the same N with matched unit ordering)

    Returns:
        mean cosine similarity in [−1, 1]; higher = more stable.
    """
    from scipy.linalg import orthogonal_procrustes

    E1 = _to_np(E1)
    E2 = _to_np(E2)
    assert E1.shape == E2.shape, "E1 and E2 must have matching shape [N, d_u]"

    R, _ = orthogonal_procrustes(E2, E1)   # find R s.t. ||E2 R - E1||_F is min
    E2_aligned = E2 @ R                    # [N, d_u]

    # cosine similarity per neuron pair
    n1 = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-8)
    n2 = E2_aligned / (np.linalg.norm(E2_aligned, axis=1, keepdims=True) + 1e-8)
    return float((n1 * n2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# 4. Nearest-neighbour purity
# ---------------------------------------------------------------------------

def nn_purity(E, labels, k: int = 5) -> float:
    """
    k-NN label purity: fraction of k nearest neighbours sharing the same label.

    Useful when cell-type labels are available (e.g. pyramidal vs interneuron).

    Args:
        E:      [N, d_u] unit embeddings
        labels: [N] integer or string labels
        k:      number of nearest neighbours (excluding self)

    Returns:
        mean purity in [0, 1]; higher = more label-coherent geometry.
    """
    E = _to_np(E)
    labels = np.asarray(labels)
    D = _pairwise_distances(E)   # [N, N]

    N = E.shape[0]
    purities = []
    for i in range(N):
        row = D[i].copy()
        row[i] = np.inf                       # exclude self
        nn_idx = np.argpartition(row, k)[:k]  # k nearest
        purity = (labels[nn_idx] == labels[i]).mean()
        purities.append(purity)

    return float(np.mean(purities))
