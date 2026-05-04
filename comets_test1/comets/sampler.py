# comets/sampler.py
import torch
from torch import Tensor


def _make_offsets(W: int, device):
    """Return offsets of total length W, skewed so left<=right."""
    left = W // 2
    right = W - left
    return torch.arange(-left, right, device=device)  # length=W


def _gather_windows(X: torch.Tensor, idx: torch.Tensor, win_len: int) -> torch.Tensor:
    """
    Gather clamped symmetric windows around 'idx' from sequence X: [T,C] -> [B,Tw,C].
    If win_len <= 1, returns just the selected timepoints with a dummy time axis.
    """
    T, C = X.shape
    if win_len <= 1:
        return X[idx].unsqueeze(1)  # [B,1,C]

    idx = idx.clamp(min=win_len//2, max=T - (win_len//2) - 1)
    offsets = _make_offsets(win_len, device=idx.device)
    win_idx = idx[:, None] + offsets[None, :]          # [B,W]
    return X[win_idx]  # [B,W,C]


class LatentSampler:
    """
    Temporal index sampler for Stage 1 (and Stage 2 time groups).
    Returns only indices; negatives are the in-batch pool with temporal
    masking handled in the loss.
    """
    def __init__(self, modA_cfg, modB_cfg, sampling_cfg, device: torch.device):
        self.win_len_A = modA_cfg.win_len
        self.win_len_B = modB_cfg.win_len
        self.batch_size = sampling_cfg.batch_size
        self.pos_offset = sampling_cfg.pos_offset
        self.device = device

    def sample(self, T: int):
        """
        Returns:
            ref_idx: [B] in [0, T - pos_offset)
            pos_idx: [B] = ref_idx + pos_offset
        """
        max_start = T - self.pos_offset
        if max_start < 0:
            ref_idx = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
            pos_idx = ref_idx
            return ref_idx, pos_idx

        ref_idx = torch.randint(0, max_start, (self.batch_size,), device=self.device)
        pos_idx = ref_idx + self.pos_offset
        return ref_idx, pos_idx


# backward-compatibility alias
InBatchSampler = LatentSampler


class UnitSampler:
    """
    Unit index sampler for Stage 2.
    Draws two independent pools of M unit indices each step,
    both sampled uniformly without replacement from {0, ..., N-1}.
    """
    def __init__(self, n_units: int, m_units: int, device: torch.device):
        self.n_units = n_units
        self.m_units = m_units
        self.device = device

    def sample(self) -> tuple[Tensor, Tensor]:
        """
        Returns:
            unit_ids_A: [M] — anchor unit pool
            unit_ids_B: [M] — positive/negative unit pool
        Both drawn uniformly without replacement from {0, ..., N-1}.
        """
        perm_A = torch.randperm(self.n_units, device=self.device)[:self.m_units]
        perm_B = torch.randperm(self.n_units, device=self.device)[:self.m_units]
        return perm_A, perm_B
