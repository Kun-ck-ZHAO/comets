# comets/infonce.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with:
      - per-batch normalization of embeddings
      - temporal masking for near-time negatives
      - numerical stabilization (subtract row-wise max)
      - negatives-only denominator (no other positives)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self,
                ref_idx: torch.Tensor,   # [B]
                ref_vecs: torch.Tensor,  # [B, D]
                pos_vecs: torch.Tensor,  # [B, D]
                neg_vecs: torch.Tensor,  # [M, D]
                neg_idx: torch.Tensor,   # [M]
                pos_offset: int,
                neg_mask_dist=None) -> torch.Tensor:
        # --- 0) Normalize embeddings each batch (as CEBRA does) ---
        ref_vecs = F.normalize(ref_vecs, dim=-1)
        pos_vecs = F.normalize(pos_vecs, dim=-1)
        neg_vecs = F.normalize(neg_vecs, dim=-1)

        # --- 1) Compute similarities ---
        pos = (ref_vecs * pos_vecs).sum(dim=-1) / self.t         # [B]
        neg = (ref_vecs @ neg_vecs.T) / self.t                   # [B, M]

        # --- 2) Temporal masking (apply to negatives only) ---
        mask_radius = neg_mask_dist if neg_mask_dist is not None else pos_offset
        if mask_radius is not None and mask_radius > 0:
            dist = (ref_idx[:, None] - neg_idx[None, :]).abs()   # [B, M]
            # mask self and near-time neighbors (<= radius)
            mask = dist <= mask_radius
            neg = neg.masked_fill(mask, float("-inf"))

        # --- 3) Numerical stabilization (subtract per-row max) ---
        row_max, _ = torch.max(neg, dim=1, keepdim=True)
        neg = neg - row_max
        pos = pos - row_max.squeeze(1)

        # --- 4) InfoNCE loss (negatives only) ---
        loss = -pos + torch.logsumexp(neg, dim=1)
        return loss.mean()
