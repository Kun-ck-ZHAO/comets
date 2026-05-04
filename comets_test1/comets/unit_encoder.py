# comets/unit_encoder.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UnitEncoderConfig:
    n_units: int        # total number of neurons N
    d_u: int = 8        # embedding dimension
    projector: bool = False  # optional linear projector on top of the embedding table


class UnitEncoder(nn.Module):
    """
    Stationary unit (neuron) embedding table.

    Each neuron i has a learnable embedding e_i in R^{d_u} that does not
    vary with time. Firing rates and latent context enter only as loss gates,
    not as input to this module.

    forward(unit_ids) returns L2-normalized embeddings for the queried units.
    embeddings()      returns L2-normalized embeddings for ALL N units [N, d_u].
    """
    def __init__(self, cfg: UnitEncoderConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_units, cfg.d_u)
        self.proj = nn.Linear(cfg.d_u, cfg.d_u, bias=False) if cfg.projector else None
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, unit_ids: torch.Tensor) -> torch.Tensor:
        """
        unit_ids: [M] long tensor of neuron indices
        returns:  [M, d_u] L2-normalized embeddings
        """
        e = self.embed(unit_ids)          # [M, d_u]
        if self.proj is not None:
            e = self.proj(e)
        return F.normalize(e, dim=-1)

    @torch.no_grad()
    def embeddings(self) -> torch.Tensor:
        """Return [N, d_u] L2-normalized embeddings for all units (CPU, detached)."""
        idx = torch.arange(self.embed.num_embeddings, device=self.embed.weight.device)
        e = self.embed(idx)
        if self.proj is not None:
            e = self.proj(e)
        return F.normalize(e, dim=-1).cpu()
