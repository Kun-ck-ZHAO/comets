# comets/configs.py
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class ModalityConfig:
    encoder_cls: nn.Module
    encoder_cfg: object
    win_len: int


@dataclass
class SamplingConfig:
    batch_size: int = 256
    pos_offset: int = 1
    negative_mask_dist: int = 0  # 0 means only self masked


@dataclass
class LatentTrainingConfig:
    epochs: int = 200
    lr: float = 3e-4
    temperature: float = 1.0
    weight_decay: float = 1e-5
    proj_dim: int = 16
    beta: float = 0.5           # within-modality vs cross balance
    alpha_cross: float = 0.5    # cross-modality weighting
    head_cls: nn.Module = None  # defaults to TwoLayerMLP (set in train_latent)


# kept for backward compatibility; prefer LatentTrainingConfig
TrainingConfig = LatentTrainingConfig


@dataclass
class UnitTrainingConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    d_u: int = 8            # unit embedding dimension
    m_units: int = 32       # units sampled per step per pool (M)
    temperature: float = 0.1
    lambda_coact: float = 1.0
    lambda_ctx: float = 1.0
    lambda_spike_ctx: float = 1.0   # relative weight of spike latent in ctx_sim
    lambda_lfp_ctx: float = 1.0     # relative weight of LFP latent in ctx_sim
                                    # ignored if lfp not passed to train_unit
