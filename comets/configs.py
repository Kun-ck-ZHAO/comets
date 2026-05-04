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
class TrainingConfig:
    epochs: int = 200
    lr: float = 3e-4
    temperature: float = 1.0
    weight_decay: float = 1e-5  # now matches CEBRA default
    proj_dim: int = 16
    beta: float = 0.5           # within-modality vs cross balance
    alpha_cross: float = 0.5    # cross-modality weighting
    head_cls: nn.Module = None  # defaults to TwoLayerMLP (set in train)
