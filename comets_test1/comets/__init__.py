# comets/__init__.py

# --- configs ---
from .configs import (
    ModalityConfig,
    SamplingConfig,
    LatentTrainingConfig,
    TrainingConfig,         # backward-compat alias for LatentTrainingConfig
    UnitTrainingConfig,
)

# --- encoders + configs ---
from .encoders import (
    Conv1DEncoderConfig, Conv2DEncoderConfig, MLPEncoderConfig, GRUEncoderConfig,
    Conv1DEncoder,       Conv2DEncoder,       MLPEncoder,       GRUEncoder,
    SkippedConvEncoderConfig, SkippedConvEncoder, CEBRASKipEncoder,
)

# --- projection heads ---
from .heads import (
    IdentityHead, LinearHead, OneLayerMLP, TwoLayerMLP, make_head,
)

# --- loss ---
from .infonce import InfoNCELoss

# --- samplers ---
from .sampler import LatentSampler, InBatchSampler, UnitSampler, _gather_windows

# --- Stage 1: latent encoder ---
from .core import CoMETsModel, train_latent, CoMETs, train

# --- Stage 2: unit encoder ---
from .unit_encoder import UnitEncoderConfig, UnitEncoder
from .unit_loss import coact_loss, ctx_loss, UnitLoss
from .unit_train import train_unit

# --- evaluation ---
from .unit_eval import (
    ratemap_correlation,
    si_gradient_correlation,
    split_half_stability,
)

# --- viz ---
from .viz import plotemb2


__all__ = [
    # configs
    "ModalityConfig", "SamplingConfig",
    "LatentTrainingConfig", "TrainingConfig",
    "UnitTrainingConfig",

    # encoders
    "Conv1DEncoderConfig", "Conv2DEncoderConfig", "MLPEncoderConfig", "GRUEncoderConfig",
    "Conv1DEncoder",       "Conv2DEncoder",       "MLPEncoder",       "GRUEncoder",
    "SkippedConvEncoderConfig", "SkippedConvEncoder", "CEBRASKipEncoder",

    # heads
    "IdentityHead", "LinearHead", "OneLayerMLP", "TwoLayerMLP", "make_head",

    # Stage 1 loss / sampler / train
    "InfoNCELoss",
    "LatentSampler", "InBatchSampler", "_gather_windows",
    "CoMETsModel", "train_latent",
    "CoMETs", "train",

    # Stage 2 sampler
    "UnitSampler",

    # Stage 2 encoder
    "UnitEncoderConfig", "UnitEncoder",

    # Stage 2 loss / train
    "coact_loss", "ctx_loss", "UnitLoss",
    "train_unit",

    # evaluation
    "ratemap_correlation", "si_gradient_correlation",
    "split_half_stability",

    # viz
    "plotemb2",
]
