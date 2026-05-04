# comets/__init__.py

from .configs import ModalityConfig, SamplingConfig, TrainingConfig
from .encoders import (
    Conv1DEncoderConfig, Conv2DEncoderConfig, MLPEncoderConfig, GRUEncoderConfig,
    Conv1DEncoder,       Conv2DEncoder,       MLPEncoder,       GRUEncoder,
    SkippedConvEncoderConfig, SkippedConvEncoder,
)
from .heads import (
    IdentityHead, LinearHead, OneLayerMLP, TwoLayerMLP, make_head,
)
from .infonce import InfoNCELoss
from .sampler import InBatchSampler, _gather_windows
from .core import train, CoMETs
from .viz import plotemb2

__all__ = [
    # configs
    "ModalityConfig", "SamplingConfig", "TrainingConfig",

    # encoders + configs
    "Conv1DEncoderConfig", "Conv2DEncoderConfig", "MLPEncoderConfig", "GRUEncoderConfig",
    "Conv1DEncoder",       "Conv2DEncoder",       "MLPEncoder",       "GRUEncoder",
    "SkippedConvEncoderConfig", "SkippedConvEncoder", "CEBRASKipEncoder"

    # heads
    "IdentityHead", "LinearHead", "OneLayerMLP", "TwoLayerMLP", "make_head",

    # loss / sampler / train
    "InfoNCELoss", "InBatchSampler", "_gather_windows", "train", "CoMETs",

    # viz
    "plotemb2",
]
