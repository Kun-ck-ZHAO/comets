# comets/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# All encoders take input [B, T, C] and output [B, D]

# -------------------------
# 1D Conv Encoder
# -------------------------
@dataclass
class Conv1DEncoderConfig:
    in_channels: int
    emb_dim: int = 32
    hidden: int = 64
    dropout: float = 0.0
    k1: int = 5
    k2: int = 3

class Conv1DEncoder(nn.Module):
    def __init__(self, cfg: Conv1DEncoderConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.hidden, kernel_size=cfg.k1, padding=cfg.k1//2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=cfg.k2, padding=cfg.k2//2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.hidden, cfg.emb_dim)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)

# -------------------------
# Skipped CNN Encoder (Residual 1D convs)
# -------------------------
@dataclass
class SkippedConvEncoderConfig:
    in_channels: int
    emb_dim: int = 32
    hidden: int = 64
    num_blocks: int = 3
    dropout: float = 0.0
    k: int = 3

class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.act = nn.GELU()
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.do(out)
        return x + out

class SkippedConvEncoder(nn.Module):
    """CEBRA-like residual 1D conv encoder."""
    def __init__(self, cfg: SkippedConvEncoderConfig):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.hidden, kernel_size=2, padding=0),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(cfg.hidden, cfg.k, cfg.dropout)
            for _ in range(cfg.num_blocks)
        ])
        self.final = nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.hidden, cfg.emb_dim)

    def forward(self, x):
        x = x.transpose(1, 2)          # [B,T,C] -> [B,C,T]
        h = self.entry(x)
        h = self.blocks(h)
        h = self.final(h)
        h = self.pool(h).squeeze(-1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)
    

class CEBRASKipEncoder(nn.Module):
    """Exactly matches CEBRA's offset10-model (skip CNN)."""
    def __init__(self, cfg: SkippedConvEncoderConfig):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.hidden, kernel_size=2, padding=0),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(cfg.hidden, cfg.k, cfg.dropout)
            for _ in range(cfg.num_blocks)
        ])
        # key change: final conv goes directly to emb_dim
        self.final = nn.Conv1d(cfg.hidden, cfg.emb_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.entry(x)
        h = self.blocks(h)
        h = self.final(h)
        h = self.pool(h).squeeze(-1)
        return F.normalize(h, dim=-1)

# -------------------------
# 2D Conv Encoder
# -------------------------
@dataclass
class Conv2DEncoderConfig:
    in_channels: int
    emb_dim: int = 32
    hidden: int = 32
    dropout: float = 0.0
    k: int = 3

class Conv2DEncoder(nn.Module):
    def __init__(self, cfg: Conv2DEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, cfg.hidden, kernel_size=cfg.k, padding=cfg.k//2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(cfg.hidden, cfg.hidden, kernel_size=cfg.k, padding=cfg.k//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(cfg.hidden, cfg.emb_dim)

    def forward(self, x):
        # x: [B, T, C] -> [B, 1, T, C]
        x = x.unsqueeze(1)
        h = self.net(x).squeeze(-1).squeeze(-1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)

# -------------------------
# MLP Encoder
# -------------------------
@dataclass
class MLPEncoderConfig:
    in_channels: int
    emb_dim: int = 32
    hidden: int = 128
    dropout: float = 0.0

class MLPEncoder(nn.Module):
    def __init__(self, cfg: MLPEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.in_channels, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.emb_dim),
        )
    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T*C)
        z = self.net(x)
        return F.normalize(z, dim=-1)

# -------------------------
# GRU Encoder
# -------------------------
@dataclass
class GRUEncoderConfig:
    in_channels: int
    emb_dim: int = 32
    hidden: int = 64
    num_layers: int = 1
    dropout: float = 0.0

class GRUEncoder(nn.Module):
    def __init__(self, cfg: GRUEncoderConfig):
        super().__init__()
        self.gru = nn.GRU(input_size=cfg.in_channels, hidden_size=cfg.hidden,
                          num_layers=cfg.num_layers, batch_first=True)
        self.fc = nn.Linear(cfg.hidden, cfg.emb_dim)
        self.do = nn.Dropout(cfg.dropout)
    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        return F.normalize(self.fc(self.do(h)), dim=-1)
