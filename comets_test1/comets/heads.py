# comets/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityHead(nn.Module):
    """No projection, just normalization."""
    def forward(self, x):
        return F.normalize(x, dim=-1)

class LinearHead(nn.Module):
    """Simple linear projection + normalization."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)

class OneLayerMLP(nn.Module):
    """Linear + LeakyReLU + normalization."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return F.normalize(F.leaky_relu(self.fc(x)), dim=-1)

class TwoLayerMLP(nn.Module):
    """Two-layer MLP (hidden → out) with ReLU in between."""
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return F.normalize(self.fc2(h), dim=-1)

def make_head(in_dim: int, out_dim: int, head_cls):
    """
    Instantiate the given projection head class/callable.
    IdentityHead takes no (in_dim, out_dim) args; others do.
    """
    if head_cls is IdentityHead:
        return head_cls()
    return head_cls(in_dim, out_dim)
