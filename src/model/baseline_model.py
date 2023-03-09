import torch
from torch import nn as nn


class BaselineModel(nn.Module):
    def __init__(self, horizon, target_idx):
        super().__init__()
        self.horizon = horizon
        self.target_idx = target_idx

    def forward(self, x: torch.tensor):
        vals = x[:, -1, self.target_idx]
        return vals.tile(self.horizon).reshape(self.horizon, -1).T
