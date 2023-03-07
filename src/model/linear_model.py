from torch import nn as nn


class LinearModel(nn.Module):
    def __init__(self, past_window, n_features, horizon):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=past_window * n_features, out_features=horizon)
        )

    def forward(self, x):
        return self.model(x)
