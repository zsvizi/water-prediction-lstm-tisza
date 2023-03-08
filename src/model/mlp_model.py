from torch import nn as nn


class MLPModel(nn.Module):
    def __init__(self, past_window, n_features, horizon):
        super().__init__()
        hidden_size_1 = 256
        hidden_size_2 = 128
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=past_window * n_features, out_features=hidden_size_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size_2, out_features=horizon)
        )

    def forward(self, x):
        return self.model(x)
