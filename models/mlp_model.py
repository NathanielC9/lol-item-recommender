import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)