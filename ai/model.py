
import torch
import torch.nn as nn


class SnakeNet(nn.Module):
    """Red neuronal que estima el valor de cada acción."""

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state):
        return self.network(state)
