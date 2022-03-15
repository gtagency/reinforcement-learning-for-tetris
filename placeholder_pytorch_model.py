import torch
import torch.nn as nn
from tetris_engine import *

class PlaceholderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(400, 5)

    def forward(self, x):
        # x is going to be 400-dimensional
        # in pytorch terms: shape (N, 400)
        # want 5 actions, so output should be (N, 5)
        x = self.fc1(x)
        return x
