import torch
import torch.nn as nn

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


if __name__ == "__main__":
    model = PlaceholderModel()
    example_input = 1.0 * (torch.randn((1, 400)) > 0)
    print(model(example_input))

# Task for Ori:
# write an agent class that acts based on "model"
# so when asked to choose an action, it converts the gamestate
# to a tensor (see {tetris_utils.py}), and then feeds it through
# the model, and then picks the action corresponding to the largest
# output.
