import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, use_dropout=False):
        super(Net, self).__init__()

        # input <- 400-dimensional vector (corresponding to 0s and 1s with the state)
        # decrease the size of the output by factors of 2^k for each layer (right now k = 2)
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, 25)
        # Output <- vector of the same dimension as the number of actions (where each component represents the q-value for that action)
        # how to get the number of actions?
        # guess: self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW] (from training_loop.py)
        self.fc3 = nn.Linear(25, 5)
        self.drop = nn.Dropout(p=0.3, inplace=False)
        self.use_dropout = use_dropout


    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        
        output = x # no activation function
        return output


        

