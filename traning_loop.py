"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Primary goal of this file is to have a training loop 
that updates the weights of the model based on the gradient.
"""

import torch
import torch.nn as nn
from tetris_engine import *

class TrainingLoop:

    def __init__(self):
        pass

    def loop(self, ephochs state):
        epochs = 2

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
