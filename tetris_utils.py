"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

TODO: add documentation here
Order of the five outputs: left, right, up, down, idle
"""

from tetris_engine import *
import torch
from collections import deque
from collections import namedtuple

def convert_gamestate_to_tensor(gamestate : GameState):
    width, height = gamestate.width, gamestate.height
    N = width * height
    # first N entries represent the locked-in board
    # next N entries represent the current piece
    output = torch.zeros((2 * N,))
    for i in range(width):
        for j in range(height):
            if gamestate.game_board[i][j] > 0:
                output[i * height + j] = 1
            elif gamestate.game_board[i][j] < 0:
                output[N + i * height + j] = 1
    return output

# Replay memory, reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
