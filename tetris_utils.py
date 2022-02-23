"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

TODO: add documentation here
"""

from tetris_engine import *
import torch


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
