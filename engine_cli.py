# Authors: Neil Thistlethwaite
# (add your name above if you contribute to this file)
# The Agency, Reinforcement Learning for Tetris

# This file is mostly meant for testing the engine with a minimal interface

from tetris_engine import *

def print_board(gamestate):
    print()
    print("N/A")
    print()

game = GameState()

action_lookup = {
    "a": Action.LEFT,
    "d": Action.RIGHT,
    "w": Action.ROTATE_CW,
    "s": Action.ROTATE_CCW
    }

while True:
    print_board(game)
    action_input = input("> ")
    action = (action_lookup[action_input] if action_input in action_lookup \
             else Action.IDLE)

    game.update(action)
             
