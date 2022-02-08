"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

DEPRECATED - this file was meant to allow initial testing of the Tetris engine
before we had a proper UI implemented. Now we do, use {tetris_ui.py} instead.
"""

from tetris_engine import *

def print_board(gamestate):
    print()
    print(gamestate.get_current_board())
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
             
