"""
Authors: Sharay Gao, Ori Yoked, Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

This file provides a UI that can be used to play the Tetris game (using the
engine in {tetris_engine.py}), and possibly in the future to watch our agents
play the game. Uses pygame and supports arrow keys and WASD controls.
"""

from random import Random
import pygame
import sys
from tetris_engine import *
from reward_functions import *
from game_agent import *

pygame.init()

game = GameState()

# Sharay: some basic information, makes adapting the program easier
CELL_SIZE = 30
GAME_WIDTH = game.width
GAME_HEIGHT = game.height
GAME_TICK_DELAY = 100

IS_KEYBOARD_MODE = False
AGENT_TYPE = ModelAgent

screen = pygame.display.set_mode((CELL_SIZE * GAME_WIDTH, CELL_SIZE * GAME_HEIGHT))

# Sharay: Actions currently supported
action_lookup = {
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_UP: Action.ROTATE_CW,
    pygame.K_DOWN: Action.ROTATE_CCW,
    pygame.K_a: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_w: Action.ROTATE_CW,
    pygame.K_s: Action.ROTATE_CCW,
    pygame.K_r: Action.RESET
}

# Ori - Colors:
COLORS = [
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 255, 0),    # Green
    (255, 53, 184)  # Pink
]

LOCKED_COLORS = [
    (250, 249, 223),
    (223, 185, 208),
    (163, 220, 228),
    (255, 203, 165),
    (167, 183, 222),
    (206, 240, 161),
    (244, 204, 203)
]

# Ori - Game Over message:
font = pygame.font.Font('freesansbold.ttf', 15)

if not IS_KEYBOARD_MODE:
    agent = AGENT_TYPE()

lines_cleared = 0

agent = ModelAgent(torch.load("./checkpoint-141.pt"))
#agent = BruteAgent2(depth=2, reward_func=HeightPenaltyReward(multiplier=0.1))

while True:
    # Sharay: this just waits a bit before running, temporary
    pygame.time.wait(GAME_TICK_DELAY)

    action = Action.IDLE
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key in action_lookup:
                action = action_lookup[event.key]

    # Sharay: Updates the game with the key action
    if IS_KEYBOARD_MODE:
        lines_cleared += game.update(action)
    else:
        lines_cleared += game.update(agent.get_move(game))

    board = game.game_board

    # Sharay and Ori: alters colors in the board
    screen.fill((0, 0, 0))
    if game.stop:
        text = font.render("Game Over - Score: " + str(lines_cleared) + " - R to restart", True, (255, 255, 255), (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (150, 100)
        screen.blit(text, textRect)
    else:
        for i in range(len(board)):
            for j in range(len(board[0])):
                # Ori - change color when locked
                if board[i][j] > 0:
                    pygame.draw.rect(screen, LOCKED_COLORS[board[i][j] - 1],
                                     pygame.Rect(CELL_SIZE * i,
                                                 CELL_SIZE * (GAME_HEIGHT - j - 1),
                                                 CELL_SIZE,
                                                 CELL_SIZE))
                elif board[i][j] < 0:
                    pygame.draw.rect(screen, COLORS[(board[i][j] * -1) - 1],
                                     pygame.Rect(CELL_SIZE * i,
                                                 CELL_SIZE * (GAME_HEIGHT - j - 1),
                                                 CELL_SIZE,
                                                 CELL_SIZE))


    pygame.display.flip()
