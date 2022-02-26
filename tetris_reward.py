#Collaborators: Max Boonbandansook
#Primary goal of this script is to run pygame/tetris in scripts (headless mode no windows)
#Limitation: halts the computation once reward gets below -100

import pygame
import sys
import os
from tetris_engine import *
from game_agent import *

os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()

game = GameState()

rewards = []

# Sharay: some basic information, makes adapting the program easier
CELL_SIZE = 30
GAME_WIDTH = game.width
GAME_HEIGHT = game.height
GAME_TICK_DELAY = 1

AGENT_TYPE = BruteAgent

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

agent = AGENT_TYPE()

reward = 0
while reward > -100:
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

    game.update(agent.get_move(game))

    board = game.game_board
    reward = game.get_reward()
    rewards.append(reward)

print(sum(rewards)/len(rewards))
