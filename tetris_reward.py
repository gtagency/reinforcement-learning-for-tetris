#Collaborators: Max Boonbandansook
#Primary goal of this script is to run pygame/tetris in scripts (headless mode no windows)
#Limitation: halts the computation once reward gets below -100

import pygame
import sys
import os
from tetris_engine import *
from game_agent import *
import statistics

os.environ["SDL_VIDEODRIVER"] = "dummy"


def game_start(AGENT_TYPE):


    totalrewards = []

    for i in range(3):
        pygame.init()
        game = GameState()
        agent = AGENT_TYPE()

        rewards = []
        reward = 0
        while not game.stop:
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
        totalrewards.append(statistics.mean(rewards))
    return totalrewards

print("BruteAgent rewards",game_start(BruteAgent))
print("RandomAgent rewards",game_start(RandomAgent))

