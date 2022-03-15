"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Primary goal of this file is to have a training loop 
that updates the weights of the model based on the gradient.
"""

import torch
import torch.nn as nn
from tetris_utils import *
from tetris_engine import *
from reward_functions import *

import random

class TrainingLoop:
    def __init__(self, reward_func=None):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        self.game = GameState()
        self.replay_memory = ReplayMemory()
        # **Question for Neil**: How do we keep track of the Q_value for each Transition state
        self.Q_value = []
        self.batch_size = 5
        if reward_func is None:
            reward_func = LinesClearedReward()
        self.reward_func = reward_func

    def loop(self, epochs, state):

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # select an action
            action = random.choice(self.actions)
            # or based on Q-value

            # need to execute action

            # convert it to tensor (using the method in utils)
            old_state = convert_gamestate_to_tensor(self.game.game_board)

            reward = self.reward_func.update_and_get_reward(game, action)

            new_state = convert_gamestate_to_tensor(self.game.game_board)

            #Store the transition in the replay memory
            # if the action increases the reward, it is a good action so we wanna
            # do change in reward instead of the new reward
            self.replay_memory.push(old_state, action, new_state, reward)
            # image from the replay memory (x_t+1),
            # s_t+1 = s_t, a_t, x_t+1

            # store transition in replay memory
            # call ReplayMemory.push()

            # sample random minibatch 
            
            if self.batch_size >= len(self.replay_memory):
                samples = self.replay_memory.sample(self.batch_size)
                # call ReplayMemory.sample()
                old_states = [sample.state for sample in samples]
                old_states = convert_gamestate_to_tensor()
                # take out all the old state variables
                y = DQN()
                
                # set y (by calling the Q-network)

                # gradient descent on (y - Q_value)




