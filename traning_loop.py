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
import random
class TrainingLoop:

    def __init__(self):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        self.game = GameState()
        self.replay_memory = ReplayMemory()
        # **Question for Neil**: How do we keep track of the Q_value for each Transition state
        self.Q_value = []
        self.batch_size = 5
        pass

    def loop(self, ephochs, state):
        epochs = 2

        for epoch in range(epochs):

        
            print("\nStart of epoch %d" % (epoch,))

            # select an action
            action = random.choice(self.actions)
            # or based on Q-value

            # need to execute action

            # need to find the reward
            oldReward = self.game.get_reward()

            # convert it to tensor (using the method in utils)
            oldState = convert_gamestate_to_tensor(self.game.game_board)

            self.game.update(action)
            
            newReward = self.game.get_reward()

            newState = convert_gamestate_to_tensor(self.game.game_board)

            changeInReward = newReward-oldReward

            #Store the transition in the replay memory
            # if the action increases the reward, it is a good action so we wanna
            # do change in reward instead of the new reward
            self.replay_memory.push(oldState, action, newState, changeInReward)
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




