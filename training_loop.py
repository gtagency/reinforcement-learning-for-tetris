"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Primary goal of this file is to have a training loop 
that updates the weights of the model based on the gradient.
"""
print("hi")
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_utils import *
from tetris_engine import *
from reward_functions import *
from model import *
from game_agent import *

import random
print("hiii")

class TrainingLoop:
    def __init__(self, reward_func=None):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        self.game = GameState()
        self.replay_memory = ReplayMemory()
        # **Question for Neil**: How do we keep track of the Q_value for each Transition state
        self.Q_value = []
        self.batch_size = 5
        self.time_step = 1000
        self.gamma = 0.99
        if reward_func is None:
            reward_func = LinesClearedReward()
        self.reward_func = reward_func

    def loop(self, epochs):
        print("loooppp")
        DQN = Net()
        model_agent = ModelAgent(model=DQN, epsilon=0.05)
        optimizer = optim.Adam(DQN.parameters(), lr=0.01)

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            game = GameState()
            for t in range(self.time_step):

                # select an action
                action = model_agent.get_move(game)

                # need to execute action

                # convert it to tensor (using the method in utils)
                old_state = convert_gamestate_to_tensor(self.game.game_board)

                reward = self.reward_func.update_and_get_reward(self.game, action)

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
                    
                    old_state_tensors = torch.zeros(self.batch_size, 400)
                    for i in range(self.batch_size):
                        old_state_tensors[i] = old_states[i]

                    # for each old state, there will be a corresponding new state
                    new_states = [sample.next_state for sample in samples]
                    
                    new_state_tensors = torch.zeros(self.batch_size, 400)
                    for i in range(self.batch_size):
                        new_state_tensors[i] = new_states[i]

                    actions = torch.zeros(self.batch_size)
                    for i in range(self.batch_size):
                        actions[i] = samples[i].action

                    # take out all the old state variables

                    # batch size for the number of rows
                    with torch.no_grad():
                        y = torch.zeros(self.batch_size)
                        outcomes = DQN(new_state_tensors)
                        for i in range(self.batch_size):
                            y[i] = samples[i].reward + self.gamma*torch.max(outcomes[i])
                    
                    optimizer.zero_grad()

                    pred = DQN(old_state_tensors)[list(range(self.batch_size)), action]
                    # pred here is now just (batch_size,)
                    loss = ((y - pred) ** 2).mean()
                    loss.backward() # produces gradients that are stored

                    optimizer.step()

                    print("loss =",loss)
                    
                    # set y (by calling the Q-network)

                    # gradient descent on (y - Q_value)

print("starting training loop...")

loop = TrainingLoop()
loop.loop(10)

