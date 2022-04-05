"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Primary goal of this file is to have a training loop 
that updates the weights of the model based on the gradient.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tetris_utils import *
from tetris_engine import *
from reward_functions import *
from model import *
from game_agent import *

import random

class TrainingLoop:
    def __init__(self, reward_func=None):
        self.replay_memory = ReplayMemory(capacity=200000)
        self.Q_value = []
        self.batch_size = 32
        self.time_step = 1000
        self.gamma = 0.90
        self.C = 10
        if reward_func is None:
            reward_func = LinesClearedReward()
        self.reward_func = reward_func

    def loop(self, epochs):
        self.DQN = DQN = Net()
        model_agent = ModelAgent(model=DQN, epsilon=1)
        optimizer = optim.Adam(DQN.parameters(), lr=0.01)
        self.old_model = self.DQN
        for epoch in range(epochs):
            print(f"Episode {epoch+1}")

            new_epsilon = max(min(1 - ((epoch - 15) / 200), 1), 0.075)
            if new_epsilon != model_agent.epsilon:
                model_agent.epsilon = new_epsilon
                print("updating epsilon to", new_epsilon)
            
            game = GameState()

            total_loss = 0.0
            total_reward = 0.0

            #if epoch == 3:
            #    print("switching to epsilon-greedy")
            #    model_agent = ModelAgent(model=DQN, epsilon=1)
            
            for t in range(self.time_step):
                
                
                # select an action
                action = model_agent.get_move(game)

                # need to execute action

                # convert it to tensor (using the method in utils)
                old_state = convert_gamestate_to_tensor(game)

                reward = self.reward_func.update_and_get_reward(game, action)
                total_reward += reward

                new_state = convert_gamestate_to_tensor(game)

                #Store the transition in the replay memory
                # if the action increases the reward, it is a good action so we wanna
                # do change in reward instead of the new reward
                if action != Action.RESET:
                    # replay memory doesn't seem to store enough line-clears, hopefully
                    # this will increase good:bad ratio in the memory
                    if reward > 0 or random.random() < 0.05:
                        # dummy = torch.reshape(new_state, (2, -1))
                        # old = torch.reshape(dummy[0,:], (game.width, game.height))
                        # print(old)
                        # print("this is one\n")
                        # is_terminal = torch.sum(old[:,game.height-1]) > 0
                        is_terminal = game.stop
                        # print(is_terminal)
                        self.replay_memory.push(old_state, action, new_state, reward, is_terminal)
                # image from the replay memory (x_t+1),
                # s_t+1 = s_t, a_t, x_t+1

                # store transition in replay memory
                # call ReplayMemory.push()

                # sample random minibatch 
                
                if self.batch_size <= len(self.replay_memory):
                    samples = self.replay_memory.sample(self.batch_size)
                    # call ReplayMemory.sample()
                    old_states = []
                    new_states = []
                    for sample in samples:

                        old_states.append(sample.state)
                        new_states.append(sample.next_state)
                    
                    old_state_tensors = torch.zeros(self.batch_size, 400)
                    new_state_tensors = torch.zeros(self.batch_size, 400)
                    actions = torch.zeros(self.batch_size, dtype=torch.long)

                    for i in range(self.batch_size):
                        old_state_tensors[i] = old_states[i]
                        new_state_tensors[i] = new_states[i]
                        actions[i] = samples[i].action.value

                    # take out all the old state variables

                    # batch size for the number of rows
                    with torch.no_grad():
                        y = torch.zeros(self.batch_size)
                        outcomes = DQN(new_state_tensors)
                        for i in range(self.batch_size):
                            y[i] = samples[i].reward + self.gamma*torch.max(outcomes[i])
                    
                    optimizer.zero_grad()

                    pred = self.old_model(old_state_tensors)[list(range(self.batch_size)), actions]
                    # pred here is now just (batch_size,)
                    loss = ((y - pred) ** 2).mean()
                    loss.backward() # produces gradients that are stored

                    optimizer.step()

                    with torch.no_grad():
                        total_loss += loss.item()


                if t % self.C == 0:
                    self.old_model = DQN

                    #print("loss =",loss)

                    # set y (by calling the Q-network)

                    # gradient descent on (y - Q_value)

            print("max reward in replay memory: %0.02f" % max(x.reward for x in self.replay_memory.memory),
                  "(size is %d)" % len(self.replay_memory))
            print("average loss: %0.05f" % (total_loss / self.time_step))
            print("average reward: %0.05f" % (total_reward / self.time_step))
            print()

            if (epoch + 1) % 10 == 0:
                model_path = "model-epoch-%03d.pt" % (epoch + 1)
                print(f"saving model to '{model_path}'")
                torch.save(DQN, model_path)
                print()

loop = TrainingLoop(reward_func=multipleRewards())
loop.loop(1000)

