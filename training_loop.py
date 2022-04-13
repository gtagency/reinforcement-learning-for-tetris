"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Primary goal of this file is to have a training loop 
that updates the weights of the model based on the gradient.
"""
from audioop import mul
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tetris_utils import *
from tetris_engine import *
from reward_functions import *
from model import *
from game_agent import *

import random
import os

class TrainingLoop:
    # epsilon schedule should be a list of:
    # [(time_step, new_epsilon), (time_step, new_epsilon), ...]
    # and it will linearly interpolate between
    # e.g. [(0, 0.8), (1000, 0.8), (2000, 0.5), (5000, 0.5)]
    # means that epsilon should start at 0.8, stay constant until 1000,
    # then decrease to 0.5 between 1000 and 2000, then stay constant
    # until 5000.
    def __init__(self, save_dir=None, reward_func=None, epsilon_schedule=None):
        self.replay_memory = ReplayMemory(capacity=200000)
        self.Q_value = []
        self.batch_size = 32
        self.training_frames = 10_000_000
        self.print_frequency = 10_000
        self.model_checkpoint_frequency = 100_000
        self.replay_start_size = 50_000
        self.gamma = 0.99
        self.target_network_update_frequency = 10_000

        if save_dir is None:
            save_dir = "." # save to same folder by default
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if epsilon_schedule is None:
            # default to complete linear
            epsilon_schedule = [(0,1.0), (self.training_frames,0.0)]
        self.epsilon_schedule = epsilon_schedule

        if reward_func is None:
            reward_func = LinesClearedReward()
        self.reward_func = reward_func

    def determine_epsilon_from_schedule(self, current_frame_count):
        # hard coded to use deepmind paper value(s)
        # TODO: make this use {self.epsilon_schedule}
        return max(1 - 0.9 * (current_frame_count / 1_000_000), 0.1)

    def loop(self, epochs):
        self.DQN = DQN = Net()
        model_agent = ModelAgent(model=DQN, epsilon=1)
        #optimizer = optim.Adam(DQN.parameters(), lr=0.0001)
        optimizer = optim.RMSprop(DQN.parameters(), lr=0.00025, momentum=0.95, alpha=0.95, eps=0.01)
        self.old_model = Net()
        episode_count = 0
        total_frame_count = 0
        last_time = time.time()

        total_avg_episode_reward = 0.0
        total_episode_length = 0
        num_episodes = 0

        total_loss = 0.0
        total_train_steps = 0
        
        while total_frame_count < self.training_frames:
            #print(f"Episode {epoch+1}")
            episode_count += 1

            # Between episodes, update epsilon

            #new_epsilon = max(min(1 - ((epoch - 15) / 200), 1), 0.075)
            #if new_epsilon != model_agent.epsilon:
            #    model_agent.epsilon = new_epsilon
            #    print("updating epsilon to", new_epsilon)
            
            game = GameState()
            
            total_episode_reward = 0.0

            #if epoch == 3:
            #    print("switching to epsilon-greedy")
            #    model_agent = ModelAgent(model=DQN, epsilon=1)

            # do one full episode
            game_timestep_counter = 0
            while not game.stop and game_timestep_counter < 1000:
                game_timestep_counter += 1
                total_frame_count += 1

                # check if we should print
                if total_frame_count % self.print_frequency == 0:
                    new_time = time.time()
                    time_taken = new_time - last_time

                    average_loss = f"{total_loss/total_train_steps:.4f}" if total_train_steps > 0 \
                                   else "N/A"
                    
                    print(f"finished {total_frame_count} frames ({time_taken:.2f} seconds since last print)")
                    print(f"current epsilon: {model_agent.epsilon:.4f}")
                    print(f"average reward per episode: {total_avg_episode_reward/num_episodes:.2f}")
                    print(f"average episode length: {total_episode_length/num_episodes:.2f}")
                    print(f"average loss: {average_loss}")
                    print()
                    
                    # reset these so we only get averages between prints
                    total_avg_episode_reward = 0.0
                    total_episode_length = 0
                    num_episodes = 0
                    total_loss = 0
                    total_train_steps = 0
                    
                    last_time = new_time

                # check if we should save the model
                if total_frame_count % self.model_checkpoint_frequency == 0:
                    checkpoint_number = total_frame_count // self.model_checkpoint_frequency
                    model_path = os.path.join(self.save_dir, "checkpoint-%03d.pt" % checkpoint_number)
                    print(f"saving the model to '{model_path}'")
                    torch.save(DQN, model_path)
                    print("saved!")
                    

                # update epsilon if we have to
                new_epsilon = self.determine_epsilon_from_schedule(total_frame_count)
                model_agent.epsilon = new_epsilon

                
                # select an action
                action = model_agent.get_move(game)

                # need to execute action

                # convert it to tensor (using the method in utils)
                old_state = convert_gamestate_to_tensor(game)

                reward = self.reward_func.update_and_get_reward(game, action)
                total_episode_reward += reward

                new_state = convert_gamestate_to_tensor(game)

                #Store the transition in the replay memory
                # if the action increases the reward, it is a good action so we wanna
                # do change in reward instead of the new reward
                is_next_state_terminal = game.stop

                # replay memory doesn't seem to store enough line-clears, hopefully
                # this will increase good:bad ratio in the memory
                self.replay_memory.push(old_state, action, new_state, reward, is_next_state_terminal)

                # image from the replay memory (x_t+1),
                # s_t+1 = s_t, a_t, x_t+1

                # store transition in replay memory
                # call ReplayMemory.push()

                # sample random minibatch
                
                if len(self.replay_memory) >= self.replay_start_size:
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
                        outcomes = self.old_model(new_state_tensors)
                        for i in range(self.batch_size):
                            if samples[i].is_next_state_terminal:
                                y[i] = samples[i].reward
                            else:
                                y[i] = samples[i].reward + self.gamma*torch.max(outcomes[i])
                    
                    optimizer.zero_grad()

                    pred = DQN(old_state_tensors)[list(range(self.batch_size)), actions]
                    # pred here is now just (batch_size,)
                    loss = ((y - pred) ** 2).mean()
                    loss.backward() # produces gradients that are stored

                    optimizer.step()

                    with torch.no_grad():
                        total_loss += loss.item()
                        total_train_steps += 1


                if total_frame_count % self.target_network_update_frequency == 0:
                    self.old_model.load_state_dict(DQN.state_dict())

                    #print("loss =",loss)

                    # set y (by calling the Q-network)

                    # gradient descent on (y - Q_value)

            
            avg_episode_reward = total_episode_reward / game_timestep_counter

            total_episode_length += game_timestep_counter
            total_avg_episode_reward += avg_episode_reward
            num_episodes += 1
            
            #print("max reward in replay memory: %0.02f" % max(x.reward for x in self.replay_memory.memory),
            #      "(size is %d)" % len(self.replay_memory))
            #print("average loss: %0.05f" % (total_loss / game_timestep_counter))
            #print("average reward: %0.05f" % (total_reward / game_timestep_counter))
            #print()

            #if (epoch + 1) % 10 == 0:
            #    model_path = "model-epoch-%03d.pt" % (epoch + 1)
            #    print(f"saving model to '{model_path}'")
            #    torch.save(self.old_model, model_path)
            #    print()

loop = TrainingLoop(save_dir="run-4-12", reward_func=HeightPenaltyReward(multiplier=0.1, game_over_penalty=10))
#loop = TrainingLoop(reward_func=HeightPenaltyReward(multiplier=0.1, game_over_penalty=1000))

loop.loop(1000)

