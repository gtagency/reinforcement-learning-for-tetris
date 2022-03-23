# This class contains game agents
from tetris_engine import *
from placeholder_pytorch_model import *
from tetris_utils import *
from reward_functions import *
import torch
import random
#from graphicsUtils import keys_waiting
#from graphicsUtils import keys_pressed

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self):
        pass

    def get_move(self, state):
        pass

class RandomAgent(Agent):
    def __init__(self):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]

    def get_move(self, state):
        if state.stop:
            return Action.RESET
        else:
            return random.choice(self.actions)


import copy
import itertools
import random

class BruteAgent2(Agent):
    def __init__(self, depth=3, reward_func=None):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        self.depth = depth
        if reward_func is None:
            reward_func = LinesClearedReward()
        self.reward_func = reward_func

    def get_move(self, state):
        if state.stop:
            return Action.RESET
        best_reward = float("-inf")
        best_moves = []
        for move in self.actions:
            state_copy = copy.deepcopy(state)
            base_reward = self.reward_func.update_and_get_reward(state_copy, move)
            sub_best_reward = float("-inf")
            for action_sequence in itertools.product(*[self.actions for _ in range(self.depth - 1)]):
                this_reward = base_reward
                state_sub_copy = copy.deepcopy(state_copy)
                for a in action_sequence:
                    this_reward += self.reward_func.update_and_get_reward(state_sub_copy, a)
                sub_best_reward = max(sub_best_reward, this_reward)
            if sub_best_reward == best_reward:
                best_moves.append(move)
            elif sub_best_reward > best_reward:
                best_moves = [move]
                best_reward = sub_best_reward
        return random.choice(best_moves)
            

class BruteAgent(Agent):
    def __init__(self, depth=3):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        self.depth = depth
        self.bias = 0.5

    def get_move(self, state):
        if state.stop:
            return Action.RESET
        # bias does a random walk between 0 and 1
        #self.bias = max(0, min(1, self.bias + 0.1 * (random.random() - 0.5)))
        
        best_move_sequence = None
        best_reward = float("-inf")
        for action_sequence in itertools.product(*[self.actions for _ in range(self.depth)]):
            state_copy = copy.deepcopy(state)
            for a in action_sequence:
                state_copy.update(a)
            reward = state_copy.get_reward()
            for a in action_sequence:
                if a == Action.LEFT:
                    reward += 0.1 * (self.bias - 0.5)
                elif a == Action.RIGHT:
                    reward -= 0.1 * (self.bias - 0.5)
            if reward > best_reward:
                best_reward = reward
                best_move_sequence = action_sequence
        return best_move_sequence[0]

class ModelAgent(Agent):
    def __init__(self, model = None, epsilon = 0):
        if model is None:
            model = PlaceholderModel()
        self.model = model
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        # reduced frequency of rotations (somewhat arbitrarily..)
        self.random_actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.IDLE, Action.LEFT, Action.RIGHT,
                               Action.ROTATE_CW, Action.ROTATE_CCW]
        self.epsilon = epsilon

    def get_move(self, state):
        if state.stop:
            return Action.RESET
        elif random.random() < self.epsilon:
            return random.choice(self.random_actions)
        else:
            action = self.actions[torch.argmax(self.model(convert_gamestate_to_tensor(state)))]
            return action
