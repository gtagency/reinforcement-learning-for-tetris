# This class contains game agents
from tetris_engine import *
from placeholder_pytorch_model import *
from tetris_utils import *
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
    def __init__(self, model = None):
        if model is None:
            model = PlaceholderModel()
        self.model = model
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]

    def get_move(self, state, epsilon):
        if state.stop:
            return Action.RESET
        elif random.random() > epsilon:
            random.choice(self.actions)
        else:
            action = self.actions[torch.argmax(self.model(convert_gamestate_to_tensor(state)))]
            return action