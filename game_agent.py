# This class contains game agents
from tetris_engine import *
import random
from graphicsUtils import keys_waiting
from graphicsUtils import keys_pressed

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self):
        pass

    def get_action(self, state):
        pass
        
class KeyboardAgent(Agent):
    """
    An agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    WEST_KEY  = 'a'
    EAST_KEY  = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'

    def __init__(self):

        self.lastMove = Action.IDLE
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]
        self.keys = []

    def get_action(self, state):
        
        keys = list(keys_waiting()) + list(keys_pressed())
        if keys != []:
            self.keys = keys

        move = self.getMove()

        # if move == Action.IDLE:
        #     # Try to move in the same direction as before
        #     if self.lastMove in self.actions:
        #         move = self.lastMove

        # if move not in self.actions:
        #     move = random.choice(self.actions)

        self.lastMove = move
        return move

    def get_move(self):
        move = Action.IDLE
        if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Action.LEFT in self.actions: move = Action.LEFT
        if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Action.RIGHT in self.actions: move = Action.RIGHT
        if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Action.ROTATE_CW in self.actions: move = Action.ROTATE_CW
        if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Action.ROTATE_CCW in self.actions: move = Action.ROTATE_CCW
        return move

class RandomAgent(Agent):
    def __init__(self):
        self.actions = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]

    def get_move(self):
        num = random.randrange(0, 5)
        return self.action[num]

