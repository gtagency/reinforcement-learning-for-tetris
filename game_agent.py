# This class contains game agents
from tetris_engine import *
import random
class randomBot:
    def __init__(self):
        self.action_list = Action()
    def action(self):
        num = random.randrange(0, 5)
        return self.action_list.num

