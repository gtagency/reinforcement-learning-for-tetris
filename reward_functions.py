"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Allows us to do "reward shaping" and provide smoother rewards to our agent
rather than just when it clears lines.
"""



## These constants aren't used right now, they were from the reward refactor.

GAME_END_REWARD = -200

# Applied when a line is cleared
LINE_CLEAR_REWARD = 10

# Applied to disincentivize higher stacks, so if current stack is 3 high,
# adds reward 3 * (PER_HIGHEST_PIECE_REWARD). Should be negative.
PER_HIGHEST_PIECE_REWARD = -5

# Applied for each "isolated hole" present in the stack so far.
PER_ISOLATED_HOLE_REWARD = -3

# Applied whenever a new piece is received
NEW_PIECE_REWARD = -2


class RewardFunction:
    def __init__(self):
        pass

    def update_and_get_reward(self, state, action):
        """Calls {state.update} with {action} and returns the reward."""
        lines_cleared = state.update(action)
        return lines_cleared


class LinesClearedReward(RewardFunction):
    """The base class effectively already does this."""
    pass


class LinesClearedMultiplierReward(RewardFunction):
    def update_and_get_reward(self, state, action):
        lines_cleared = state.update(action)
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 1
        elif lines_cleared == 2:
            return 3
        elif lines_cleared == 3:
            return 6
        elif lines_cleared == 4:
            return 10


### old code from state refactor: consider using later?
##    def get_reward(self):
##        reward = self.reward
##        for row in range(self.height-1, -1, -1):
##            if any(self.game_board[i][row] > 0 for i in range(self.width)):
##                # something is on row i, apply penalty
##                reward += PER_HIGHEST_PIECE_REWARD * (row + 1)
##                break
##        for i in range(1, self.width - 1):
##            for j in range(1, self.height - 1):
##                if self.game_board[i][j] == 0:
##                    if all(self.game_board[i+dx][j+dy] > 0 for dx, dy in
##                           ((1,0),(0,1),(-1,0),(0,-1))):
##                        reward += PER_ISOLATED_HOLE_REWARD
##        return reward
