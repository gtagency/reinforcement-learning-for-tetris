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
        end_game_penalty = -10000 * state.stop
        # print(end_game_penalty)
        return lines_cleared + end_game_penalty


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

class HeightPenaltyReward(RewardFunction):
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def update_and_get_reward(self, state, action):
        lines_cleared = state.update(action)
        # will add a penalty between [0, -multiplier]
        # depending on how tall the stack is
        highest_row = 0
        for row in range(state.height-1, -1, -1):
            if any(state.game_board[i][row] > 0 for i in range(state.width)):
                highest_row = row + 1
                break
        reward = lines_cleared - self.multiplier * (highest_row / state.height)
        end_game_penalty = -10 * state.stop
        # print(end_game_penalty)
        return reward + end_game_penalty

# http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
class multipleRewards(RewardFunction):
    def __init__(self, height_mult=-0.5, hole_mult=-0.36, lineclear_mult=0.8):
        self.height_mult = height_mult
        self.hole_mult = hole_mult
        self.lineclear_mult = lineclear_mult
    def update_and_get_reward(self, state, action):
        parent_reward = super().update_and_get_reward(state, action)
        lines_cleared = state.update(action)
        height_penalty = 0
        hole_penalty = 0
        piece_height_sum = 0
        top_row = 0
        for i in range(state.width):
            max_height = 0
            j = 0
            while j < state.height:
                if state.game_board[i][j] > 0:
                    max_height = j
                # if state.game_board[i][j] < 0:
                #     piece_height_sum += j
                j += 1
            height_penalty += 1.3**(max_height + 1) 
            top_row = max(top_row, max_height)
            # hole_penalty += sum(state.game_board[i][h] == 0 for h in range(0, max_height))
            # if hole_count > max_height:
            #     print(state.get_current_board())
            # print("max height and hole count: ", max_height, hole_count)
            
        # print("hole penalty:", hole_penalty)
        # print("height penalty", height_penalty)
        # height_penalty = highest_row * self.height_mult * 20

        height_penalty *= self.height_mult
        for j in range(top_row):
            for i in range(state.width):
                hole_penalty += state.game_board[i][j] == 0
        hole_penalty *= self.hole_mult
        
        return parent_reward + lines_cleared*self.lineclear_mult+height_penalty+hole_penalty 

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
