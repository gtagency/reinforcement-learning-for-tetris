"""
Authors: Zhao Chen (Joe), Ori Yoked, Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Primary goal of this file is to allow us to simulate the game of Tetris. Each
"GameState" object will keep track of the current state of the game, and we'll
call its {update} method every game tick with inputs. This will also expose
methods for the UI and agent to see the game.
"""

from enum import Enum
import random

class Action(Enum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    # TODO-someday: change this to allow simultaneous movement and rotation?
    ROTATE_CW = 3
    ROTATE_CCW = 4


class GamePiece:
    # seven different shapes in tetris
    def __init__(self, board_width=10, board_height=20):
        self.shape_num = random.randrange(0, 7)
        self.width = board_width
        self.height = board_height
        self.shape = []
        self._initialize()
        
    def _initialize(self):
        #  x represents the shape
        width = self.width
        height = self.height
        if self.shape_num == 0:
            #  ----------
            #  ------x---
            #  ----xxx---
            #  ----------
            self.shape.append((int(width/2), int(height-2)))
            self.shape.append((int(width/2+1), int(height-2)))
            self.shape.append((int(width/2-1), int(height-2)))
            self.shape.append((int(width/2+1), int(height-1)))
        elif self.shape_num == 1:
            #  ----------
            #  ----xxxx--
            #  ----------
            #  ----------
            self.shape.append((int(width/2), int(height-1)))
            self.shape.append((int(width/2-1), int(height-1)))
            self.shape.append((int(width/2+1), int(height-1)))
            self.shape.append((int(width/2+2), int(height-1)))
        elif self.shape_num == 2:
            #  ----------
            #  ----xx----
            #  ----xx----
            #  ----------
            self.shape.append((int(width/2), int(height-2)))
            self.shape.append((int(width/2), int(height-1)))
            self.shape.append((int(width/2+1), int(height-1)))
            self.shape.append((int(width/2+1), int(height-2)))
        elif self.shape_num == 3:
            #  ----------
            #  ----x-----
            #  ----xxx---
            #  ----------
            self.shape.append((int(width/2), int(height-2)))
            self.shape.append((int(width/2-1), int(height-1)))
            self.shape.append((int(width/2-1), int(height-2)))
            self.shape.append((int(width/2+1), int(height-2)))
        elif self.shape_num == 4:
            #  ----------
            #  ----xx----
            #  -----xx---
            #  ----------
            self.shape.append((int(width/2), int(height-2)))
            self.shape.append((int(width/2), int(height-1)))
            self.shape.append((int(width/2-1), int(height-1)))
            self.shape.append((int(width/2+1), int(height-2)))
        elif self.shape_num == 5:
            #  ----------
            #  -----x----
            #  ----xxx---
            #  ----------
            self.shape.append((int(width/2), int(height-2)))
            self.shape.append((int(width/2-1), int(height-2)))
            self.shape.append((int(width/2), int(height-1)))
            self.shape.append((int(width/2+1), int(height-2)))
        else:
            #  ----------
            #  ----xx----
            #  ---xx-----
            #  ----------
            self.shape.append((int(width/2), int(height-2)))
            self.shape.append((int(width/2), int(height-1)))
            self.shape.append((int(width/2+1), int(height-1)))
            self.shape.append((int(width/2-1), int(height-2)))



class GameState:
    def __init__(self, width=10, height=20, init_val=0):
        # Joe: assume bottom left to be (0, 0), x coordinate goes to the right,
        #      y coordinate goes up
        self.width = width
        self.height = height
        self.curr_piece = None
        self.action_map = {
            Action.IDLE: (0, 0), 
            Action.LEFT: (-1, 0), Action.RIGHT: (1, 0), 
            Action.ROTATE_CW: (1, -1), Action.ROTATE_CCW: (-1, 1)
        }
        # {self.game_board} entries correspond to following cell states:
        #    0  empty cell
        #   +k  locked with piece of type k
        #   -k  holding current piece of type k
        self.game_board = [[init_val for col in range(height)] for row in range(width)]
        self.game_piece = None
        self._initialize_piece()
        self._fill_board(-1)

    def _initialize_piece(self):
        self.game_piece = GamePiece(board_width=self.width, board_height=self.height)
        self.curr_piece = self.game_piece.shape

    def _fill_board(self, val):
        for piece in self.curr_piece:
            self.game_board[piece[0]][piece[1]] = val * (self.game_piece.shape_num + 1)

    ## DEPRECATED
    def get_current_board(self):
        out = [['x' if self.game_board[row][col] else '.' for row in range(self.width)] for col in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def update(self, action):
        # convert action to a tuple(dx, dy)
        x, y = self.action_map[action]
        final_piece = []
        if (action == Action.ROTATE_CCW or action == Action.ROTATE_CW):
            if self.game_piece.shape_num == 2:
                self._gravity()
                return
            center_x, center_y = self.curr_piece[0]
            for piece in self.curr_piece:
                dx = piece[0] - center_x
                dy = piece[1] - center_y
                if self._is_valid_piece_location(center_x+dy*x, center_y+dx*y):
                    final_piece += [(center_x+dy*x, center_y+dx*y)]
                else:
                    self._gravity()
                    return    
        else:
            for piece in self.curr_piece:
                if self._is_valid_piece_location(piece[0] + x, piece[1] + y):
                    final_piece += [(piece[0] + x, piece[1] + y)]
                else:
                    self._gravity()
                    return
        
        # clear the previous blocks
        self._fill_board(0)

        # update current piece
        self.curr_piece = final_piece
        
        # update board
        self._fill_board(-1)

        # TODO-someday: consider making gravity happen less often?
        self._gravity()

    def _gravity(self):
        x, y = (0, -1)
        final_piece = []
        for piece in self.curr_piece:
            # print(piece)
            if self._is_valid_piece_location(piece[0] + x, piece[1] + y):
                final_piece += [(piece[0] + x, piece[1] + y)]
            else:
                self._lock_and_reset()
                return
        
        # clear the previous blocks
        self._fill_board(0)

        # update current piece
        self.curr_piece = final_piece
        
        # update board
        self._fill_board(-1)

    def _is_valid_piece_location(self, row, col):
        if row < 0 or row >= self.width:
            return False 
        if col < 0 or col >= self.height:
            return False
        return self.game_board[row][col] <= 0

    def _clear_line(self):
        pass
    
    def _lock_and_reset(self):
        self._fill_board(1)
        self._initialize_piece()
        self._fill_board(-1)

