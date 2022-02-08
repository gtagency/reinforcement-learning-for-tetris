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
    def __init__(self, width=10, height=20):
        # Joe: assume bottom left to be (0, 0), x coordinate goes to the right,
        #      y coordinate goes up
        self.width = width
        self.height = height
        self.current_piece = None
        # {self.game_board} entries correspond to following cell states:
        #    0  empty cell
        #   +k  locked with piece of type k
        #   -k  holding current piece of type k
        self.game_board = [[0 for col in range(height)] for row in range(width)]
        self.game_piece = None
        self._initialize_piece()
        self._fill_piece_in_board(-1)

    def _initialize_piece(self):
        self.game_piece = GamePiece(board_width=self.width,
                                    board_height=self.height)
        self.current_piece = self.game_piece.shape

    def _fill_piece_in_board(self, multiplier):
        # multiplier should be 0, +1, or -1, according to piece state
        for piece in self.current_piece:
            self.game_board[piece[0]][piece[1]] = multiplier * (self.game_piece.shape_num + 1)

    ## DEPRECATED
    def get_current_board(self):
        out = [['x' if self.game_board[row][col] else '.' for row in range(self.width)] for col in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def update(self, action):
        # convert action to a tuple(dx, dy)
        new_piece = []
        is_new_piece_valid = True

        if action == Action.ROTATE_CCW or action == Action.ROTATE_CW:
            if self.game_piece.shape_num == 2:
                self._gravity()
                return
            xy_mul = -1 if action == Action.ROTATE_CCW else 1
            center_x, center_y = self.current_piece[0]
            for piece in self.current_piece:
                rel_x = piece[0] - center_x
                rel_y = piece[1] - center_y
                new_x = center_x + rel_y * xy_mul
                new_y = center_y + rel_x * xy_mul * -1
                if self._is_valid_piece_location(new_x, new_y):
                    new_piece.append((new_x, new_y))
                else:
                    is_new_piece_valid = False
                    break
        elif action == Action.LEFT or action == Action.RIGHT:
            dx, dy = (1, 0) if action == Action.RIGHT else (-1, 0)
            for piece in self.current_piece:
                if self._is_valid_piece_location(piece[0] + dx, piece[1] + dy):
                    new_piece.append((piece[0] + dx, piece[1] + dy))
                else:
                    is_new_piece_valid = False
                    break

        if action != Action.IDLE and is_new_piece_valid:
            # Clear previous piece, update current piece, and fill it in board
            self._fill_piece_in_board(0)
            self.current_piece = new_piece
            self._fill_piece_in_board(-1)

        # TODO-someday: consider making gravity happen less often?
        self._gravity()

    def _gravity(self):
        x, y = (0, -1)
        new_piece = []
        for piece in self.current_piece:
            if self._is_valid_piece_location(piece[0] + x, piece[1] + y):
                new_piece.append((piece[0] + x, piece[1] + y))
            else:
                self._lock_and_reset()
                return
        
        # Clear previous piece, update current piece, and fill it in board
        self._fill_piece_in_board(0)
        self.current_piece = new_piece
        self._fill_piece_in_board(-1)

    def _is_valid_piece_location(self, row, col):
        if row < 0 or row >= self.width:
            return False 
        if col < 0 or col >= self.height:
            return False
        return self.game_board[row][col] <= 0

    def _clear_line(self):
        pass
    
    def _lock_and_reset(self):
        self._fill_piece_in_board(1)
        self._initialize_piece()
        self._fill_piece_in_board(-1)

