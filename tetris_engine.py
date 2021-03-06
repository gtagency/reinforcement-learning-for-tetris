"""
Authors: Zhao Chen (Joe), Ori Yoked, Akash Vemulapalli, Neil Thistlethwaite
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
    RESET = 5


GAME_PIECE_OFFSETS = [
    # Shape 0: L piece
    #  -------
    #  ----x--
    #  --xxx--
    #  -------
    [(0, -1), (1, -1), (-1, -1), (1, 0)],

    # Shape 1: I piece
    #  --------
    #  --xxxx--
    #  --------
    [(0, 0), (-1, 0), (1, 0), (2, 0)],

    # Shape 2: O piece
    #  ------
    #  --xx--
    #  --xx--
    #  ------
    [(0, -1), (0, 0), (1, 0), (1, -1)],

    # Shape 3: J piece
    #  -------
    #  --x----
    #  --xxx--
    #  -------
    [(0, -1), (-1, 0), (-1, -1), (1, -1)],

    # Shape 4: Z piece
    #  -------
    #  --xx---
    #  ---xx--
    #  -------
    [(0, -1), (0, 0), (-1, 0), (1, -1)],

    # Shape 5: T piece
    #  -------
    #  ---x---
    #  --xxx--
    #  -------
    [(0, -1), (-1, -1), (0, 0), (1, -1)],

    # Shape 6: S piece
    #  -------
    #  ---xx--
    #  --xx---
    #  -------
    [(0, -1), (0, 0), (1, 0), (-1, -1)],
]


class GamePiece:
    # seven different shapes in tetris
    def __init__(self, board_width=10, board_height=20):
        self.shape_num = random.randrange(0, 7)
        self.width = board_width
        self.height = board_height
        self.shape = []
        self._initialize()
        
    def _initialize(self):
        shape_offsets = GAME_PIECE_OFFSETS[self.shape_num]
        w_offset = random.randint(-1, 0)
        for dx, dy in shape_offsets:
            self.shape.append((self.width//2 + w_offset + dx, self.height - 1 + dy))


class GameState:
    def __init__(self, width=10, height=20):
        # Joe: assume bottom left to be (0, 0), x coordinate goes to the right,
        #      y coordinate goes up
        self.width = width
        self.height = height
        self.current_piece = None
        self.state_num = 0
        # {self.game_board} entries correspond to following cell states:
        #    0  empty cell
        #   +k  locked with piece of type k
        #   -k  holding current piece of type k
        self.game_board = [[0 for col in range(height)] for row in range(width)]
        self.game_piece = None
        self._initialize_piece()
        self._fill_piece_in_board(-1)
        self.stop = False

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

    def _reset(self):
        self.current_piece = None
        self.game_board = [[0 for col in range(self.height)] for row in range(self.width)]
        self.game_piece = None
        self._initialize_piece()
        self._fill_piece_in_board(-1)

    # Neil: now returns number of lines that were cleared
    def update(self, action):
        # convert action to a tuple(dx, dy)
        new_piece = []
        is_new_piece_valid = True

        if action == Action.RESET:
            self.stop = False
            self._reset()
            return 0

        if self.stop == True:
            return 0

        if action == Action.ROTATE_CCW or action == Action.ROTATE_CW:
            if self.game_piece.shape_num == 2:
                return self._gravity()
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
        return self._gravity()
    
    # Neil: now returns number of lines that were cleared
    def _gravity(self):
        # gravity occurs once per 2s
        #if self.state_num < 2:
        #    self.state_num += 1
        #    return
        #self.state_num = 0

        x, y = (0, -1)
        new_piece = []
        for piece in self.current_piece:
            if self._is_valid_piece_location(piece[0] + x, piece[1] + y):
                new_piece.append((piece[0] + x, piece[1] + y))
            else:
                if piece[1] >= self.height - 2:
                    self.stop = True
                return self._lock_and_reset()
        
        # Clear previous piece, update current piece, and fill it in board
        self._fill_piece_in_board(0)
        self.current_piece = new_piece
        self._fill_piece_in_board(-1)

        # Neil: if we get down to here, we didn't lock in any pieces (and can't
        # have cleared any lines)
        return 0

    def _is_valid_piece_location(self, row, col):
        if row < 0 or row >= self.width:
            return False 
        if col < 0 or col >= self.height:
            return False
        return self.game_board[row][col] <= 0

    def _clear_lines(self):
        i = 0
        num_cleared = 0
        while i < self.height - 1:
            full_row = True
            for j in range(self.width):
                if self.game_board[j][i] <= 0:
                    full_row = False
    
            if full_row:
                num_cleared += 1
                self._fall(i)
                i -= 1

            i += 1
        return num_cleared

    def _fall(self, row):
        for i in range(self.height - 1):
            for j in range(self.width):
                if i >= row:
                    self.game_board[j][i] = self.game_board[j][i+1]

        for j in range(self.width):
            self.game_board[j][self.height - 1] = 0
    
    def _lock_and_reset(self):
        self._fill_piece_in_board(1)
        num_cleared = self._clear_lines()
        self._initialize_piece()
        self._fill_piece_in_board(-1)
        return num_cleared




