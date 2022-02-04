# Authors: Neil Thistlethwaite, Zhao Chen(Joe)
# (add your name above if you contribute to this file)
# The Agency, Reinforcement Learning for Tetris

# Primary goal of this file is to allow us to simulate the game of Tetris. Each
# "GameState" object will keep track of the current state of the game, and we'll
# call its {update} method every game tick with inputs. This will also expose
# methods for the UI and agent to see the game.

from enum import Enum

class Action(Enum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    # TODO-someday: change this to allow simultaneous movement and rotation?
    ROTATE_CW = 3
    ROTATE_CCW = 4

# General advice from Neil for implementing this:
# Start simple. It's tempting to want to immediately implement full
# functionality, but it's going to be overwhelming (and result in very messy
# code). Instead, start simple and add features incrementally. Start off with a
# single 1x1 piece, and no colors. 

class GamePiece:
    # Joe:  I think we need different shapes of pieces and probably a class will help
    def __init__(self):
        pass

class GameState:
    # TODO:
    # - initialize game board here (10x20 booleans for now)
    # - initialize a current piece, with its x, y coordinates
    def __init__(self, width=10, height=20, init_val=0):
        # Joe: assume left bottom to be (0, 0), x coordinate goes to the right, y coordinate goes to the top

        self.width = width
        self.height = height
        self.curr_piece = None
        self.action_map = {Action.IDLE:(0, 0), Action.LEFT:(-1, 0), Action.RIGHT:(1, 0)}
        self.game_board = [[init_val for col in range(height)] for row in range(width)]
        # game_board: value = 1: locked, value = 0: empty, value = -1, current piece

        self._initialize_piece()
        self._fill_board(-1)

    def _initialize_piece(self):
        width = self.width
        height = self.height
        self.curr_piece = [
            (int(width/2), int(height-1)), 
            (int(width/2+1), int(height-1)),
            (int(width/2+1), int(height-2))]
                        
    def _fill_board(self, val):
        for piece in self.curr_piece:
            self.game_board[piece[0]][piece[1]] = val
    # TODO:
    # - return view of current game board
    def get_current_board(self):
        # Joe: I use 'x' to indicate an occupied position and '.' for an empty position
        out = [['x' if self.game_board[row][col] else '.' for row in range(self.width)] for col in range(self.height)]
        # print(out)
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    # TODO:
    # - first: process left/right movement based on action
    #   - if the movement results in invalid location/state, don't allow it
    # - then: process "gravity", attempting to move piece down

    #   - if the movement is invalid, then we hit something: lock the piece in
    #     place onto the board, and reset the piece to a new one
    #   Joe: how do we know which piece to reset to? My guess is that we take the lowest piece

    # Neil: what I've suggested above is, in my opinion, the cleanest way to
    #  implement the core game logic, with the least code repetition. It will
    #  require the helper below, which I've also described.
    def update(self, action):
        # action should be a tuple(dx, dy)
        x, y = self.action_map[action]
        # print(x, y)
        final_piece = []
        for piece in self.curr_piece:
            # print(piece)
            if self._is_valid_piece_location(piece, x, y):
                final_piece += [(piece[0] + x, piece[1] + y)]
            else:
                print("Invalid movement")
                self._gravity()
                return
        
        # clear the previous blocks
        self._fill_board(0)

        # update current piece
        self.curr_piece = final_piece
        
        # update board
        self._fill_board(-1)

        # in one unit of time, the piece will be pulled down by gravity by one unit
        self._gravity()
        

    def _gravity(self):
        x, y = (0, -1)
        final_piece = []
        for piece in self.curr_piece:
            # print(piece)
            if self._is_valid_piece_location(piece, x, y):
                final_piece += [(piece[0] + x, piece[1] + y)]
            else:
                print("Invalid movement")
                self._lock_and_reset()
                return
        
        # clear the previous blocks
        self._fill_board(0)

        # update current piece
        self.curr_piece = final_piece
        
        # update board
        self._fill_board(-1)

    # TODO:
    # - implement this helper that just checks if this is a valid piece location
    # - will need to check for
    #   - out of bounds
    #   - overlap with existing squares (have not being implemented)
    
    def _is_valid_piece_location(self, piece, x, y):
        row, col = piece
        row += x
        col += y
        if row < 0 or row >= self.width:
            return False 
        if col < 0 or col >= self.height:
            return False
        return self.game_board[row][col] != 1 

    def _clear_line(self):
        pass
    def _lock_and_reset(self):
        self._fill_board(1)
        self._initialize_piece()
        self._fill_board(-1)
        
