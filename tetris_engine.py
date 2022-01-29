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

class GameState:
    # TODO:
    # - initialize game board here (10x20 booleans for now)
    # - initialize a current piece, with its x, y coordinates
    def __init__(self, width=10, height=20, initVal=0):
        # Joe: assume left bottom to be (0, 0), x coordinate goes to the right, y coordinate goes to the top

        self.width = width
        self.height = height
        self.gameBoard = [[initVal for col in range(height)] for row in range(width)]
        self.currPiece = (int(width/2), int(height-1))
        self.gameBoard[int(width/2)][height-1] = 1

    # TODO:
    # - return view of current game board
    def get_current_board(self):
        # Joe: I use 'x' to indicate an occupied position and '.' for an empty position
        out = [['x' if self.gameBoard[row][col] else '.' for row in range(self.width)] for col in range(self.height)]
        # print(out)
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    # TODO:
    # - first: process left/right movement based on action
    #   - if the movement results in invalid location/state, don't allow it
    # - then: process "gravity", attempting to move piece down
    #   - if the movement is invalid, then we hit something: lock the piece in
    #     place onto the board, and reset the piece to a new one
    # Neil: what I've suggested above is, in my opinion, the cleanest way to
    #  implement the core game logic, with the least code repetition. It will
    #  require the helper below, which I've also described.
    def update(self, action):
        # action should be a tuple(dx, dy)
        # newX, newY = self.currPiece + action
        # newPiece = (newX, newY)
        x, y = action
        # print(x, y)
        if self._is_valid_piece_location(self.currPiece, x, y):
            # print(self.currPiece)
            # clear the previous block and update the board
            self.gameBoard[self.currPiece[0]][self.currPiece[1]] = 0
            # update the current piece position
            # print(action)
            self.currPiece = (self.currPiece[0] + x, self.currPiece[1] + y)
            # print(self.currPiece)
            self.gameBoard[self.currPiece[0]][self.currPiece[1]] = 1
            print(self.get_current_board())
        else:
            print("Invalid movement")


    # TODO:
    # - implement this helper that just checks if this is a valid piece location
    # - will need to check for
    #   - out of bounds
    #   - overlap with existing squares (have not being implemented)
    def _is_valid_piece_location(self, piece, x, y):
        row, col = piece
        # print(row, col, x, y)
        if row + x < 0 or row + x >= self.width:
            return False
        if col + y < 0 or col + y >= self.height:
            return False
        return True

game = GameState()
print(game.get_current_board())
game.update((-1, 0))
print()
game.update((-1, 0))
print()
game.update((-1, 0))
print()
game.update((-1, 0))
print()
game.update((-1, 0))
print()
game.update((-1, 0))
print()