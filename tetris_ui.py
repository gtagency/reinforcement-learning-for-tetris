import pygame
import sys
from tetris_engine import *
# Sharay: makes a game but doesn't do anything with it yet
game = GameState()

pygame.init()

# Sharay: board to be displayed
"""board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]"""

# Sharay: some basic information, makes adapting the program easier
pixelSize = 30
gridx = game.width
gridy = game.height

screen = pygame.display.set_mode((pixelSize * gridx, pixelSize * gridy))

# Sharay: Actions currently supported
action_lookup = {
    pygame.K_a: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_w: Action.ROTATE_CW,
    pygame.K_s: Action.ROTATE_CCW
    }


# clock = pygame.time.Clock()


while True:
    # Sharay: this just waits a bit before running, temporary
    pygame.time.wait(100)

    # Neil: changed this a bit so that game update happens once per iteration of the outer while
    # loop, instead of inside the pygame event loop.
    action = Action.IDLE
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            # Sharay: Updates the game with the key action
            if event.key in action_lookup:
                action = action_lookup[event.key]
    game.update(action)

    # cells[x][y] = True
    board = game.game_board

    # Ori - colors: yellow, magenta, cyan, red, blue, green, pink
    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 53, 184)]
    locked_colors = [(250, 249, 223), (223, 185, 208), (163, 220, 228), (255, 203, 165), (167, 183, 222),
                     (206, 240, 161), (244, 204, 203)]

    # Sharay: alters colors in the board
    screen.fill((0, 0, 0))
    for i in range(len(board)):
        for j in range(len(board[0])):
    # Ori - change color when locked
            if board[i][j] > 0:
                pygame.draw.rect(screen, locked_colors[board[i][j] - 1],
                                 pygame.Rect(pixelSize * (i), pixelSize * (len(board[0]) - j - 1), pixelSize, pixelSize))
            elif board[i][j] < 0:
                pygame.draw.rect(screen, colors[(board[i][j] * -1) - 1],
                                 pygame.Rect(pixelSize * (i), pixelSize * (len(board[0]) - j - 1), pixelSize, pixelSize))


    pygame.display.flip()
