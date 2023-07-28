import pygame
import numpy as np
from maze import Maze
import random

# Constants
GAME_HEIGHT = 600
GAME_WIDTH = 600
NUMBER_OF_TILES = 25
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700
TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES

# Maze layout
level = [
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
    "X XXXXXXXX          XXXXX",
    "X XXXXXXXX  XXXXXX  XXXXX",
    "X      XXX  XXXXXX  XXXXX",
    "X      XXX  XXX        PX",
    "XXXXXX  XX  XXX        XX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "X  XXX      XXXXXXXXXXXXX",
    "X  XXX  XXXXXXXXXXXXXXXXX",
    "X         XXXXXXXXXXXXXXX",
    "X             XXXXXXXXXXX",
    "XXXXXXXXXXX      XXXXX  X",
    "XXXXXXXXXXXXXXX  XXXXX  X",
    "XXX  XXXXXXXXXX         X",
    "XXX                     X",
    "XXX         XXXXXXXXXXXXX",
    "XXXXXXXXXX  XXXXXXXXXXXXX",
    "XXXXXXXXXX              X",
    "XX   XXXXX              X",
    "XX   XXXXXXXXXXXXX  XXXXX",
    "XX    XXXXXXXXXXXX  XXXXX",
    "XX        XXXX          X",
    "XXXX                    X",
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
]

env = Maze(
    level,
    goal_pos=(23, 20),
    MAZE_HEIGHT=GAME_HEIGHT,
    MAZE_WIDTH=GAME_WIDTH,
    SIZE=NUMBER_OF_TILES,
)
env.reset()
env.solve()

SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700

TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES


# Initialize Pygame
pygame.init()

# Create the game window
screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
pygame.display.set_caption("Maze Solver")  # Set a window title

surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
clock = pygame.time.Clock()
running = True

# Get the initial player and goal positions
treasure_pos = env.goal_pos
player_pos = env.state

time = 0

def reset_goal():
    env.reset()
    env.solve()

# Game loop
while running:
    time = time + clock.get_time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the surface
    surface.fill((27, 64, 121))

    # Draw the walls in the maze
    for row in range(len(level)):
        for col in range(len(level[row])):
            if level[row][col] == "X":
                pygame.draw.rect(
                    surface,
                    (241, 162, 8),
                    (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )

    # Draw the player's position
    pygame.draw.rect(
        surface,
        (255, 51, 102),
        pygame.Rect(
            player_pos[1] * TILE_SIZE,
            player_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=3,
    )

    # Draw the goal position
    pygame.draw.rect(
        surface,
        "green",
        pygame.Rect(
            env.goal_pos[1] * TILE_SIZE,
            env.goal_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=TILE_SIZE,
    )

    # Update the screen
    screen.blit(
        surface, ((SCREEN_HEIGHT - GAME_HEIGHT) / 2, (SCREEN_WIDTH - GAME_WIDTH) / 2)
    )
    pygame.display.flip()

    # Get the action based on the current policy
    action = np.argmax(env.policy_probs[player_pos])

    # Move the player based on the action
    if action == 1 and player_pos[0] > 0 and (player_pos[0] - 1, player_pos[1]) not in env.walls:
        player_pos = (player_pos[0] - 1, player_pos[1])
        env.state = player_pos
    elif action == 3 and player_pos[0] < NUMBER_OF_TILES - 1 and (player_pos[0] + 1, player_pos[1]) not in env.walls:
        player_pos = (player_pos[0] + 1, player_pos[1])
        env.state = player_pos
    elif action == 0 and player_pos[1] > 0 and (player_pos[0], player_pos[1] - 1) not in env.walls:
        player_pos = (player_pos[0], player_pos[1] - 1)
        env.state = player_pos
    elif action == 2 and player_pos[1] < NUMBER_OF_TILES - 1 and (player_pos[0], player_pos[1] + 1) not in env.walls:
        player_pos = (player_pos[0], player_pos[1] + 1)
        env.state = player_pos

    # Check if the player reached the goal, then reset the goal
    if env.state == env.goal_pos:
        reset_goal()

    # Control the frame rate of the game
    clock.tick(30)

# Quit Pygame when the game loop is exited
pygame.quit()