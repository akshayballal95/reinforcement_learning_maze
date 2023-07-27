import pygame
from rl_functions import Maze
import numpy as np

GAME_HEIGHT = 600
GAME_WIDTH = 600

NUMBER_OF_TILES = 25

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

print(env.policy_probs)

SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700

TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES

pygame.init()


screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
wall_surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
clock = pygame.time.Clock()
running = True

treasure_pos = env.goal_pos


player_pos = env.state

time = 0

while running:
    time = time + clock.get_time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(
        surface, ((SCREEN_HEIGHT - GAME_HEIGHT) / 2, (SCREEN_WIDTH - GAME_WIDTH) / 2)
    )

    surface.fill((27, 64, 121))

    for row in range(len(level)):
        for col in range(len(level[row])):
            if level[row][col] == "X":
                pygame.draw.rect(
                    surface,
                    (241, 162, 8),
                    (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )

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

    pygame.draw.rect(
        surface,
        "green",
        pygame.Rect(
            treasure_pos[1] * TILE_SIZE,
            treasure_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=TILE_SIZE,
    )

    action = np.argmax(env.policy_probs[player_pos])

    if action == 1:
        if player_pos[0] > 0 and (player_pos[0] - 1, player_pos[1]) not in env.walls:
            player_pos = (player_pos[0] - 1, player_pos[1])
            env.state = player_pos
    elif action == 3:
        if player_pos[0] < NUMBER_OF_TILES - 1 and (player_pos[0] + 1, player_pos[1]) not in env.walls:
            player_pos = (player_pos[0] + 1, player_pos[1])
            env.state = player_pos
    elif action == 0:
        if player_pos[1] > 0 and (player_pos[0], player_pos[1] - 1) not in env.walls:
            player_pos = (player_pos[0], player_pos[1] - 1)
            env.state = player_pos
    
    elif action == 2:
        if player_pos[1] < NUMBER_OF_TILES - 1 and (player_pos[0], player_pos[1] + 1) not in env.walls:
            player_pos = (player_pos[0], player_pos[1] + 1)
            env.state = player_pos


    # if time > 50:
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_w]:
    #         if (
    #             player_pos[0] > 0
    #             and (player_pos[0] - 1, player_pos[1]) not in env.walls
    #         ):
    #             player_pos = (player_pos[0] - 1, player_pos[1])
    #     if keys[pygame.K_s]:
    #         if (
    #             player_pos[0] < NUMBER_OF_TILES - 1
    #             and (player_pos[0] + 1, player_pos[1]) not in env.walls
    #         ):
    #             player_pos = (player_pos[0] + 1, player_pos[1])
    #     if keys[pygame.K_a]:
    #         if (
    #             player_pos[1] > 0
    #             and (player_pos[0], player_pos[1] - 1) not in env.walls
    #         ):
    #             player_pos = (player_pos[0], player_pos[1] - 1)
    #     if keys[pygame.K_d]:
    #         if (
    #             player_pos[1] < NUMBER_OF_TILES - 1
    #             and (player_pos[0], player_pos[1] + 1) not in env.walls
    #         ):
    #             player_pos = (player_pos[0], player_pos[1] + 1)

    #     time = 0
    #     env.state = player_pos

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
