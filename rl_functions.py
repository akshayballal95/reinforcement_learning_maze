from typing import Tuple, Dict, Optional, Iterable
import numpy as np


class Maze:
    def __init__(
        self, level, goal_pos: Tuple[int, int], MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25
    ):
        self.goal = (23, 20)
        self.number_of_tiles = SIZE
        self.tile_size = MAZE_HEIGHT // self.number_of_tiles
        self.walls = self.create_walls(level)
        self.goal_pos = goal_pos
        self.state = self.get_init_state(level)

        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )

    def reset(self):
        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )


    def create_walls(self, level):
        walls = []
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "X":
                    walls.append((row, col))

        return walls

    def get_init_state(self, level):
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "P":
                    return (row, col)

    def _get_next_state(self, state: Tuple[int, int], action: int):
        if action == 0:
            next_state = (state[0], state[1] - 1)
        elif action == 1:
            next_state = (state[0] - 1, state[1])
        elif action == 2:
            next_state = (state[0], state[1] + 1)
        elif action == 3:
            next_state = (state[0] + 1, state[1])
        else:
            raise ValueError("Action value not supported:", action)
        if (next_state[0], next_state[1]) not in self.walls:
            return next_state
        return state

    def compute_reward(self, state: Tuple[int, int], action: int):
        next_state = self._get_next_state(state, action)
        return -float(state != self.goal_pos)

    def simulate_step(self, state, action):
        next_state = self._get_next_state(state, action)
        reward = self.compute_reward(state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def step(self, action):
        next_state = self._get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def policy(self, state):
        return self.policy_probs[state]

    def solve(self, gamma=0.99, theta=1e-6):
        delta = float("inf")

        while delta > theta:
            delta = 0
            for row in range(self.number_of_tiles):
                for col in range(self.number_of_tiles):
                    if (row, col) not in self.walls:
                        old_value = self.state_values[row, col]
                        q_max = float("-inf")

                        for action in range(4):
                            next_state, reward, done = self.simulate_step(
                                (row, col), action
                            )
                            value = reward + gamma * self.state_values[next_state]
                            if value > q_max:
                                q_max = value
                                action_probs = np.zeros(shape=(4))
                                action_probs[action] = 1

                        self.state_values[row, col] = q_max
                        self.policy_probs[row, col] = action_probs

                        delta = max(delta, abs(old_value - self.state_values[row, col]))
