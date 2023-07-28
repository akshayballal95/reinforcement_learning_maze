import numpy as np
import random
from typing import Tuple

class Maze:
    def __init__(
        self, level, goal_pos: Tuple[int, int], MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25
    ):
        """
        Maze class to represent a simple maze environment.

        Args:
            level (List[str]): A list of strings representing the maze layout.
            goal_pos (Tuple[int, int]): The goal position (row, col) in the maze.
            MAZE_HEIGHT (int, optional): Height of the maze in pixels. Defaults to 600.
            MAZE_WIDTH (int, optional): Width of the maze in pixels. Defaults to 600.
            SIZE (int, optional): Number of tiles per row/column in the maze. Defaults to 25.
        """
        self.goal = (23, 20)
        self.number_of_tiles = SIZE
        self.tile_size = MAZE_HEIGHT // self.number_of_tiles
        self.walls = self.create_walls(level)
        self.goal_pos = goal_pos
        self.state = self.get_init_state(level)
        self.maze = self.create_maze(level)

        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )

    def reset(self):
        """Reset the maze environment."""
        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )
        self.goal_pos = random.sample(self.maze, 1)[0]

    def create_walls(self, level):
        """
        Create a list of wall positions in the maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            List[Tuple[int, int]]: A list of wall positions (row, col).
        """
        walls = []
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "X":
                    walls.append((row, col))
        return walls

    def create_maze(self, level):
        """
        Create a list of positions that are not walls in the maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            List[Tuple[int, int]]: A list of maze positions (row, col) that are not walls.
        """
        maze = []
        for row in range(len(level)):
            for col in range(len(level[row])):
                if (row, col) not in self.walls:
                    maze.append((row, col))
        return maze

    def get_init_state(self, level):
        """
        Get the initial state (player's position) in the maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            Tuple[int, int]: The initial state (row, col) in the maze.
        """
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "P":
                    return (row, col)

    def _get_next_state(self, state: Tuple[int, int], action: int):
        """
        Get the next state based on the current state and action.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[int, int]: The next state (row, col) after taking the action.
        """
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
        """
        Compute the reward for taking an action from the current state.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            float: The reward for taking the action from the current state.
        """
        next_state = self._get_next_state(state, action)
        return -float(state != self.goal_pos)

    def simulate_step(self, state, action):
        """
        Simulate a step in the maze environment.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[Tuple[int, int], float, bool]: Tuple containing the next state, reward, and done flag.
        """
        next_state = self._get_next_state(state, action)
        reward = self.compute_reward(state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def step(self, action):
        """
        Take a step in the maze environment.

        Args:
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[Tuple[int, int], float, bool]: Tuple containing the next state, reward, and done flag.
        """
        next_state = self._get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def policy(self, state):
        """
        Get the policy probabilities for different actions at a given state.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.

        Returns:
            np.ndarray: Array of shape (4,) containing the action probabilities.
        """
        return self.policy_probs[state]

    def solve(self, gamma=0.99, theta=1e-6):
        """
        Solve the maze environment using the value iteration algorithm.

        Args:
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            theta (float, optional): Threshold for convergence. Defaults to 1e-6.
        """
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
