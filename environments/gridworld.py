import numpy as np
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    """
    A simple GridWorld environment for Reinforcement Learning.
    """
    def __init__(self, grid_size=5, max_steps=50):
        """
        Initialize the GridWorld environment.
        
        Args:
            grid_size (int): Size of the grid (N x N).
            max_steps (int): Maximum steps per episode.
        """
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Action space: 4 cardinal directions
        self.action_space = spaces.Discrete(4)

        # Observation space: flattened grid with one-hot encoding
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size * self.grid_size,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.array: Flattened grid state.
        """
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.steps = 0

        # Generate grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.grid[tuple(self.agent_pos)] = 1  # Agent position
        self.grid[tuple(self.goal_pos)] = 2  # Goal position

        return self._get_obs()

    def _get_obs(self):
        """
        Get the current state as a flattened grid.
        """
        return self.grid.flatten()

    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Action taken by the agent (0: up, 1: down, 2: left, 3: right).
        
        Returns:
            tuple: (state, reward, done, info)
        """
        self.steps += 1
        x, y = self.agent_pos

        # Update agent position based on action
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Right
            y += 1

        self.agent_pos = [x, y]

        # Update grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.grid[tuple(self.agent_pos)] = 1
        self.grid[tuple(self.goal_pos)] = 2

        # Check if the agent reached the goal
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.1  # Reward for reaching goal, penalty for each step

        # End the episode if max steps reached
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        """
        Render the environment (print grid).
        """
        for row in self.grid:
            print(' '.join(['A' if cell == 1 else 'G' if cell == 2 else '.' for cell in row]))
        print()
