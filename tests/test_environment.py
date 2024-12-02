import pytest
from environments.gridworld import GridWorldEnv
import numpy as np

# Test environment reset
def test_reset():
    env = GridWorldEnv(grid_size=5)
    state = env.reset()
    
    assert state.shape == (env.grid_size * env.grid_size,)  # State should be flattened
    assert np.count_nonzero(state == 1) == 1  # There should be exactly one agent (1) in the grid
    assert np.count_nonzero(state == 2) == 1  # There should be exactly one goal (2) in the grid

# Test environment step
def test_step():
    env = GridWorldEnv(grid_size=5)
    state = env.reset()
    
    # Take action: move right
    next_state, reward, done, _ = env.step(3)  # Action 3 corresponds to moving right
    assert next_state.shape == (env.grid_size * env.grid_size,)  # Next state should also be flattened
    assert reward == -0.1  # Reward should be -0.1 for a step
    assert not done  # The episode should not be done yet
    
    # Simulate the agent reaching the goal
    env.agent_pos = [4, 4]  # Move agent to the goal
    next_state, reward, done, _ = env.step(3)  # Action 3 again
    assert reward == 1.0  # Reward should be 1.0 when the agent reaches the goal
    assert done  # The episode should end

