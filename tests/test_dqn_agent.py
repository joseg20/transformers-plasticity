import pytest
import torch
import numpy as np
from agents.dqn_agent import DQNAgent
from environments.gridworld import GridWorldEnv
from utils.seed import set_seed

# Setup the seed
set_seed(42)

# Test DQN agent initialization
def test_dqn_agent_initialization():
    env = GridWorldEnv(grid_size=5)
    config = {
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
        "gamma": 0.99,
        "batch_size": 64,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_length": 25,
        "learning_rate": 1e-3,
        "replay_buffer_capacity": 10000,
        "seed": 42
    }
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )
    
    # Check that agent has a Q-network and target network
    assert isinstance(agent.q_network, torch.nn.Linear)
    assert isinstance(agent.target_network, torch.nn.Linear)

# Test the agent action selection (ε-greedy policy)
def test_select_action():
    env = GridWorldEnv(grid_size=5)
    config = {
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
        "gamma": 0.99,
        "batch_size": 64,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_length": 25,
        "learning_rate": 1e-3,
        "replay_buffer_capacity": 10000,
        "seed": 42
    }
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )
    
    state = np.zeros(env.observation_space.shape[0])
    
    # Test action selection with epsilon-greedy
    action = agent.select_action(state)
    assert action in [0, 1, 2, 3]  # Actions should be one of the 4 cardinal directions

# Test the agent's update function
def test_update_agent():
    env = GridWorldEnv(grid_size=5)
    config = {
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
        "gamma": 0.99,
        "batch_size": 64,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_length": 25,
        "learning_rate": 1e-3,
        "replay_buffer_capacity": 10000,
        "seed": 42
    }
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )
    
    state = np.zeros(env.observation_space.shape[0])
    action = 0  # Take action "up"
    reward = 1.0
    next_state = np.zeros(env.observation_space.shape[0])
    done = False
    
    # Add a transition to the replay buffer
    agent.replay_buffer.add(state, action, reward, next_state, done)
    
    # Update the agent
    agent.update()
    
    # Check if the agent's Q-network parameters are updated
    for param in agent.q_network.parameters():
        assert param.requires_grad  # Ensure the parameters are trainable

