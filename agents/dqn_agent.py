import torch
import torch.nn as nn
import torch.optim as optim
from agents.transformer_backbone import TransformerBackbone
from agents.replay_buffer import ReplayBuffer
import random

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        """
        DQN Agent with Transformer backbone.
        
        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Number of possible actions.
            config (dict): Configuration parameters.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = config["epsilon_start"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]

        # Transformer backbone
        self.model = TransformerBackbone(
            state_dim=state_dim,
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_seq_length=config["max_seq_length"]
        )
        self.q_network = nn.Linear(config["embed_dim"], action_dim)
        self.target_network = nn.Linear(config["embed_dim"], action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(self.model(state_tensor))
            return torch.argmax(q_values).item()

    def update(self):
        """
        Update the Q-network using a batch from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute targets
        with torch.no_grad():
            next_q_values = self.target_network(self.model(next_states)).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute current Q-values
        q_values = self.q_network(self.model(states)).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """
        Update the target network to match the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
