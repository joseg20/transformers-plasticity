import torch
import yaml
from agents.dqn_agent import DQNAgent
from environments.gridworld import GridWorldEnv
from environments.task_manager import TaskManager
from utils.metrics import evaluate_agent

# Load configuration
with open("configs/dqn_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize environment and task manager
env = GridWorldEnv(grid_size=config["grid_size"], max_steps=config["max_steps"])
task_manager = TaskManager(env, num_tasks=config["num_tasks"])

# Load agent for evaluation
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config
)
agent.q_network.load_state_dict(torch.load("logs/dqn_task_10_reset.pt"))

# Evaluate on new tasks
evaluate_agent(agent, env, task_manager, "logs/evaluation_metrics.json")
