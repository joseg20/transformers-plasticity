import argparse
import yaml
from experiments.train_reset import train_reset_every_task
from experiments.train_never_reset import train_never_reset
from experiments.evaluate import evaluate_agent
from environments.gridworld import GridWorldEnv
from environments.task_manager import TaskManager
from agents.dqn_agent import DQNAgent
from utils.seed import set_seed
from utils.metrics import log_metrics
import torch

def load_config(config_path):
    """Carga la configuración desde un archivo YAML."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run DQN experiments with Transformer backbone.")
    parser.add_argument('--train', choices=['reset', 'never_reset'], required=True, help="Select training strategy")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate trained agent")
    parser.add_argument('--config', default='configs/dqn_config.yaml', help="Path to the config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    set_seed(config['seed'])

    # Initialize environment and task manager
    env = GridWorldEnv(grid_size=config["grid_size"], max_steps=config["max_steps"])
    task_manager = TaskManager(env, num_tasks=config["num_tasks"])

    # Initialize agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )

    # Training
    if args.train == 'reset':
        print("Starting training with 'Reset Every Task' strategy...")
        train_reset_every_task(agent, env, task_manager, config)
    
    elif args.train == 'never_reset':
        print("Starting training with 'Never Reset' strategy...")
        train_never_reset(agent, env, task_manager, config)

    # Evaluation
    if args.evaluate:
        print("Starting evaluation of the trained agent...")
        agent.q_network.load_state_dict(torch.load("logs/dqn_task_10_reset.pt"))  # Adjust path as needed
        evaluate_agent(agent, env, task_manager, "logs/evaluation_metrics.json")

if __name__ == "__main__":
    main()
