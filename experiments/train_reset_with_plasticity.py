import torch
import yaml
from agents.dqn_agent_plasticity import DQNAgent
from environments.gridworld import GridWorldEnv
from environments.task_manager import TaskManager
from utils.seed import set_seed
from utils.metrics import log_metrics

# Load configuration
with open("configs/dqn_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set random seed for reproducibility
set_seed(config["seed"])

# Initialize environment and task manager
env = GridWorldEnv(grid_size=config["grid_size"], max_steps=config["max_steps"])
task_manager = TaskManager(env, num_tasks=config["num_tasks"])

# Initialize DQN agent with plasticity
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config
)

# Training loop
all_rewards = []
for task in range(config["num_tasks"]):
    print(f"Starting Task {task + 1}/{config['num_tasks']}...")

    # Reset task (reinitialize environment and task manager)
    state = task_manager.reset_task()
    total_reward = 0

    # Plasticity injection: Reset and reinitialize some parameters every 5 tasks
    if task % 5 == 0:
        print("Injecting plasticity...")
        # Reset the parameters for plasticity injection
        agent.model.head_prime_1.apply(lambda x: x.reset_parameters())  # Reinitialize head_prime_1
        agent.model.head_prime_2.apply(lambda x: x.reset_parameters())  # Reinitialize head_prime_2

    # Training for each episode in the task
    for episode in range(config["episodes_per_task"]):
        state = env.reset()
        episode_reward = 0

        for step in range(config["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}/{config['episodes_per_task']} - Reward: {episode_reward}")
        total_reward += episode_reward

    all_rewards.append(total_reward)
    print(f"Task {task + 1} Total Reward: {total_reward}")

    # Save model weights after each task
    torch.save(agent.q_network.state_dict(), f"logs/dqn_task_{task + 1}_with_plasticity.pt")

# Log and plot metrics
log_metrics(all_rewards, "logs/rewards_with_plasticity.json")
