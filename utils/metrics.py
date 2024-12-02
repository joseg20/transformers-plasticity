import json
import numpy as np

def calculate_mse(predictions, targets):
    """
    Calculate Mean Squared Error (MSE) between predictions and targets.
    
    Args:
        predictions (np.array): Predicted Q-values.
        targets (np.array): Target Q-values.
    
    Returns:
        float: Mean Squared Error.
    """
    return np.mean((predictions - targets) ** 2)

def log_metrics(rewards, filename):
    """
    Save metrics (e.g., rewards) to a JSON file.
    
    Args:
        rewards (list): List of rewards per task or episode.
        filename (str): Path to save the metrics.
    """
    with open(filename, "w") as f:
        json.dump({"rewards": rewards}, f)

def evaluate_agent(agent, env, task_manager, save_path):
    """
    Evaluate an agent on a sequence of tasks and log metrics.
    
    Args:
        agent (DQNAgent): Trained agent.
        env (GridWorldEnv): Environment instance.
        task_manager (TaskManager): Task manager instance.
        save_path (str): Path to save evaluation metrics.
    """
    rewards = []
    for task in range(task_manager.num_tasks):
        state = task_manager.reset_task()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f"Task {task + 1}: Total Reward = {total_reward}")

    log_metrics(rewards, save_path)
