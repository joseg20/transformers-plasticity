import matplotlib.pyplot as plt
import json

def plot_rewards(log_file, save_path=None):
    """
    Plot rewards from a JSON log file.
    
    Args:
        log_file (str): Path to the JSON file containing rewards.
        save_path (str): Path to save the plot (optional).
    """
    with open(log_file, "r") as f:
        data = json.load(f)
    
    rewards = data["rewards"]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o", label="Rewards per Task")
    plt.xlabel("Task Number", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title("Rewards Across Tasks", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_mse(mse_values, save_path=None):
    """
    Plot Mean Squared Error (MSE) values over training.
    
    Args:
        mse_values (list): List of MSE values.
        save_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_values) + 1), mse_values, label="MSE per Update")
    plt.xlabel("Update Step", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title("MSE During Training", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()
