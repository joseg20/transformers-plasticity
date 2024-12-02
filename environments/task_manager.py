import numpy as np

class TaskManager:
    """
    Task manager for sequential tasks in the GridWorld environment.
    """
    def __init__(self, gridworld_env, num_tasks=10):
        """
        Initialize the task manager.
        
        Args:
            gridworld_env (GridWorldEnv): Instance of the GridWorld environment.
            num_tasks (int): Number of sequential tasks.
        """
        self.env = gridworld_env
        self.num_tasks = num_tasks
        self.current_task = 0

    def reset_task(self):
        """
        Reset the current task in the GridWorld environment.
        
        Returns:
            np.array: Initial state of the new task.
        """
        self.current_task += 1
        if self.current_task > self.num_tasks:
            self.current_task = 1

        # Example: Change the goal position dynamically
        self
