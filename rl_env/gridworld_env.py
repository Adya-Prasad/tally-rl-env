"""
The canonical "hello world" RL environment. An agent sits on a 5x5 grid,
starts at (0,0), and must reach the goal at (4,4). Each step it can move
up, down, left, or right. Episode ends when the goal is reached or after
50 steps (whichever first).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """5x5 grid. Agent starts at (0,0). Goal at (4,4). +1 reward at goal."""
    metadata = {"render_modes": ["text"]}

    # actions constants: readability ove magic numbers
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, grid_size:int = 5, max_steps:int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # OBSERVATION SPACE: agent's (row, col) position as 2 integers
        self.observation_space = spaces.MultiDiscrete([grid_size, grid_size])

        # ACTION SPACE: 4 discrete actions
        self.action_space = spaces.Discrete(4)
 
        # Internal state: initialized properly in reset()
        self.agent_pos: np.ndarray = np.array([0, 0])
        self.goal_pos: np.ndarray = np.array([grid_size - 1, grid_size - 1])
        self.step_count: int = 0
 
    def reset(self, seed=None, options=None):
        """Start a new episode. Returns (initial_obs, info_dict)."""
        super().reset(seed=seed)  # seeds self.np_random
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        return self._get_obs(), self._get_info()
 
    def step(self, action: int):
        """Apply action; return (obs, reward, terminated, truncated, info)."""
        self.step_count += 1
 
        # Apply movement; clip to grid bounds (walls)
        if action == self.UP:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == self.DOWN:
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == self.LEFT:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == self.RIGHT:
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
 
        # Reward design: SPARSE — +1 only when goal reached, else 0
        reached_goal = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if reached_goal else 0.0
 
        # Two distinct reasons an episode ends:
        # 1. terminated = "the task itself ended" (goal reached, agent died, etc.)
        # 2. truncated  = "we cut it short for external reasons" (time limit)
        terminated = bool(reached_goal)
        truncated = self.step_count >= self.max_steps
 
        return self._get_obs(), reward, terminated, truncated, self._get_info()
 
    def _get_obs(self):
        return self.agent_pos.copy()
 
    def _get_info(self):
        # info is for diagnostics; agents don't observe it during training.
        # Useful for logging, debugging, and downstream analysis.
        return {
            "step_count": self.step_count,
            "distance_to_goal": int(np.abs(self.agent_pos - self.goal_pos).sum()),
        }
 
    def render(self):
        """Print a text grid showing agent (A) and goal (G)."""
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()
 
 
def demo_random_agent():
    """Run a random policy for 5 episodes and print the outcomes."""
    env = GridWorldEnv()
    for ep in range(5):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()  # random policy
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        outcome = "REACHED GOAL" if terminated else "timed out"
        print(f"Episode {ep + 1}: {outcome} in {info['step_count']} steps "
              f"| total reward = {total_reward}")
 
 
if __name__ == "__main__":
    demo_random_agent()
