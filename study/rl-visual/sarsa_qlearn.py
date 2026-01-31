"""
SARSA and Q-Learning Implementations
=====================================

This module implements two classic Temporal Difference (TD) learning algorithms:
- SARSA (State-Action-Reward-State-Action): On-policy TD control
- Q-Learning: Off-policy TD control

Key Difference:
- SARSA updates using the action actually taken (on-policy)
- Q-Learning updates using the best possible action (off-policy)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random


class GridWorld:
    """
    Simple Grid World environment for testing RL algorithms.
    
    Grid layout (5x5):
    S . . . .
    . # . # .
    . . . . .
    . # . # .
    . . . . G
    
    S = Start, G = Goal, # = Obstacle, . = Empty cell
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (1, 3), (3, 1), (3, 3)]
        self.state = self.start
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.action_effects = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to start state."""
        self.state = self.start
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take action and return next state, reward, and done flag.
        
        Returns:
            next_state: New position
            reward: Reward for transition
            done: Whether episode is finished
        """
        row, col = self.state
        d_row, d_col = self.action_effects[action]
        
        new_row = row + d_row
        new_col = col + d_col
        
        # Check boundaries
        if new_row < 0 or new_row >= self.size or new_col < 0 or new_col >= self.size:
            new_row, new_col = row, col  # Stay in place
            reward = -1.0
        # Check obstacles
        elif (new_row, new_col) in self.obstacles:
            new_row, new_col = row, col  # Stay in place
            reward = -1.0
        # Check goal
        elif (new_row, new_col) == self.goal:
            reward = 10.0
        else:
            reward = -0.1  # Small penalty for each step
        
        self.state = (new_row, new_col)
        done = (self.state == self.goal)
        
        return self.state, reward, done


class SARSA:
    """
    SARSA (State-Action-Reward-State-Action) Algorithm
    
    On-policy TD control algorithm that updates Q-values based on the
    action actually taken by the current policy.
    
    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    """
    
    def __init__(self, 
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        """
        Initialize SARSA agent.
        
        Args:
            n_states: Number of states (for gridworld: size * size)
            n_actions: Number of actions
            learning_rate: α (alpha) - learning rate
            discount_factor: γ (gamma) - discount factor
            epsilon: ε for epsilon-greedy policy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))
    
    def state_to_index(self, state: Tuple[int, int], size: int) -> int:
        """Convert (row, col) state to index."""
        return state[0] * size + state[1]
    
    def epsilon_greedy_policy(self, state_idx: int) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state_idx])
    
    def train(self, env: GridWorld, episodes: int = 500) -> List[float]:
        """
        Train using SARSA algorithm.
        
        Returns:
            rewards_per_episode: List of total rewards per episode
        """
        rewards_per_episode = []
        
        for episode in range(episodes):
            state = env.reset()
            state_idx = self.state_to_index(state, env.size)
            action = self.epsilon_greedy_policy(state_idx)
            
            total_reward = 0
            done = False
            
            while not done:
                # Take action
                next_state, reward, done = env.step(action)
                next_state_idx = self.state_to_index(next_state, env.size)
                
                # Choose next action using current policy
                next_action = self.epsilon_greedy_policy(next_state_idx)
                
                # SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                td_target = reward + self.gamma * self.Q[next_state_idx, next_action]
                td_error = td_target - self.Q[state_idx, action]
                self.Q[state_idx, action] += self.alpha * td_error
                
                # Move to next state and action
                state_idx = next_state_idx
                action = next_action
                total_reward += reward
            
            rewards_per_episode.append(total_reward)
        
        return rewards_per_episode
    
    def get_policy(self, size: int) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        policy = np.zeros((size, size), dtype=int)
        for row in range(size):
            for col in range(size):
                state_idx = self.state_to_index((row, col), size)
                policy[row, col] = np.argmax(self.Q[state_idx])
        return policy


class QLearning:
    """
    Q-Learning Algorithm
    
    Off-policy TD control algorithm that updates Q-values based on the
    maximum Q-value of the next state (greedy action), regardless of
    the action actually taken.
    
    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: α (alpha) - learning rate
            discount_factor: γ (gamma) - discount factor
            epsilon: ε for epsilon-greedy exploration
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))
    
    def state_to_index(self, state: Tuple[int, int], size: int) -> int:
        """Convert (row, col) state to index."""
        return state[0] * size + state[1]
    
    def epsilon_greedy_policy(self, state_idx: int) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state_idx])
    
    def train(self, env: GridWorld, episodes: int = 500) -> List[float]:
        """
        Train using Q-Learning algorithm.
        
        Returns:
            rewards_per_episode: List of total rewards per episode
        """
        rewards_per_episode = []
        
        for episode in range(episodes):
            state = env.reset()
            state_idx = self.state_to_index(state, env.size)
            
            total_reward = 0
            done = False
            
            while not done:
                # Choose action using epsilon-greedy policy
                action = self.epsilon_greedy_policy(state_idx)
                
                # Take action
                next_state, reward, done = env.step(action)
                next_state_idx = self.state_to_index(next_state, env.size)
                
                # Q-Learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                # Key difference: uses max Q-value, not the action actually taken
                td_target = reward + self.gamma * np.max(self.Q[next_state_idx])
                td_error = td_target - self.Q[state_idx, action]
                self.Q[state_idx, action] += self.alpha * td_error
                
                # Move to next state
                state_idx = next_state_idx
                total_reward += reward
            
            rewards_per_episode.append(total_reward)
        
        return rewards_per_episode
    
    def get_policy(self, size: int) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        policy = np.zeros((size, size), dtype=int)
        for row in range(size):
            for col in range(size):
                state_idx = self.state_to_index((row, col), size)
                policy[row, col] = np.argmax(self.Q[state_idx])
        return policy


def visualize_results(sarsa_rewards: List[float], 
                     qlearning_rewards: List[float],
                     window: int = 10):
    """Visualize training results for both algorithms."""
    
    # Compute moving average
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    sarsa_ma = moving_average(sarsa_rewards, window)
    qlearning_ma = moving_average(qlearning_rewards, window)
    
    plt.figure(figsize=(12, 5))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_rewards, alpha=0.3, label='SARSA (raw)')
    plt.plot(qlearning_rewards, alpha=0.3, label='Q-Learning (raw)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards (Raw)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    plt.plot(sarsa_ma, label='SARSA', linewidth=2)
    plt.plot(qlearning_ma, label='Q-Learning', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Training Rewards ({window}-episode moving average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("Training comparison plot saved!")


def print_policy(policy: np.ndarray, obstacles: List[Tuple[int, int]], 
                goal: Tuple[int, int]):
    """Print policy in a readable format."""
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    for row in range(policy.shape[0]):
        for col in range(policy.shape[1]):
            if (row, col) in obstacles:
                print('#', end=' ')
            elif (row, col) == goal:
                print('G', end=' ')
            elif (row, col) == (0, 0):
                print('S', end=' ')
            else:
                print(arrows[policy[row, col]], end=' ')
        print()


def main():
    """Main function to demonstrate SARSA and Q-Learning."""
    
    print("=" * 60)
    print("SARSA vs Q-Learning Comparison")
    print("=" * 60)
    
    # Environment setup
    env_size = 5
    n_states = env_size * env_size
    n_actions = 4
    episodes = 500
    
    # Initialize environment
    env_sarsa = GridWorld(size=env_size)
    env_qlearning = GridWorld(size=env_size)
    
    # Initialize agents
    print("\nInitializing agents...")
    sarsa_agent = SARSA(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    qlearning_agent = QLearning(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    # Train agents
    print(f"\nTraining SARSA for {episodes} episodes...")
    sarsa_rewards = sarsa_agent.train(env_sarsa, episodes=episodes)
    
    print(f"Training Q-Learning for {episodes} episodes...")
    qlearning_rewards = qlearning_agent.train(env_qlearning, episodes=episodes)
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    print(f"\nSARSA - Final 10 episodes average reward: {np.mean(sarsa_rewards[-10:]):.2f}")
    print(f"Q-Learning - Final 10 episodes average reward: {np.mean(qlearning_rewards[-10:]):.2f}")
    
    # Extract and display policies
    print("\n" + "-" * 60)
    print("SARSA Learned Policy:")
    print("-" * 60)
    sarsa_policy = sarsa_agent.get_policy(env_size)
    print_policy(sarsa_policy, env_sarsa.obstacles, env_sarsa.goal)
    
    print("\n" + "-" * 60)
    print("Q-Learning Learned Policy:")
    print("-" * 60)
    qlearning_policy = qlearning_agent.get_policy(env_size)
    print_policy(qlearning_policy, env_qlearning.obstacles, env_qlearning.goal)
    
    # Visualize results
    print("\nGenerating comparison plots...")
    visualize_results(sarsa_rewards, qlearning_rewards)
    
    print("\n" + "=" * 60)
    print("Key Differences:")
    print("=" * 60)
    print("SARSA (On-Policy):")
    print("  - Updates based on action actually taken")
    print("  - More conservative, learns safe policy")
    print("  - Update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]")
    print("\nQ-Learning (Off-Policy):")
    print("  - Updates based on maximum Q-value")
    print("  - More aggressive, learns optimal policy")
    print("  - Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    main()
