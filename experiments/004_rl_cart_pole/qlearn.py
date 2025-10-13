import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import pickle

env = gym.make("CartPole-v1", render_mode='rgb_array')
print(env)

observation, info = env.reset()
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# Parameters for Q-Learning
alpha = 0.1 # Learning rate
gamma = 0.99 # Discount factor
epsilon = 0.1 # Exploration rate
n_episodes = 30000 # Number of episodes for training
# Initialize Q-table (for discrete states)
n_actions = env.action_space.n
q_table = np.zeros((24, 24, 24, 24, n_actions))
"""
CartPole's state space is continuous,
but we’ll discretize it to make Q-learning feasible. Here, the 4 dimensions of the
state space are divided into 24 bins each
"""
print("Shape of Q-table: ", q_table.shape)

# Define state space boundaries and number of bins for each dimension
state_bins = [
 np.linspace(-2.4, 2.4, 24), # Cart position
 np.linspace(-3.0, 3.0, 24), # Cart velocity
 np.linspace(-0.5, 0.5, 24), # Pole angle
 np.linspace(-2.0, 2.0, 24) # Pole velocity
]
def discretize_state(state):
	"""
	Discretize the continuous state to an index in the Q-table.
	"""
	state_discretized = []
	for i, (s, bins) in enumerate(zip(state, state_bins)):
		# np.digitize maps continuous state value to a bin index
		state_discretized.append(np.digitize(s, bins) - 1)
	return tuple(state_discretized)

rewards = []
for episode in range(n_episodes):
	state, _ = env.reset() # Reset environment to start a new episode
	total_reward = 0
	done = False
	while not done:
		state_discretized = discretize_state(state)

		# Exploration vs Exploitation: Choose action
		if np.random.rand() < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(q_table[state_discretized])

		next_state, reward, terminated, truncated, _ = env.step(action)
		next_state_discretized = discretize_state(next_state)

		# Q-learning update rule
		q_table[state_discretized + (action,)] = q_table[state_discretized + (action,)] + \
            alpha * (reward + gamma * np.max(q_table[next_state_discretized]) - q_table[state_discretized + (action,)])

		total_reward += reward
		state = next_state

		if terminated or truncated:
			done = True

	rewards.append(total_reward)
	if episode % 50 == 0:
		print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward}")

# Plot the reward curve over episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards Over Training Episodes')
plt.savefig("training_rewards.png")
plt.show()

# Save Q-table using pickle
with open('q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)
print("Q-table saved to q_table.pkl")
