import pickle
import gymnasium as gym
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


state_bins = [
 np.linspace(-2.4, 2.4, 24), # Cart position
 np.linspace(-3.0, 3.0, 24), # Cart velocity
 np.linspace(-0.5, 0.5, 24), # Pole angle
 np.linspace(-2.0, 2.0, 24) # Pole velocity
]

def discretize_state(state):
	state_discretized = []
	# np.digitize maps continuous state value to a bin index
	for i, (s, bins) in enumerate(zip(state, state_bins)):
		state_discretized.append(np.digitize(s, bins) - 1)
	return tuple(state_discretized)

q_table = None
with open('q_table.pkl', 'rb') as f:
	q_table = pickle.load(f)

env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []
for episode in range(5):
	state, _ = env.reset()
	done = False
	while not done:
		state_discretized = discretize_state(state)
		action = np.argmax(q_table[state_discretized])
		state, reward, terminated, truncated, _ = env.step(action)
		im = env.render()
		frames.append(im)
		if terminated or truncated:
			done = True

fig = plt.figure()
img = plt.imshow(frames[0])
def update(frame):
	img.set_data(frames[frame])
	return img
ani = FuncAnimation(fig, update, frames=len(frames), interval=50)
ani.save("CartPoleTrained.mp4", fps=10, writer='ffmpeg')
plt.show()
env.close()
