import gymnasium
import flappy_bird_gymnasium
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Register the custom environment (same as in main.py)
def env_creator(config):
    import flappy_bird_gymnasium
    env = gymnasium.make("FlappyBird-v0", **config)
    return env

register_env("my_env", env_creator)

#checkpoint_path = "~/ray_results/PPO_FlappyBird-v0_2025-10-13_20-04-28_qeegw1g/checkpoint_000050"
checkpoint_path = "/tmp/tmpzzgwdtbp/"

config = (
    PPOConfig()
    .environment(env="my_env", env_config={"use_lidar": False})
    .framework("torch")
    .env_runners(num_env_runners=0)  # No parallel workers needed for inference
)

algo = config.build()
algo.restore(checkpoint_path)

# Get the RLModule for inference
rl_module = algo.get_module()

# Create the environment with rendering enabled
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    steps = 0

    while not (done or truncated):
        # Get action from trained policy using RLModule
        batch = {"obs": torch.from_numpy(np.array([obs])).float()}
        output = rl_module.forward_inference(batch)
        # Debug: print available keys
        #if steps == 0 and episode == 0:
        #    print(f"Available keys in output: {output.keys()}")
        # Extract action from output
        action = output["action_dist_inputs"].argmax(dim=-1)[0].cpu().numpy()

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

env.close()
