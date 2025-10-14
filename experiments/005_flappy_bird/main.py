import gymnasium
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.tune.registry import register_env

def env_creator(config):
    import flappy_bird_gymnasium  # Import here so workers have access to registration
    env = gymnasium.make("FlappyBird-v0", **config)
    return env

register_env("my_env", env_creator)

config = (
    PPOConfig()
    .environment(env="my_env", env_config={"use_lidar": False})
    #.environment(env="my_env", env_config={"render_mode": "human", "use_lidar": False})
    .framework("torch")
    .env_runners(num_env_runners=8)
    .training(
        train_batch_size=8000,
        lr=0.0005,
        gamma=0.995,
        minibatch_size=512,
        num_epochs=5,
        entropy_coeff=0.01,
        clip_param=0.2,
        vf_clip_param=10.0,
    )
    .evaluation(
        evaluation_interval=10,
        evaluation_duration=10,
    )
)

algo = config.build_algo()

num_iterations = 250
for i in range(num_iterations):
    result = algo.train()

    print(f"\n=== Iteration {result['training_iteration']} ===")
	# Episode performance
    print(f"Episode Reward: {result['env_runners']['episode_return_mean']:.2f} "
          f"(min: {result['env_runners']['episode_return_min']:.0f}, "
          f"max: {result['env_runners']['episode_return_max']:.0f})")
    # Loss metrics
    learner = result['learners']['default_policy']
    print(f"Policy Loss: {learner['policy_loss']:.4f}")
    print(f"Value Loss: {learner['vf_loss']:.4f}")
    print(f"Total Loss: {learner['total_loss']:.4f}")
    print(f"Entropy: {learner['entropy']:.4f}")
    print(f"KL Divergence: {learner['mean_kl_loss']:.4f}")
    print(f"VF Explained Variance: {learner['vf_explained_var']:.4f}")
    # Training progress
    print(f"Total Steps Trained: {result['learners']['__all_modules__']['num_env_steps_trained_lifetime']}")

    if i % 50 == 0 and i > 0:
        checkpoint_dir = algo.save()
        print(f"  Checkpoint saved at: {checkpoint_dir}")

input()
ray.shutdown()
