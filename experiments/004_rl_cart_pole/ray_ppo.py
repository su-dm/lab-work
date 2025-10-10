import ray
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()
config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .framework("torch")
    .env_runners(num_env_runners=2)
    .training(
        train_batch_size=500,
        lr=0.0003,
        gamma=0.99,
        minibatch_size=128,
        num_epochs=5,
    )
    .evaluation(
        evaluation_interval=10,
        evaluation_duration=10,
    )
)

algo = config.build_algo()

num_iterations = 100
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

    if i == 50:
        checkpoint_dir = algo.save()
        print(f"  Checkpoint saved at: {checkpoint_dir}")

input()
ray.shutdown()
