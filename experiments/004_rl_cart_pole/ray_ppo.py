import ray
from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint

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
        lambda_=0.95,
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
    
    print(f"Iteration {i+1}/{num_iterations}")
    pprint(result)
    #print(f"  Episode Reward Mean: {result['env_runners/episode_reward_mean']:.2f}")
    #print(f"  Episodes This Iter: {result['env_runners/episodes_this_iter']}")
    
    if i == 50:
        checkpoint_dir = algo.save()
        print(f"  Checkpoint saved at: {checkpoint_dir}")

ray.shutdown()