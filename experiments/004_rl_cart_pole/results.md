# RAY PPO

Pole cart is a physics problem where you can move the cart left and right.
The goal is to keep the pole upright for as long as possible.
More information on the environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/

## RLLib

Starting with this config:

```
```
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
```


=== Iteration 1 ===
Episode Reward: 20.87 (min: 10, max: 49)
Policy Loss: -0.0178
Value Loss: 7.6651
Total Loss: 7.6528
Entropy: 0.6796
KL Divergence: 0.0278
VF Explained Variance: 0.0022
Total Steps Trained: 11025

=== Iteration 100 ===
Episode Reward: 235.36 (min: 34, max: 500)
Policy Loss: -0.0044
Value Loss: 9.4106
Total Loss: 9.4061
Entropy: 0.4722
KL Divergence: 0.0064
VF Explained Variance: 0.0226
Total Steps Trained: 1011181


