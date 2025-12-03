import csv
import os
from datetime import datetime
from typing import Any, Dict
import gymnasium
import numpy as np
import ray
import torch
import torch.nn as nn
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.tune import CLIReporter

from multi_lstm_ensemble import MultiLSTMEnsemble, MultiLSTMEnsembleWithProjection


class MultiLSTMEnsembleRLModule(TorchRLModule):
    """
    RLModule using the MultiLSTMEnsemble architecture for the new RayRLlib API stack.
    This module can be used with any RayRLlib algorithm (PPO, SAC, etc.)
    """

    def setup(self):
        """Initialize the model architecture."""
        # Get model configuration
        model_config = self.model_config.get("model_config_dict", {})

        # LSTM architecture parameters
        self.hidden_size = model_config.get("hidden_size", 128)
        self.num_lstms = model_config.get("num_lstms", 3)
        self.num_layers = model_config.get("num_layers", 1)
        self.final_hidden_size = model_config.get("final_hidden_size", None)
        self.dropout = model_config.get("dropout", 0.0)
        self.bidirectional = model_config.get("bidirectional", False)
        self.use_projection = model_config.get("use_projection", False)
        self.projection_size = model_config.get("projection_size", None)

        # Input size from observation space
        obs_space = self.observation_space
        if isinstance(obs_space, gymnasium.spaces.Box):
            input_size = int(np.product(obs_space.shape))
        else:
            input_size = obs_space.n

        # Create the multi-LSTM ensemble
        if self.use_projection and self.projection_size is not None:
            self.lstm_ensemble = MultiLSTMEnsembleWithProjection(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_lstms=self.num_lstms,
                num_layers=self.num_layers,
                projection_size=self.projection_size,
                final_hidden_size=self.final_hidden_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
        else:
            self.lstm_ensemble = MultiLSTMEnsemble(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_lstms=self.num_lstms,
                num_layers=self.num_layers,
                final_hidden_size=self.final_hidden_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )

        # Get output size from the ensemble
        num_directions = 2 if self.bidirectional else 1
        final_hidden = self.final_hidden_size or self.hidden_size
        lstm_output_size = final_hidden * num_directions

        # Action space size
        action_space = self.action_space
        if isinstance(action_space, gymnasium.spaces.Discrete):
            num_outputs = action_space.n
        else:
            num_outputs = int(np.product(action_space.shape))

        # Policy head (action logits for discrete, mean for continuous)
        self.pi = nn.Linear(lstm_output_size, num_outputs)

        # Value head (state value estimation)
        self.vf = nn.Linear(lstm_output_size, 1)

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inference (action selection)."""
        obs = batch["obs"]

        # Flatten observation if needed and add sequence dimension
        if len(obs.shape) == 2:
            # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)

        # Forward pass through the ensemble
        lstm_output, _ = self.lstm_ensemble(obs)

        # Take the last timestep output
        features = lstm_output[:, -1, :]

        # Get action logits
        action_logits = self.pi(features)

        return {"action_dist_inputs": action_logits}

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for exploration (same as inference for PPO)."""
        return self._forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for training."""
        obs = batch["obs"]

        # Flatten observation if needed and add sequence dimension
        if len(obs.shape) == 2:
            # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)

        # Forward pass through the ensemble
        lstm_output, _ = self.lstm_ensemble(obs)

        # Take the last timestep output
        features = lstm_output[:, -1, :]

        # Get action logits and value
        action_logits = self.pi(features)
        vf_output = self.vf(features).squeeze(-1)

        return {
            "action_dist_inputs": action_logits,
            "vf_preds": vf_output,
        }


def main():
    """
    Main training loop using MultiLSTMEnsemble with RayRLlib PPO.
    Tracks metrics by training steps and saves to CSV for analysis.
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Configure the PPO algorithm with new API stack
    config = (
        PPOConfig()
        .environment(env="CartPole-v1")  # Change this to your environment
        .framework("torch")
        .env_runners(num_env_runners=4)
        .rl_module(
            rl_module_spec={"module_class": MultiLSTMEnsembleRLModule},
            model_config={
                "model_config_dict": {
                    "hidden_size": 64,
                    "num_lstms": 3,
                    "num_layers": 1,
                    "final_hidden_size": 128,
                    "dropout": 0.1,
                    "bidirectional": False,
                    "use_projection": False,
                    "projection_size": None,
                }
            },
        )
        .training(
            train_batch_size=4000,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            vf_clip_param=10.0,
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
    )

    # Build the algorithm
    algo = config.build_algo()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")

    # CSV headers
    csv_headers = [
        'training_steps', 'iteration', 'episode_reward_mean', 'episode_reward_min',
        'episode_reward_max', 'policy_loss', 'value_loss', 'total_loss',
        'entropy', 'kl_divergence', 'vf_explained_var', 'time_elapsed'
    ]

    # Initialize CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # Training loop
    num_iterations = 100
    best_reward = -float('inf')
    start_time = datetime.now()

    model_cfg = config.model_config["model_config_dict"]

    print("\n" + "=" * 80)
    print("Training MultiLSTMEnsemble with RayRLlib PPO (New API Stack)")
    print("=" * 80)
    print(f"Environment: CartPole-v1")
    print(f"Model: MultiLSTMEnsembleRLModule")
    print(f"  - Hidden size: {model_cfg['hidden_size']}")
    print(f"  - Number of LSTMs: {model_cfg['num_lstms']}")
    print(f"  - Final hidden size: {model_cfg['final_hidden_size']}")
    print(f"\nMetrics will be saved to: {csv_file}")
    print("=" * 80)
    print(f"{'Steps':<10} {'Iter':<6} {'Reward':<12} {'P-Loss':<10} {'V-Loss':<10} {'Entropy':<10} {'Conv':<8}")
    print("-" * 80)

    for i in range(num_iterations):
        result = algo.train()

        # Extract metrics
        iteration = result['training_iteration']
        episode_reward_mean = result['env_runners']['episode_return_mean']
        episode_reward_min = result['env_runners']['episode_return_min']
        episode_reward_max = result['env_runners']['episode_return_max']

        learner = result['learners']['default_policy']
        policy_loss = learner['policy_loss']
        vf_loss = learner['vf_loss']
        total_loss = learner['total_loss']
        entropy = learner['entropy']
        kl = learner['mean_kl_loss']
        vf_explained_var = learner['vf_explained_var']

        total_steps = result['learners']['__all_modules__']['num_env_steps_trained_lifetime']
        time_elapsed = (datetime.now() - start_time).total_seconds()

        # Convergence indicator (based on VF explained variance and low KL)
        converged = "✓" if vf_explained_var > 0.8 and kl < 0.01 else ""

        # Compact progress line (print every iteration)
        print(f"{total_steps:<10,} {iteration:<6} {episode_reward_mean:>6.2f} ({episode_reward_min:>3.0f}-{episode_reward_max:>3.0f}) "
              f"{policy_loss:>9.4f} {vf_loss:>9.2f} {entropy:>9.4f} {converged:<8}")

        # Detailed output every 10 iterations
        if i % 10 == 0 or i == num_iterations - 1:
            print("\n" + "-" * 80)
            print(f"Iteration {iteration} | Steps: {total_steps:,} | Time: {time_elapsed:.1f}s")
            print(f"  Reward: {episode_reward_mean:.2f} (min: {episode_reward_min:.0f}, max: {episode_reward_max:.0f})")
            print(f"  Policy Loss: {policy_loss:.4f} | Value Loss: {vf_loss:.4f} | Total: {total_loss:.4f}")
            print(f"  Entropy: {entropy:.4f} | KL: {kl:.6f} | VF Expl. Var: {vf_explained_var:.4f}")
            print("-" * 80 + "\n")

        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow({
                'training_steps': total_steps,
                'iteration': iteration,
                'episode_reward_mean': episode_reward_mean,
                'episode_reward_min': episode_reward_min,
                'episode_reward_max': episode_reward_max,
                'policy_loss': policy_loss,
                'value_loss': vf_loss,
                'total_loss': total_loss,
                'entropy': entropy,
                'kl_divergence': kl,
                'vf_explained_var': vf_explained_var,
                'time_elapsed': time_elapsed
            })

        # Save checkpoints for best performing models
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_dir = algo.save()
            print(f"\n  ★ New best reward! {best_reward:.2f} | Checkpoint: {checkpoint_dir}\n")

        # Save periodic checkpoints
        if i % 25 == 0 and i > 0:
            checkpoint_dir = algo.save()
            print(f"\n  Periodic checkpoint saved at: {checkpoint_dir}\n")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Total training time: {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"Metrics saved to: {csv_file}")
    print("=" * 80)

    # Print summary statistics
    print("\nTo visualize results, run:")
    print(f"  python plot_training.py {csv_file}")

    # Clean up
    algo.stop()
    ray.shutdown()


def hyperparameter_tuning():
    """
    Example of hyperparameter tuning with Ray Tune using new API stack.
    """
    ray.init(ignore_reinit_error=True)

    # Configure with hyperparameter search
    config = (
        PPOConfig()
        .environment(env="CartPole-v1")
        .framework("torch")
        .env_runners(num_env_runners=2)
        .rl_module(
            rl_module_spec={"module_class": MultiLSTMEnsembleRLModule},
            model_config={
                "model_config_dict": {
                    "hidden_size": tune.choice([64, 128]),
                    "num_lstms": tune.choice([2, 3, 4]),
                    "num_layers": 1,
                    "final_hidden_size": tune.choice([128, 256]),
                    "dropout": 0.1,
                    "bidirectional": False,
                    "use_projection": False,
                    "projection_size": None,
                }
            },
        )
        .training(
            train_batch_size=4000,
            lr=tune.grid_search([0.0001, 0.0003, 0.001]),
            gamma=0.99,
            sgd_minibatch_size=tune.grid_search([128, 256]),
            num_sgd_iter=tune.choice([5, 10]),
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
        )
    )

    # Set up tuner
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=train.RunConfig(
            stop={"training_iteration": 50},
            verbose=2,
            progress_reporter=CLIReporter(
                metric_columns={
                    "training_iteration": "iter",
                    "env_runners/episode_return_mean": "reward",
                    "learners/default_policy/policy_loss": "p_loss",
                    "learners/default_policy/vf_loss": "v_loss",
                    "time_total_s": "time(s)",
                },
                parameter_columns=["lr", "sgd_minibatch_size"],
                max_report_frequency=30,
            )
        )
    )

    # Run tuning
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
    print("\n" + "=" * 70)
    print("Best hyperparameters found:")
    print("=" * 70)
    print(best_result.config)

    ray.shutdown()


if __name__ == "__main__":
    # Run standard training
    main()

    # Uncomment to run hyperparameter tuning instead
    # hyperparameter_tuning()
