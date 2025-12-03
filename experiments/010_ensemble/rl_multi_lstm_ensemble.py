import csv
import os
from datetime import datetime
import gymnasium
import numpy as np
import ray
import torch
import torch.nn as nn
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.tune import CLIReporter

from multi_lstm_ensemble import MultiLSTMEnsemble, MultiLSTMEnsembleWithProjection


class MultiLSTMEnsembleRLModel(TorchModelV2, nn.Module):
    """
    RayRLlib compatible model using the MultiLSTMEnsemble architecture.
    This model can be used with any RayRLlib algorithm (PPO, SAC, etc.)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Get custom model configuration
        custom_config = model_config.get("custom_model_config", {})

        # LSTM architecture parameters
        self.hidden_size = custom_config.get("hidden_size", 128)
        self.num_lstms = custom_config.get("num_lstms", 3)
        self.num_layers = custom_config.get("num_layers", 1)
        self.final_hidden_size = custom_config.get("final_hidden_size", None)
        self.dropout = custom_config.get("dropout", 0.0)
        self.bidirectional = custom_config.get("bidirectional", False)
        self.use_projection = custom_config.get("use_projection", False)
        self.projection_size = custom_config.get("projection_size", None)

        # Input size from observation space
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

        # Policy head (action logits)
        self.policy_head = nn.Linear(lstm_output_size, num_outputs)

        # Value head (state value estimation)
        self.value_head = nn.Linear(lstm_output_size, 1)

        # Store the current value output
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass of the model.

        Args:
            input_dict: Dictionary with observation tensor
            state: List of hidden states (for recurrent models)
            seq_lens: Tensor of sequence lengths for each batch element

        Returns:
            policy_logits: Action logits
            state: Updated hidden state
        """
        obs = input_dict["obs"]

        # Flatten observation if needed and add sequence dimension
        if len(obs.shape) == 2:
            # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)

        # Forward pass through the ensemble
        lstm_output, (h_n, c_n) = self.lstm_ensemble(obs)

        # Take the last timestep output
        # Shape: (batch_size, lstm_output_size)
        features = lstm_output[:, -1, :]

        # Get policy logits and value
        policy_logits = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(-1)

        return policy_logits, state

    @override(TorchModelV2)
    def value_function(self):
        """
        Returns the value function output for the most recent forward pass.
        """
        assert self._value_out is not None, "Must call forward() first"
        return self._value_out


def main():
    """
    Main training loop using MultiLSTMEnsemble with RayRLlib PPO.
    Tracks metrics by training steps and saves to CSV for analysis.
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the custom model
    ModelCatalog.register_custom_model("multi_lstm_ensemble", MultiLSTMEnsembleRLModel)

    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment(env="CartPole-v1")  # Change this to your environment
        .framework("torch")
        .env_runners(num_env_runners=4)
        .training(
            train_batch_size=4000,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            minibatch_size=256,
            num_epochs=10,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            model={
                "custom_model": "multi_lstm_ensemble",
                "custom_model_config": {
                    "hidden_size": 64,
                    "num_lstms": 3,
                    "num_layers": 1,
                    "final_hidden_size": 128,
                    "dropout": 0.1,
                    "bidirectional": False,
                    "use_projection": False,
                    "projection_size": None,
                },
            },
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

    print("\n" + "=" * 80)
    print("Training MultiLSTMEnsemble with RayRLlib PPO")
    print("=" * 80)
    print(f"Environment: CartPole-v1")
    print(f"Model: MultiLSTMEnsemble")
    print(f"  - Hidden size: {config['model']['custom_model_config']['hidden_size']}")
    print(f"  - Number of LSTMs: {config['model']['custom_model_config']['num_lstms']}")
    print(f"  - Final hidden size: {config['model']['custom_model_config']['final_hidden_size']}")
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
    Example of hyperparameter tuning with Ray Tune.
    """
    ray.init(ignore_reinit_error=True)

    # Register the custom model
    ModelCatalog.register_custom_model("multi_lstm_ensemble", MultiLSTMEnsembleRLModel)

    # Configure with hyperparameter search
    config = (
        PPOConfig()
        .environment(env="CartPole-v1")
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(
            train_batch_size=4000,
            lr=tune.grid_search([0.0001, 0.0003, 0.001]),
            gamma=0.99,
            minibatch_size=tune.grid_search([128, 256]),
            num_epochs=tune.choice([5, 10]),
            model={
                "custom_model": "multi_lstm_ensemble",
                "custom_model_config": {
                    "hidden_size": tune.choice([64, 128]),
                    "num_lstms": tune.choice([2, 3, 4]),
                    "num_layers": 1,
                    "final_hidden_size": tune.choice([128, 256]),
                    "dropout": 0.1,
                    "bidirectional": False,
                    "use_projection": False,
                },
            },
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
                parameter_columns=["lr", "minibatch_size"],
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
