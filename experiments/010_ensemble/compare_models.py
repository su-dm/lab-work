"""
Compare Multi-LSTM Ensemble vs Simple LSTM on CartPole-v1.

This script trains both models with identical hyperparameters and plots
their learning curves for comparison.
"""
import csv
import os
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rl_multi_lstm_ensemble import MultiLSTMEnsembleRLModule
from simple_lstm_module import SimpleLSTMRLModule


def train_model(
    model_name: str,
    module_class,
    model_config_dict: Dict,
    num_iterations: int = 50,
    log_dir: str = "comparison_logs"
) -> List[Dict]:
    """
    Train a single model and return metrics.

    Args:
        model_name: Name of the model for logging
        module_class: RLModule class to use
        model_config_dict: Model configuration dictionary
        num_iterations: Number of training iterations
        log_dir: Directory to save logs

    Returns:
        List of metric dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    # Configure PPO
    config = (
        PPOConfig()
        .environment(env="CartPole-v1")
        .framework("torch")
        .env_runners(num_env_runners=4)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=module_class,
                model_config={"model_config_dict": model_config_dict}
            ),
        )
        .training(
            train_batch_size=4000,
            lr=0.0003,
            gamma=0.99,
            minibatch_size=256,
            num_epochs=10,
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
    )

    # Build algorithm
    algo = config.build_algo()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, f"{model_name}_{timestamp}.csv")

    # CSV headers
    csv_headers = [
        'iteration', 'training_steps', 'episode_reward_mean', 'episode_reward_min',
        'episode_reward_max', 'policy_loss', 'value_loss', 'total_loss',
        'entropy', 'kl_divergence', 'vf_explained_var', 'time_elapsed'
    ]

    # Initialize CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # Training loop
    metrics_list = []
    start_time = datetime.now()
    best_reward = -float('inf')

    print(f"{'Iter':<6} {'Steps':<10} {'Reward':<12} {'P-Loss':<10} {'V-Loss':<10} {'Entropy':<10}")
    print('-' * 70)

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

        # Store metrics
        metrics = {
            'iteration': iteration,
            'training_steps': total_steps,
            'episode_reward_mean': episode_reward_mean,
            'episode_reward_min': episode_reward_min,
            'episode_reward_max': episode_reward_max,
            'policy_loss': float(policy_loss),
            'value_loss': float(vf_loss),
            'total_loss': float(total_loss),
            'entropy': float(entropy),
            'kl_divergence': float(kl),
            'vf_explained_var': float(vf_explained_var),
            'time_elapsed': time_elapsed
        }
        metrics_list.append(metrics)

        # Print progress
        print(f"{iteration:<6} {total_steps:<10,} {episode_reward_mean:>6.2f} ({episode_reward_min:>3.0f}-{episode_reward_max:>3.0f}) "
              f"{policy_loss:>9.4f} {vf_loss:>9.2f} {entropy:>9.4f}")

        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(metrics)

        # Track best reward
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean

    print(f"\n{model_name} completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final reward: {episode_reward_mean:.2f}")
    print(f"Total time: {time_elapsed:.1f}s")

    # Cleanup
    algo.stop()

    return metrics_list, csv_file


def plot_comparison(
    ensemble_metrics: List[Dict],
    simple_metrics: List[Dict],
    save_path: str = "comparison_plots.png"
):
    """
    Plot comparison graphs between the two models.

    Args:
        ensemble_metrics: Metrics from ensemble model
        simple_metrics: Metrics from simple LSTM model
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-LSTM Ensemble vs Simple LSTM Comparison', fontsize=16, fontweight='bold')

    # Extract data
    ensemble_iters = [m['iteration'] for m in ensemble_metrics]
    ensemble_steps = [m['training_steps'] for m in ensemble_metrics]
    simple_iters = [m['iteration'] for m in simple_metrics]
    simple_steps = [m['training_steps'] for m in simple_metrics]

    # 1. Episode Reward (most important for convergence)
    ax = axes[0, 0]
    ax.plot(ensemble_steps, [m['episode_reward_mean'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=2, marker='o', markersize=3)
    ax.plot(simple_steps, [m['episode_reward_mean'] for m in simple_metrics],
            label='Simple LSTM', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Episode Return (Convergence Speed)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Policy Loss
    ax = axes[0, 1]
    ax.plot(ensemble_steps, [m['policy_loss'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=2, marker='o', markersize=3)
    ax.plot(simple_steps, [m['policy_loss'] for m in simple_metrics],
            label='Simple LSTM', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Value Loss
    ax = axes[0, 2]
    ax.plot(ensemble_steps, [m['value_loss'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=2, marker='o', markersize=3)
    ax.plot(simple_steps, [m['value_loss'] for m in simple_metrics],
            label='Simple LSTM', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Total Loss
    ax = axes[1, 0]
    ax.plot(ensemble_steps, [m['total_loss'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=2, marker='o', markersize=3)
    ax.plot(simple_steps, [m['total_loss'] for m in simple_metrics],
            label='Simple LSTM', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Value Function Explained Variance
    ax = axes[1, 1]
    ax.plot(ensemble_steps, [m['vf_explained_var'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=2, marker='o', markersize=3)
    ax.plot(simple_steps, [m['vf_explained_var'] for m in simple_metrics],
            label='Simple LSTM', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('VF Explained Variance')
    ax.set_title('Value Function Quality', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Good threshold')

    # 6. Entropy
    ax = axes[1, 2]
    ax.plot(ensemble_steps, [m['entropy'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=2, marker='o', markersize=3)
    ax.plot(simple_steps, [m['entropy'] for m in simple_metrics],
            label='Simple LSTM', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy (Exploration)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")

    # Also create a focused convergence comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ensemble_steps, [m['episode_reward_mean'] for m in ensemble_metrics],
            label='Multi-LSTM Ensemble', linewidth=3, marker='o', markersize=4, alpha=0.8)
    ax.plot(simple_steps, [m['episode_reward_mean'] for m in simple_metrics],
            label='Simple LSTM', linewidth=3, marker='s', markersize=4, alpha=0.8)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Learning Curve Comparison: Multi-LSTM Ensemble vs Simple LSTM',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add CartPole solved threshold
    ax.axhline(y=195, color='g', linestyle='--', alpha=0.5, linewidth=2, label='Solved (195)')
    ax.legend(fontsize=12)

    convergence_path = save_path.replace('.png', '_convergence.png')
    plt.tight_layout()
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {convergence_path}")


def print_summary_stats(ensemble_metrics: List[Dict], simple_metrics: List[Dict]):
    """Print summary statistics comparing both models."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Find when each model reaches certain thresholds
    thresholds = [50, 100, 150, 195]

    print("\nSteps to Reach Reward Thresholds:")
    print(f"{'Threshold':<12} {'Multi-LSTM Ensemble':<25} {'Simple LSTM':<25} {'Winner':<15}")
    print("-" * 80)

    for threshold in thresholds:
        ensemble_steps = next((m['training_steps'] for m in ensemble_metrics
                              if m['episode_reward_mean'] >= threshold), None)
        simple_steps = next((m['training_steps'] for m in simple_metrics
                            if m['episode_reward_mean'] >= threshold), None)

        if ensemble_steps and simple_steps:
            winner = "Ensemble" if ensemble_steps < simple_steps else "Simple LSTM"
            speedup = abs(simple_steps - ensemble_steps) / max(simple_steps, ensemble_steps) * 100
            print(f"{threshold:<12} {ensemble_steps:<25,} {simple_steps:<25,} {winner} ({speedup:.1f}% faster)")
        elif ensemble_steps:
            print(f"{threshold:<12} {ensemble_steps:<25,} {'Not reached':<25} Ensemble")
        elif simple_steps:
            print(f"{threshold:<12} {'Not reached':<25} {simple_steps:<25,} Simple LSTM")
        else:
            print(f"{threshold:<12} {'Not reached':<25} {'Not reached':<25} Neither")

    # Final performance
    print("\nFinal Performance:")
    ensemble_final = ensemble_metrics[-1]
    simple_final = simple_metrics[-1]

    print(f"Multi-LSTM Ensemble: {ensemble_final['episode_reward_mean']:.2f} "
          f"(steps: {ensemble_final['training_steps']:,})")
    print(f"Simple LSTM:         {simple_final['episode_reward_mean']:.2f} "
          f"(steps: {simple_final['training_steps']:,})")

    # Best performance
    ensemble_best = max(ensemble_metrics, key=lambda x: x['episode_reward_mean'])
    simple_best = max(simple_metrics, key=lambda x: x['episode_reward_mean'])

    print(f"\nBest Performance:")
    print(f"Multi-LSTM Ensemble: {ensemble_best['episode_reward_mean']:.2f} "
          f"at iteration {ensemble_best['iteration']}")
    print(f"Simple LSTM:         {simple_best['episode_reward_mean']:.2f} "
          f"at iteration {simple_best['iteration']}")

    # Average performance over last 10 iterations
    ensemble_avg = np.mean([m['episode_reward_mean'] for m in ensemble_metrics[-10:]])
    simple_avg = np.mean([m['episode_reward_mean'] for m in simple_metrics[-10:]])

    print(f"\nAverage Performance (last 10 iterations):")
    print(f"Multi-LSTM Ensemble: {ensemble_avg:.2f}")
    print(f"Simple LSTM:         {simple_avg:.2f}")

    if ensemble_avg > simple_avg:
        improvement = (ensemble_avg - simple_avg) / simple_avg * 100
        print(f"\nEnsemble is {improvement:.1f}% better on average")
    else:
        improvement = (simple_avg - ensemble_avg) / ensemble_avg * 100
        print(f"\nSimple LSTM is {improvement:.1f}% better on average")


def main():
    """Main comparison function."""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Configuration
    NUM_ITERATIONS = 50
    LOG_DIR = "comparison_logs"

    # Multi-LSTM Ensemble configuration
    ensemble_config = {
        "hidden_size": 64,
        "num_lstms": 3,
        "num_layers": 1,
        "final_hidden_size": 128,
        "dropout": 0.1,
        "bidirectional": False,
        "use_projection": False,
        "projection_size": None,
    }

    # Simple LSTM configuration (comparable parameter count)
    # The ensemble has 3 LSTMs of size 64 -> 192 total hidden units
    # Plus a final LSTM of 128. Let's use a single 192-unit LSTM for fair comparison
    simple_config = {
        "hidden_size": 192,
        "num_layers": 2,  # 2 layers to match complexity
        "dropout": 0.1,
        "bidirectional": False,
    }

    print("\n" + "="*80)
    print("MODEL COMPARISON: Multi-LSTM Ensemble vs Simple LSTM")
    print("="*80)
    print(f"\nEnvironment: CartPole-v1")
    print(f"Training iterations: {NUM_ITERATIONS}")
    print(f"\nMulti-LSTM Ensemble Config:")
    for k, v in ensemble_config.items():
        print(f"  {k}: {v}")
    print(f"\nSimple LSTM Config:")
    for k, v in simple_config.items():
        print(f"  {k}: {v}")

    # Train Multi-LSTM Ensemble
    ensemble_metrics, ensemble_csv = train_model(
        model_name="multi_lstm_ensemble",
        module_class=MultiLSTMEnsembleRLModule,
        model_config_dict=ensemble_config,
        num_iterations=NUM_ITERATIONS,
        log_dir=LOG_DIR
    )

    # Train Simple LSTM
    simple_metrics, simple_csv = train_model(
        model_name="simple_lstm",
        module_class=SimpleLSTMRLModule,
        model_config_dict=simple_config,
        num_iterations=NUM_ITERATIONS,
        log_dir=LOG_DIR
    )

    # Plot comparison
    plot_path = os.path.join(LOG_DIR, "model_comparison.png")
    plot_comparison(ensemble_metrics, simple_metrics, save_path=plot_path)

    # Print summary statistics
    print_summary_stats(ensemble_metrics, simple_metrics)

    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"CSV files saved:")
    print(f"  - {ensemble_csv}")
    print(f"  - {simple_csv}")
    print(f"Plots saved:")
    print(f"  - {plot_path}")
    print(f"  - {plot_path.replace('.png', '_convergence.png')}")
    print("="*80)

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
