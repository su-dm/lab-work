"""
Simple script to plot training metrics from CSV logs.
Usage: python plot_training.py <csv_file>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_metrics(csv_file):
    """Plot training metrics from CSV file."""
    # Read CSV
    df = pd.read_csv(csv_file)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Metrics - {csv_file}', fontsize=14, fontweight='bold')

    # Plot 1: Episode Reward vs Steps
    ax = axes[0, 0]
    ax.plot(df['training_steps'], df['episode_reward_mean'], 'b-', linewidth=2, label='Mean')
    ax.fill_between(df['training_steps'], df['episode_reward_min'],
                     df['episode_reward_max'], alpha=0.3, label='Min-Max Range')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Losses vs Steps
    ax = axes[0, 1]
    ax.plot(df['training_steps'], df['policy_loss'], 'r-', label='Policy Loss', linewidth=2)
    ax.plot(df['training_steps'], df['value_loss'], 'g-', label='Value Loss', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Policy and Value Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Entropy vs Steps
    ax = axes[0, 2]
    ax.plot(df['training_steps'], df['entropy'], 'purple', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy (Exploration)')
    ax.grid(True, alpha=0.3)

    # Plot 4: KL Divergence vs Steps
    ax = axes[1, 0]
    ax.plot(df['training_steps'], df['kl_divergence'], 'orange', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence (Policy Change)')
    ax.grid(True, alpha=0.3)

    # Plot 5: Value Function Explained Variance vs Steps
    ax = axes[1, 1]
    ax.plot(df['training_steps'], df['vf_explained_var'], 'cyan', linewidth=2)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Good Threshold (0.8)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Value Function Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Convergence Speed (Reward per Second)
    ax = axes[1, 2]
    reward_per_step = df['episode_reward_mean'] / (df['training_steps'] + 1)
    ax.plot(df['training_steps'], reward_per_step, 'brown', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward per Step')
    ax.set_title('Learning Efficiency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.show()


def print_convergence_stats(csv_file):
    """Print convergence statistics."""
    df = pd.read_csv(csv_file)

    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    # Find when reward plateaus (95% of max)
    max_reward = df['episode_reward_mean'].max()
    plateau_threshold = 0.95 * max_reward
    plateau_idx = df[df['episode_reward_mean'] >= plateau_threshold].index[0] if len(df[df['episode_reward_mean'] >= plateau_threshold]) > 0 else None

    if plateau_idx is not None:
        plateau_steps = df.loc[plateau_idx, 'training_steps']
        plateau_time = df.loc[plateau_idx, 'time_elapsed']
        print(f"\nReached 95% of max reward ({plateau_threshold:.2f}):")
        print(f"  Steps: {plateau_steps:,}")
        print(f"  Time: {plateau_time:.1f}s")
        print(f"  Iteration: {df.loc[plateau_idx, 'iteration']}")

    # Overall statistics
    print(f"\nFinal Performance:")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Final Reward: {df['episode_reward_mean'].iloc[-1]:.2f}")
    print(f"  Total Steps: {df['training_steps'].iloc[-1]:,}")
    print(f"  Total Time: {df['time_elapsed'].iloc[-1]:.1f}s")

    # Learning rate (reward improvement per 1000 steps)
    steps_array = df['training_steps'].values
    reward_array = df['episode_reward_mean'].values
    if len(steps_array) > 10:
        # Calculate average improvement rate
        total_improvement = reward_array[-1] - reward_array[0]
        total_steps = steps_array[-1] - steps_array[0]
        improvement_per_1k = (total_improvement / total_steps) * 1000
        print(f"  Avg improvement rate: {improvement_per_1k:.3f} reward per 1000 steps")

    # Convergence indicators
    print(f"\nFinal Convergence Metrics:")
    print(f"  VF Explained Variance: {df['vf_explained_var'].iloc[-1]:.4f}")
    print(f"  KL Divergence: {df['kl_divergence'].iloc[-1]:.6f}")
    print(f"  Entropy: {df['entropy'].iloc[-1]:.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        print_convergence_stats(csv_file)
        plot_training_metrics(csv_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
