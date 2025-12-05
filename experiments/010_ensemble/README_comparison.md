# Multi-LSTM Ensemble vs Simple LSTM Comparison

## Overview

This comparison evaluates the Multi-LSTM Ensemble architecture against a simple single LSTM baseline on the CartPole-v1 environment.

## Models Being Compared

### 1. Multi-LSTM Ensemble
- **Architecture**: 3 parallel LSTMs (64 units each) → Concatenation → Final LSTM (128 units)
- **Total Parameters**: ~219,011
- **Configuration**:
  - Hidden size per LSTM: 64
  - Number of parallel LSTMs: 3
  - Final LSTM hidden size: 128
  - Number of layers: 1
  - Dropout: 0.1
  - Bidirectional: False

### 2. Simple LSTM (Baseline)
- **Architecture**: Single 2-layer LSTM (192 units per layer)
- **Configuration**:
  - Hidden size: 192 (comparable to 3×64 ensemble)
  - Number of layers: 2 (to match complexity)
  - Dropout: 0.1
  - Bidirectional: False

## Training Configuration

Both models use identical training hyperparameters:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: CartPole-v1
- **Training iterations**: 50
- **Train batch size**: 4,000
- **Learning rate**: 0.0003
- **Gamma**: 0.99
- **Mini-batch size**: 256
- **Number of epochs**: 10
- **Number of env runners**: 4

## Metrics Collected

For each training iteration, we collect:
1. **Episode Return** (mean, min, max) - Primary metric for convergence
2. **Policy Loss** - How well the policy is learning
3. **Value Loss** - How accurate the value function is
4. **Total Loss** - Combined loss
5. **Entropy** - Exploration vs exploitation balance
6. **KL Divergence** - Policy update magnitude
7. **VF Explained Variance** - Value function quality (target: >0.8)

## Output Files

The comparison script generates:

1. **CSV Files**:
   - `comparison_logs/multi_lstm_ensemble_<timestamp>.csv`
   - `comparison_logs/simple_lstm_<timestamp>.csv`

2. **Plots**:
   - `comparison_logs/model_comparison.png` - 6-panel comparison showing all metrics
   - `comparison_logs/model_comparison_convergence.png` - Focused learning curve comparison

3. **Console Output**:
   - Real-time training progress
   - Summary statistics
   - Performance comparison table

## Key Metrics to Watch

### 1. Convergence Speed
- How quickly each model reaches reward thresholds (50, 100, 150, 195)
- CartPole is "solved" at average reward of 195 over 100 episodes

### 2. Final Performance
- Average reward over last 10 iterations
- Maximum reward achieved

### 3. Stability
- Variance in episode returns
- Consistency of value function (VF explained variance)

### 4. Learning Efficiency
- Training steps needed to reach performance milestones
- Policy loss reduction rate

## Hypothesis

The Multi-LSTM Ensemble is expected to:
- **Potentially converge faster** due to diverse feature representations from parallel LSTMs
- **Show more stable learning** due to ensemble averaging effect
- **Achieve comparable or better final performance** with similar parameter count

However:
- The simple LSTM may be more parameter-efficient per-layer
- The ensemble adds architectural complexity that may or may not provide benefits on simple tasks like CartPole

## Running the Comparison

```bash
# From the lab-work directory
uv run python experiments/010_ensemble/compare_models.py
```

The script will:
1. Train Multi-LSTM Ensemble (50 iterations, ~5-7 minutes)
2. Train Simple LSTM (50 iterations, ~5-7 minutes)
3. Generate comparison plots
4. Print summary statistics

## Interpreting Results

### Good Signs for Ensemble:
- Reaches reward thresholds with fewer training steps
- More consistent episode returns (smaller variance)
- Higher VF explained variance (>0.8 consistently)
- Better final average performance

### Good Signs for Simple LSTM:
- Faster training iterations (simpler architecture)
- Comparable performance with fewer parameters
- More stable policy loss curve

## Files Created

```
experiments/010_ensemble/
├── compare_models.py              # Main comparison script
├── simple_lstm_module.py          # Simple LSTM RLModule
├── rl_multi_lstm_ensemble.py      # Ensemble RLModule
├── multi_lstm_ensemble.py         # Ensemble architecture
├── comparison_logs/               # Output directory
│   ├── multi_lstm_ensemble_*.csv
│   ├── simple_lstm_*.csv
│   ├── model_comparison.png
│   └── model_comparison_convergence.png
└── README_comparison.md           # This file
```

## Next Steps

After the comparison completes:

1. **Analyze the plots** to see which model learns faster
2. **Check summary statistics** for quantitative comparison
3. **Review CSV files** for detailed iteration-by-iteration data
4. **Consider adjustments**:
   - If ensemble underperforms: try different LSTM counts or sizes
   - If ensemble outperforms: test on more complex environments
   - Compare training time vs performance tradeoffs

## Expected Runtime

- Total: ~10-15 minutes
  - Ensemble training: ~5-7 minutes
  - Simple LSTM training: ~5-7 minutes
  - Plotting and analysis: <1 minute
