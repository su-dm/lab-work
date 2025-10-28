"""
Analyzing the learning and training part of nanochat.

scripts/base_train.py

Parameters

Model:
depth

Training:
num_iterations
target_flops
target_param_data_ratio

Optimization:
device_batch_size
total_batch_size
embedding_lr / unembedding_lr
weight_decay
matrix_lr
grad_clip
warmup_ratio
warmdown_ratio
final_lr_frac

Initialize compute, model, optimizers, tokenizer
min_val_bpb
smooth_train_loss
ema_beta

Different evals metrics:
eval bpb
CORE metric
also sometimes sample

Actual training loop:
micro_step in range(grad_accum_steps)
loss = model(x, y)
train_loss = loss.detach()
loss = loss / grad_accum_steps
loss.backward()
clip gradient
get learning rate multiplier
step the optimizer
model.zero_grad
syncronize()

logging
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })
"""

"""
Additional training files
scripts:
chat_rl.py
chat_sft.py
mid_train.py
tok_train.py

├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
"""

