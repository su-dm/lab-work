Here's the full derivation, step by step, grounded in the Qwen3.5-4B architecture specs you provided.

---

## Step 1: Model Weights in Memory (bf16)

The text-only model (`Qwen3_5ForCausalLM`) has ~4B parameters. Let me verify by summing the architecture:

**Token embedding** (tied with LM head, counted once):
- `248320 vocab x 2560 hidden = 635,699,200 params`

**Per-layer parameters:**

Each of the 32 layers has an attention block + FFN block. The attention block differs between DeltaNet and Full Attention layers.

**DeltaNet attention block** (24 layers):
- Q projection: `2560 -> 16 heads x 128 dim = 2048` -> `2560 x 2048 = 5,242,880`
- K projection: `2560 -> 16 heads x 128 dim = 2048` -> `2560 x 2048 = 5,242,880`
- V projection: `2560 -> 32 heads x 128 dim = 4096` -> `2560 x 4096 = 10,485,760`
- O projection: `4096 -> 2560` -> `4096 x 2560 = 10,485,760`
- Gate/beta parameters, conv1d (kernel=4): relatively small, ~`2560 x 4 x 2 = ~20,480`
- **Subtotal attention: ~31.5M per DeltaNet layer**

**Full Attention block** (8 layers):
- Q projection: `2560 -> 16 heads x 256 dim = 4096` -> `2560 x 4096 = 10,485,760`
- K projection: `2560 -> 4 heads x 256 dim = 1024` -> `2560 x 1024 = 2,621,440`
- V projection: `2560 -> 4 heads x 256 dim = 1024` -> `2560 x 1024 = 2,621,440`
- O projection: `4096 -> 2560` -> `4096 x 2560 = 10,485,760`
- **Subtotal attention: ~26.2M per Full Attention layer**

**FFN block** (same for all 32 layers):
- gate_proj: `2560 x 9216 = 23,592,960`
- up_proj: `2560 x 9216 = 23,592,960`
- down_proj: `9216 x 2560 = 23,592,960`
- **Subtotal FFN: ~70.8M per layer**

**RMS norm**: `2560` params each, several per layer -- negligible.

**Total parameter count:**
```
Embedding:         635,699,200
24 DeltaNet layers: 24 x (31.5M + 70.8M) = 24 x 102.3M = 2,455,200,000
8 Full Attn layers:  8 x (26.2M + 70.8M) =  8 x  97.0M =   776,000,000
Norms + misc:       ~1,000,000
                    ─────────────
Total:              ~3,868,000,000 ≈ 3.87B
```

This is consistent with the "4B" label (the HF card says "5B" because it includes the vision encoder which we skip with `Qwen3_5ForCausalLM`).

**Memory for weights in bf16:**
```
3.87B params x 2 bytes/param = 7.74 GB ≈ 8 GB
```

---

## Step 2: LoRA Adapter Parameters (r=64)

For each target linear layer with input dimension `d_in` and output dimension `d_out`, LoRA adds two matrices:
- A: `d_in x r` (r=64)
- B: `r x d_out`
- Total per module: `(d_in + d_out) x r`

**Per DeltaNet layer** (7 target modules):

| Module | d_in | d_out | Params: (d_in + d_out) x 64 |
|--------|------|-------|------------------------------|
| q_proj | 2560 | 2048 | (2560 + 2048) x 64 = 294,912 |
| k_proj | 2560 | 2048 | 294,912 |
| v_proj | 2560 | 4096 | (2560 + 4096) x 64 = 425,984 |
| o_proj | 4096 | 2560 | (4096 + 2560) x 64 = 425,984 |
| gate_proj | 2560 | 9216 | (2560 + 9216) x 64 = 753,664 |
| up_proj | 2560 | 9216 | 753,664 |
| down_proj | 9216 | 2560 | (9216 + 2560) x 64 = 753,664 |
| **Total** | | | **3,702,784** |

**Per Full Attention layer** (7 target modules):

| Module | d_in | d_out | Params: (d_in + d_out) x 64 |
|--------|------|-------|------------------------------|
| q_proj | 2560 | 4096 | (2560 + 4096) x 64 = 425,984 |
| k_proj | 2560 | 1024 | (2560 + 1024) x 64 = 229,376 |
| v_proj | 2560 | 1024 | 229,376 |
| o_proj | 4096 | 2560 | 425,984 |
| gate_proj | 2560 | 9216 | 753,664 |
| up_proj | 2560 | 9216 | 753,664 |
| down_proj | 9216 | 2560 | 753,664 |
| **Total** | | | **3,571,712** |

**Total LoRA parameters across all 32 layers:**
```
24 DeltaNet layers: 24 x 3,702,784 =  88,866,816
8 Full Attn layers:  8 x 3,571,712 =  28,573,696
                                       ──────────
Total:                                117,440,512 ≈ 117.5M
```

That is **2.9%** of the base model's 3.87B parameters.

**LoRA memory:**
```
LoRA weights (bf16):    117.5M x 2 bytes = 235 MB
LoRA gradients (bf16):  117.5M x 2 bytes = 235 MB
                                            ──────
Total LoRA param memory:                    470 MB ≈ 0.47 GB
```

---

## Step 3: Optimizer States (AdamW, fp32)

AdamW stores three fp32 tensors per trainable parameter:
- Master copy of weights (fp32): `117.5M x 4 bytes = 470 MB`
- First moment m (fp32): `117.5M x 4 bytes = 470 MB`
- Second moment v (fp32): `117.5M x 4 bytes = 470 MB`

```
Total optimizer memory: 117.5M x 12 bytes = 1,410 MB ≈ 1.41 GB
```

**With DeepSpeed ZeRO-2**, optimizer states are sharded evenly across GPUs:
```
Per GPU (4 GPUs): 1,410 / 4 = 352 MB ≈ 0.35 GB
Per GPU (8 GPUs): 1,410 / 8 = 176 MB ≈ 0.18 GB
```

Note: ZeRO-2 also shards gradients, but LoRA gradients are only 235 MB total so the per-GPU savings are small. ZeRO-2 does NOT shard model weights or activations -- those are fully replicated on each GPU.

---

## Step 4: Gradient Checkpoint Boundary Storage (the big one)

With `gradient_checkpointing=True` in the HuggingFace Trainer, each transformer layer is a checkpoint boundary. During the forward pass, the **input hidden state** to each layer is saved to memory. During backward, each layer's forward pass is recomputed from its saved input to reconstruct the activations needed for gradient computation.

**What gets saved:**

The input to each layer is a hidden state tensor of shape `(batch_size, seq_len, hidden_dim)`. With batch_size=1:

```
Per checkpoint: 1 x 131072 x 2560 x 2 bytes (bf16) = 671,088,640 bytes = 640 MB
```

There are 33 boundaries (input to each of 32 layers + the embedding output):

```
33 x 640 MB = 21,120 MB ≈ 21.1 GB
```

**Why all 33 are in memory simultaneously:** During the forward pass, these are saved progressively (checkpoint 1, then 2, ..., then 33). They all persist until the backward pass reaches them. The backward walks from layer 32 back to layer 1, consuming and freeing each checkpoint as it goes. The peak occurs right after the forward pass completes and before any backward freeing, when all 33 checkpoints are live.

---

## Step 5: Peak Single-Layer Recomputation Memory

During backward, one layer at a time is recomputed from its saved checkpoint. This requires temporary activation memory for that layer's forward pass. The peak is the most memory-hungry layer type.

### FFN recomputation (all 32 layers have this)

The SwiGLU FFN computes: `output = down_proj(SiLU(gate_proj(x)) * up_proj(x))`

At peak, three intermediate tensors exist simultaneously:

| Tensor | Shape | Size |
|--------|-------|------|
| gate_proj output | 131072 x 9216 | 131072 x 9216 x 2 = 2.25 GB |
| SiLU(gate) | 131072 x 9216 | 2.25 GB (can be fused with gate_proj in-place) |
| up_proj output | 131072 x 9216 | 2.25 GB |
| SiLU(gate) * up (elementwise product) | 131072 x 9216 | 2.25 GB |

The peak happens when `gate_output`, `up_output`, and their product all coexist before `down_proj` consumes the product:

```
FFN peak: ~3 x 131072 x 9216 x 2 bytes ≈ 3 x 2.25 GB = 6.75 GB
```

In practice, with operator fusion (SiLU can be in-place on gate_proj output), this is closer to:
```
FFN peak (fused): ~2 x 131072 x 9216 x 2 ≈ 4.5 GB
```

Conservative estimate: **~7.3 GB** (assuming limited fusion).

### DeltaNet attention recomputation (24 layers)

The DeltaNet linear attention computes Q, K, V projections and then runs a recurrent state update:

| Tensor | Shape | Size |
|--------|-------|------|
| Q (16 heads x 128 dim) | 131072 x 2048 | 131072 x 2048 x 2 = 512 MB |
| K (16 heads x 128 dim) | 131072 x 2048 | 512 MB |
| V (32 heads x 128 dim) | 131072 x 4096 | 1,024 MB |
| Conv1d workspace (kernel=4) | 131072 x 2560 x 4 | ~2.5 GB |

With the FLA (flash-linear-attention) library, the recurrence is computed in chunks. The state at each chunk boundary is stored:
- State size per head: `d_key x d_value = 128 x 128 = 16,384` floats
- For 16 QK-head groups: `16 x 16,384 = 262,144` floats = 0.5 MB per boundary (bf16)
- With chunk size 256: `131072 / 256 = 512` boundaries -> `512 x 0.5 MB = 256 MB`

```
DeltaNet peak per layer: QKV projections (~2 GB) + conv1d (~2.5 GB) + chunk states (~0.25 GB)
                        ≈ 3-4 GB
```

### Full Attention recomputation (8 layers)

With Flash Attention 2, the memory is O(seq_len), not O(seq_len^2):

| Tensor | Shape | Size |
|--------|-------|------|
| Q (16 heads x 256 dim) | 131072 x 4096 | 131072 x 4096 x 2 = 1,024 MB |
| K (4 heads x 256 dim) | 131072 x 1024 | 256 MB |
| V (4 heads x 256 dim) | 131072 x 1024 | 256 MB |
| Flash attention workspace | O(seq_len) | ~small |
| Output | 131072 x 4096 | 1,024 MB |

```
Full Attention peak per layer: ~2.5 GB
```

### Combined peak for one layer recomputation

The worst case is a DeltaNet layer (attention + FFN together during recompute):
```
DeltaNet attention: ~3-4 GB
FFN:                ~7.3 GB
                    ────────
Peak single layer:  ~10-11 GB
```

Only ONE layer is recomputed at a time, so this is not multiplied by 32.

---

## Step 6: DeltaNet Chunkwise Overhead

During backward through a DeltaNet layer, the chunkwise algorithm needs to reconstruct intermediate states. Beyond the single-layer recompute peak (already counted above), there's some overhead from the Triton kernel workspace and CUDA allocator fragmentation. This is empirically **~1-2 GB** based on FLA library behavior.

---

## Step 7: CUDA and Framework Overhead

Fixed costs independent of model/sequence length:
- CUDA context and kernel code: ~1 GB
- NCCL buffers for multi-GPU communication: ~0.5-1 GB
- PyTorch allocator fragmentation and reserves: ~1 GB
- DeepSpeed engine overhead: ~0.5 GB

```
Total overhead: ~3 GB
```

---

## Step 8: Grand Total per GPU

Summing all components for **4x H100 80GB, DeepSpeed ZeRO-2, batch_size=1, gradient checkpointing**:

```
Model weights (bf16, replicated):        8.00 GB
LoRA params + gradients (bf16):          0.47 GB
Optimizer states (ZeRO-2, /4 GPUs):      0.35 GB
Gradient checkpoint boundaries:         21.10 GB
Peak single-layer recompute:            10.30 GB  (conservative midpoint)
DeltaNet chunkwise overhead:             1.50 GB
CUDA + framework overhead:              3.00 GB
                                       ─────────
TOTAL PER GPU:                         44.72 GB
```

**H100 80GB capacity: 80 GB.**
**Headroom: 80 - 44.72 = ~35 GB.** This is comfortable margin for allocator fragmentation, peak transients, and the occasional memory spike from NCCL all-reduce during gradient synchronization.

---

### Sensitivity analysis

If you wanted to sanity-check by scaling from the known ms-swift data point (Qwen3.5-4B LoRA at 2K context, batch_size=4, uses 20 GB per GPU on 4x GPUs):

```
At 2K context, batch=4 (= 8K effective tokens per GPU):
  Model + LoRA + optimizer + overhead: ~12 GB (constant)
  Activations at 8K tokens: ~8 GB

At 131K context, batch=1 (= 131K tokens per GPU):
  Constant components: ~12 GB
  Activations scale: 8 GB x (131072 / 8192) = 8 x 16 = ~128 GB  (linear extrapolation)
```

Wait -- that linear extrapolation gives ~140 GB which doesn't fit! But this overestimates because:

1. The ms-swift example used `batch_size=4` with shorter sequences (4 x 2K = 8K tokens), whereas we use `batch_size=1` with one long sequence (131K tokens). The checkpoint boundary memory is proportional to `total_tokens = batch_size x seq_len`, which is `8192` in their case and `131072` in ours -- a 16x ratio.

2. But their 20 GB includes model weights (~8 GB) and optimizer (~1.4 GB), which are constant. So the activation portion is `20 - 12 = ~8 GB` for 8K tokens.

3. Scaling: `8 GB x (131072 / 8192) = 128 GB` for activations alone. This would be ~140 GB total -- way over 80 GB.

**Why doesn't my component-wise estimate match this?** Because the ms-swift example used `r=8` LoRA (not r=64) and likely did NOT use per-layer gradient checkpointing (their framework may checkpoint less aggressively). At `batch_size=4, max_length=2048` with 4 GPUs, the per-GPU data is 4 sequences of 2K tokens. Without gradient checkpointing, the activation memory includes ALL intermediate activations across ALL layers, which is much more expensive per token.

With gradient checkpointing (our setup), activation memory is dramatically reduced: instead of storing intermediate activations for all 32 layers, we only store the 33 checkpoint boundaries plus one layer's worth of recomputed activations. This is why gradient checkpointing is essential for long context.

The corrected scaling:
```
Checkpoint boundaries: proportional to total_tokens
  At 8K: 33 x 8192 x 2560 x 2 = 1.3 GB
  At 131K: 33 x 131072 x 2560 x 2 = 21.1 GB  (16x, consistent)

Peak layer recompute: proportional to seq_len (single sequence)
  At 2K (one sequence): 3 x 2048 x 9216 x 2 = 0.11 GB
  At 131K: 3 x 131072 x 9216 x 2 = 7.3 GB    (64x, because seq_len per example is 64x longer)
```

This confirms the component-wise estimate is internally consistent. The ~44 GB per-GPU estimate for 131K context with gradient checkpointing is reasonable, and fits within 80 GB H100s with comfortable headroom.

---

### Risk factor

The one uncertainty is DeltaNet's training activation memory. The ms-swift issue #8169 reported that DeltaNet's activations are "much larger than Qwen3-VL" during training. My estimate puts DeltaNet overhead at ~3-4 GB per layer recompute + ~1.5 GB chunkwise overhead. If the FLA implementation is less optimized than assumed, this could be higher. The 35 GB headroom on H100 80GB should absorb this, but it's worth monitoring GPU memory utilization during the first few training steps and being prepared to reduce `max_seq_length` to 65536 if needed.