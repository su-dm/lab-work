from vllm import LLM, SamplingParams

#llm = LLM(
#    model="Qwen/Qwen3-4B-Instruct-2507",
#    quantization="fp8",  # fp8 reduces memory without needing special model
#    gpu_memory_utilization=0.95,  # Use more VRAM efficiently
#    # enable_chunked_prefill=True,  # Already enabled by default
#    # max_num_batched_tokens=8192,  # Limit tokens processed at once
#    # kv_cache_dtype="fp8",  # Reduce KV cache memory usage
#    enforce_eager=False,  # Use CUDA graphs for better memory efficiency
#    tensor_parallel_size=1
#    )

#llm = LLM(
#    model="Qwen/Qwen3-4B-Instruct-2507",
#    quantization="fp8",
#    kv_cache_dtype="fp8",
#    gpu_memory_utilization=0.5,     # Leave room for paging
#    max_model_len=80000,
#    enable_chunked_prefill=True,
#    max_num_batched_tokens=512,
#    swap_space=16,                   # Heavy CPU swap (requires RAM)
#    block_size=16,                   # Smaller blocks for better paging
#    kv_cache_memory_bytes=1024*1024*1024*5.5
#)

llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    quantization="fp8",
    kv_cache_dtype="fp8",
    gpu_memory_utilization=0.3,     # Very conservative
    max_model_len=40000,            # Half of 80K
    enable_chunked_prefill=True,
    max_num_batched_tokens=256,     # Even smaller chunks
    swap_space=20,                  # Max CPU offload
    block_size=8,                   # Smaller blocks
    cpu_offload_gb=3,               # Offload some model weights to CPU (if supported)
)

with open("/home/djole/code/lab-work/experiments/008_tiny_inference/uber_v_heller.txt", "r") as f:
    text = f.read()
    outputs = llm.generate([text], SamplingParams(max_tokens=256))
    print(outputs[0].outputs[0].text)