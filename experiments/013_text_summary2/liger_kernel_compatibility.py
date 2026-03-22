import torch
from transformers import AutoModelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

def validate_liger_patch(model_id: str = "Qwen/Qwen3.5-4B"):
    print(f"Applying Liger Kernel patches for Qwen3.5...")
    apply_liger_kernel_to_qwen3_5()

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Check if standard layers were replaced by Triton-optimized Liger equivalents.
    expected_liger_types = (
        "LigerRMSNormForQwen3Next",
        "LigerQwen3MoeSwiGLUMLP",
    )
    
    patched_types = set()
    for name, module in model.named_modules():
        module_name = module.__class__.__name__
        if module_name in expected_liger_types:
            patched_types.add(module_name)
            
    # Check if fused linear cross-entropy forward was patched (replaces forward method, not the class)
    lce_patched = "lce_forward" in model.forward.__name__ if hasattr(model.forward, '__name__') else False

    print("\n--- Validation Results ---")
    if patched_types or lce_patched:
        print("SUCCESS: Liger Kernel patches applied:")
        for pt in patched_types:
            print(f"  - {pt}")
        if lce_patched:
            print(f"  - fused_linear_cross_entropy (lce_forward)")
    else:
        print("WARNING: No Liger Kernel modules detected.")

if __name__ == "__main__":
    validate_liger_patch()
