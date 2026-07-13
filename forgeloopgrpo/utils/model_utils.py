"""Model setup — tokenizer only. vLLM and training model are managed separately.

Updated for dual-model support: Qwen3.5-4B and Gemma-4-E2B-IT.
"""

from typing import Tuple, Any
from transformers import AutoTokenizer
import torch


def load_tokenizer_only(model_path: str):
    """Load only tokenizer — model is handled by vLLM (gen) or AutoModelForCausalLM (train).

    Works for both Qwen3.5 and Gemma 4 families.
    """
    # Removed fix_mistral_regex argument to prevent runtime instantiation TypeErrors
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def init_vllm_and_extract_model(model_path: str, config) -> Tuple[Any, Any, None]:
    """Initialize vLLM engine and return (engine, SamplingParams class, None).

    Engine is configured for LoRA hot-swap with params from config.vllm.

    Supports both Qwen3.5 and Gemma 4 model families with automatic dtype
    and architecture detection.
    """
    from vllm import LLM, SamplingParams

    vllm_cfg = config.vllm
    lora_cfg = config.lora

    dtype = vllm_cfg.dtype
    if dtype == "auto":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    # Detect model family for architecture-specific hints
    model_lower = model_path.lower()
    is_gemma = "gemma" in model_lower

    if is_gemma:
        print(f"[Model Utils] Gemma 4 model detected: {model_path}")
        print(f"[Model Utils] Gemma 4 uses alternating sliding/global attention.")
        print(f"[Model Utils] Ensure vLLM >= 0.22.1 with Gemma 4 support.")

    engine = LLM(
        model=model_path,
        tensor_parallel_size=vllm_cfg.tensor_parallel_size,
        gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
        max_model_len=vllm_cfg.max_model_len,
        dtype=dtype,
        swap_space=vllm_cfg.swap_space,
        max_num_seqs=vllm_cfg.max_num_seqs,
        enable_lora=True,
        max_lora_rank=lora_cfg.r,
        max_loras=1,
        max_cpu_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
    )

    return engine, SamplingParams, None