import os
import json
import re
from pathlib import Path
from typing import Optional, Dict

import torch

from .config import ForgeLoopGRPOConfig
from .trainer.swap_manager import ModelSwapManager
from .utils.merge_utils import merge_peft_adapter


def train(config_path: str, resume_dir: Optional[str] = None, overrides: Optional[Dict] = None):
    """Main training orchestrator featuring seamless CPU/GPU VRAM hot-swapping."""

    script_dir = Path(__file__).parent.resolve() if '__file__' in dir() else Path(".").resolve()

    target_path = Path(config_path)
    if not target_path.is_absolute() and len(target_path.parts) == 1:
        target_path = script_dir / config_path
    else:
        target_path = target_path.resolve()

    if not target_path.exists():
        print(f"[ForgeLoop Error] Config file not discovered at: {target_path}")
        return

    with open(target_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # Implement nested JSON dot-notation injection overrides safely
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            target = config_dict
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = value

    config = ForgeLoopGRPOConfig(**config_dict)

    # Detect model family for degeneracy auditor configuration
    model_lower = config.model_path.lower()
    is_gemma = "gemma" in model_lower

    # Inject model-family hints into degeneracy auditor config safely
    degen_config = config_dict.setdefault("degeneracy_auditor", {})
    if isinstance(degen_config, dict):
        degen_config["_is_gemma_model"] = is_gemma
        degen_config["gemma_thinking_enabled"] = config_dict.get("gemma_thinking_enabled", True)
        degen_config["think_tag_start"] = config_dict.get("think_tag_start", "<think>")
        degen_config["think_tag_end"] = config_dict.get("think_tag_end", "</think>")
        # Explicit link enforcement back to structural configuration tree
        config_dict["degeneracy_auditor"] = degen_config

    from .utils.model_utils import load_tokenizer_only
    tokenizer = load_tokenizer_only(config.model_path)

    from .utils.data_utils import load_dataset, shuffle_dataset, build_prompt
    dataset = load_dataset(config.dataset_path)
    print(f"[ForgeLoop] Dataset parsed. Found {len(dataset)} items.")
    print(f"[ForgeLoop] Model: {config.model_path} | Gemma detected: {is_gemma}")

    swap = ModelSwapManager(config, tokenizer, config_dict)
    swap.init_training_model_cpu(config.model_path)

    # Automatically sync global_step to match checkpoint name on resume
    global_step = 0
    if resume_dir:
        ckpt = Path(resume_dir) / "custom_lora.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            swap.lora_manager.load_state_dict(state)

            # Extract digits from checkpoint directory string
            step_match = re.search(r"checkpoint-(\d+)", str(resume_dir))
            if step_match:
                global_step = int(step_match.group(1))
            print(f"[ForgeLoop] Resumed from {ckpt}. Global Step sync: {global_step}")

    from .rewards.engine import PolicyRewardOrchestrationEngine
    from .embeddings import CPUEmbeddingStore
    from .trainer.compute import ComputeMixin

    embedding_store = CPUEmbeddingStore(
        model_name=config.performance.embedding_model,
        device=config.performance.embedding_device,
    )
    reward_engine = PolicyRewardOrchestrationEngine(config, embedding_store)

    # Instantiate compute mixin; we dynamically sync the live model instance inside the swap loop
    compute_mixin = ComputeMixin(reward_engine, config, model=None, tokenizer=tokenizer)

    per_step = config.per_device_train_batch_size
    accum = config.gradient_accumulation_steps
    macro_size = per_step * accum

    for epoch in range(config.num_train_epochs):
        print(f"\n{'='*60}")
        print(f"[EPOCH {epoch + 1}/{config.num_train_epochs}]")
        print(f"{'='*60}")

        shuffled = shuffle_dataset(dataset, seed=42 + epoch)

        for i in range(0, len(shuffled), macro_size):
            raw_macro_batch = shuffled[i:i + macro_size]
            if not raw_macro_batch:
                continue

            # Skip trailing fractional batches to keep math group statistics perfectly stable
            if len(raw_macro_batch) < macro_size and global_step > 0:
                print(f"[ForgeLoop Warnings] Dropped trailing batch of size {len(raw_macro_batch)} to avoid bias.")
                continue

            macro_batch = []
            for item in raw_macro_batch:
                formatted_prompt, original_text, meta = build_prompt(global_step, item)
                macro_batch.append({
                    "prompt": formatted_prompt,
                    "text": original_text,
                    "metadata": meta
                })

            print(f"\n[MACRO-BATCH] Logical Steps {global_step + 1}-{global_step + accum} | Engine size: {len(macro_batch)}")

            # --- GENERATION PHASE (VRAM SPIKE 1) ---
            swap.export_lora_for_vllm(swap._lora_staging_path)
            swap.load_vllm(swap._lora_staging_path)
            stored = swap.generate_all_mini_batches(macro_batch, swap._lora_staging_path)
            swap.unload_vllm()

            # --- OPTIMIZATION PHASE (VRAM SPIKE 2) ---
            swap.load_training_to_gpu()
            compute_mixin.model = swap.model  # Dynamically points compute mixin to loaded GPU model graph

            prev_step = global_step  # Log step before the macro-batch evaluation advances it
            avg_loss, global_step = swap.train_on_stored(
                stored, global_step, reward_engine, compute_mixin
            )
            swap.unload_training()
            compute_mixin.model = None  # Clean reference leak protection

            print(f"[MACRO-BATCH] Complete. Base Loss: {avg_loss:.4f} | Current Step: {global_step}")

            # Safe interval tracking for macro-stepped boundary intervals
            if global_step > 0 and (prev_step // config.save_steps) != (global_step // config.save_steps):
                target_step = (global_step // config.save_steps) * config.save_steps
                ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{target_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                state = swap.lora_manager.state_dict()
                torch.save(state, os.path.join(ckpt_dir, "custom_lora.pt"))
                print(f"[Checkpoint Engine] Successfully compiled state checkpoint at step {target_step}")

    # --- FINAL EXPORT METRICS AND SAVES ---
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # 1. Custom weight save
    state = swap.lora_manager.state_dict()
    torch.save(state, os.path.join(final_dir, "custom_lora_final.pt"))

    # 2. Native PEFT output
    peft_output_dir = os.path.join(final_dir, "peft_adapter")
    swap.lora_manager.export_peft_format(peft_output_dir, base_model_path=config.model_path)

    # 3. Structural merge back execution
    if getattr(config, 'auto_merge_final', False):
        print("[ForgeLoop] Auto-merge enabled. Commencing structural parameter union...")
        merge_peft_adapter(
            base_model_path=config.model_path,
            adapter_path=peft_output_dir,
            output_path=os.path.join(final_dir, "merged_model"),
        )

    print("\n[ForgeLoop] Execution concluded safely. Pipeline Targets Ready:")
    print(f"  - Pipeline Weight Payload:  {final_dir}/custom_lora_final.pt")
    print(f"  - Hugging Face PEFT Export: {peft_output_dir}/")
    if getattr(config, 'auto_merge_final', False):
        print(f"  - Compiled Monolith Model: {final_dir}/merged_model/")