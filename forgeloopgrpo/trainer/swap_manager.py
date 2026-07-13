"""CPU/GPU swap manager for memory-safe GRPO training. INVARIANT: Only ONE of {vllm_engine, training_model} is in GPU VRAM at any time."""

import os
import gc
import random
import time
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch

from ..config import ForgeLoopGRPOConfig
from .generation import GenerationMixin


class ModelSwapManager:
    """Manages CPU/GPU swap for memory-safe GRPO training."""

    def __init__(self, config: ForgeLoopGRPOConfig, tokenizer, config_dict: Optional[dict] = None):
        self.config = config
        self.tokenizer = tokenizer
        self._config_dict = config_dict or {}

        self.generation_mixin: Optional[GenerationMixin] = None
        self.model = None
        self.lora_manager = None
        self.optimizer = None

        self._vram_threshold_gb = 2.0
        self._lora_staging_path = "/dev/shm/forge_loop_lora_live"

    # ==========================================================================
    # Phase 0: Init — Load training model and optimizer to CPU RAM once
    # ==========================================================================
    def init_training_model_cpu(self, model_path: str):
        """Load model into CPU RAM. This master copy shifts between RAM and VRAM."""
        print("[SwapManager] Loading training model to CPU RAM (once)...")

        from transformers import AutoModelForCausalLM
        from ..custom_lora import CustomLoRAManager
        from ..custom_optimizer import CustomOptimizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[SwapManager] Model on CPU: {total:,} params (~{total * 2 / 1e9:.1f}GB bfloat16)")

        print("[SwapManager] Injecting LoRA...")
        self.lora_manager = CustomLoRAManager(self.model, self.config)
        self.lora_manager.inject()

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.lora_manager.get_trainable_parameters():
            p.requires_grad = True

        self.optimizer = CustomOptimizer(
            self.lora_manager.get_trainable_parameters(),
            lr=self.config.learning_rate
        )

        n_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"[SwapManager] Trainable params: {n_trainable}")
        return self.model

    # ==========================================================================
    # SWAP 1: vLLM Generation Phase (in-process via GenerationMixin)
    # ==========================================================================
    def load_vllm(self, lora_path: Optional[str] = None):
        """Load vLLM to GPU via GenerationMixin."""
        if self.model is not None and next(self.model.parameters()).device.type == "cuda":
            raise RuntimeError("Training model still on GPU! Call unload_training() first.")

        self._verify_vram_freed()
        print("[SwapManager] === SWAP 1: vLLM GenerationMixin → GPU ===")
        start = time.time()

        self.generation_mixin = GenerationMixin(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        if lora_path and os.path.exists(lora_path):
            self.generation_mixin.generate_group(
                ".",
                num_generations=1,
            )

        print(f"[SwapManager] vLLM loaded in {time.time() - start:.1f}s")
        self._print_vram("vLLM loaded")

    def generate_all_mini_batches(self, macro_batch: List[Dict], lora_path: Optional[str] = None) -> List[Dict]:
        """Generate all mini-batches. Store completions in CPU RAM."""
        if self.generation_mixin is None:
            raise RuntimeError("vLLM not loaded!")

        per_step = self.config.per_device_train_batch_size
        accum = self.config.gradient_accumulation_steps
        results = []

        for step in range(accum):
            start = step * per_step
            end = start + per_step
            mini = macro_batch[start:end]
            if not mini:
                continue

            prompts = [item.get("prompt", item.get("text", "")) for item in mini]
            metas = [item.get("metadata", {}) for item in mini]

            outputs = self.generation_mixin.generate_groups(
                prompts,
                num_generations=self.config.num_generations,
            )

            grouped = []
            for texts in outputs:
                while len(texts) < self.config.num_generations:
                    texts.append(texts[-1] if texts else "")
                grouped.append(texts[:self.config.num_generations])

            results.append({
                'prompts': prompts,
                'completions': grouped,
                'metainfos': metas,
            })
            print(f"  [Gen] Mini-batch {step+1}/{accum}: {len(prompts)}x{self.config.num_generations} completions")

        return results

    def unload_vllm(self):
        """Unload vLLM engine and free GPU memory."""
        if self.generation_mixin is not None:
            print("[SwapManager] Unloading vLLM GenerationMixin...")
            if hasattr(self.generation_mixin, 'vllm_engine'):
                engine = self.generation_mixin.vllm_engine
                if hasattr(engine, 'close'):
                    try:
                        engine.close()
                    except Exception:
                        pass
                elif hasattr(getattr(engine, 'llm_engine', None), 'shutdown_background_workers'):
                    try:
                        engine.llm_engine.shutdown_background_workers()
                    except Exception:
                        pass
                        
                del self.generation_mixin.vllm_engine
            del self.generation_mixin
            self.generation_mixin = None

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[SwapManager] vLLM fully unloaded.")

    # ==========================================================================
    # SWAP 2: Training Phase (Pure HF Transformers)
    # ==========================================================================
    def load_training_to_gpu(self):
        """Move training model and optimizer CPU → GPU."""
        if self.generation_mixin is not None:
            raise RuntimeError("vLLM still on GPU!")

        self._verify_vram_freed()
        print("[SwapManager] === SWAP 2: Training CPU → GPU ===")
        start = time.time()

        # Move the ENTIRE model (parameters + buffers + submodules)
        # instead of manually iterating named_parameters only.
        self.model = self.model.to("cuda")
        torch.cuda.synchronize()

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:
            def make_requires_grad_hook():
                def _force_grad_hook(module, input, output):
                    if isinstance(output, tuple):
                        output[0].requires_grad_(True)
                        return output
                    return output.requires_grad_(True)
                return _force_grad_hook

            if hasattr(self.model, "get_input_embeddings"):
                self.model.get_input_embeddings().register_forward_hook(make_requires_grad_hook())

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self._move_optimizer_to_device("cuda")

        print(f"[SwapManager] Training model on GPU in {time.time() - start:.1f}s")
        self._print_vram("Training loaded")

    def train_on_stored(self, stored_batches: List[Dict], global_step: int,
                        reward_engine, compute_mixin) -> Tuple[float, int]:
        """Train on pre-generated batches using micro-batching to save VRAM."""
        if next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("Training model not on GPU!")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        sub_batch_size = 1
        num_gens = self.config.num_generations

        for accum_step, batch_data in enumerate(stored_batches):
            all_prompts, all_completions, all_metas = [], [], []
            for p, comps, m in zip(batch_data['prompts'], batch_data['completions'], batch_data['metainfos']):
                for c in comps:
                    all_prompts.append(p)
                    all_completions.append(c)
                    all_metas.append(m)

            rewards, diagnostics = reward_engine.score_group(
                all_completions,
                all_prompts,
                all_metas
            )

            # Enforce strict GRPO group boundaries for advantage calculation
            # Reshape rewards to (num_prompts, num_generations) to calculate isolated relative advantages
            num_prompts = len(batch_data['prompts'])
            rewards_per_prompt = rewards.reshape(num_prompts, num_gens)
            
            advantages_list = []
            for r_group in rewards_per_prompt:
                advantages_list.append(reward_engine.compute_advantages(r_group))
            advantages = np.concatenate(advantages_list, axis=0)

            # --- ADD THIS TEMPORARY DEBUG PRINT HERE ---
            print(f"  [DEBUG GRPO] Raw Rewards Mean: {rewards.mean().item():.4f} | Min: {rewards.min().item():.4f} | Max: {rewards.max().item():.4f}")
            print(f"  [DEBUG GRPO] Advantages Mean: {advantages.mean().item():.4f} | Min: {advantages.min().item():.4f} | Max: {advantages.max().item():.4f}")
            # -------------------------------------------

            # Extract reference model distributions
            self.lora_manager.set_enabled(False)
            ref_logprobs_list = []
            with torch.no_grad():
                for ch_idx in range(0, len(all_prompts), sub_batch_size):
                    ch_prompts = all_prompts[ch_idx:ch_idx+sub_batch_size]
                    ch_comps = all_completions[ch_idx:ch_idx+sub_batch_size]
                    ch_ref = compute_mixin.compute_logprobs(
                        self.model, self.tokenizer,
                        ch_prompts, ch_comps, is_policy=False
                    )
                    ref_logprobs_list.append(ch_ref)
            ref_logprobs = torch.cat(ref_logprobs_list, dim=0)

            torch.cuda.empty_cache()

            # Active Policy gradient tracking
            self.lora_manager.set_enabled(True)
            accum_step_loss = 0.0
            total_items = len(all_prompts)

            for ch_idx in range(0, total_items, sub_batch_size):
                ch_prompts = all_prompts[ch_idx:ch_idx+sub_batch_size]
                ch_comps = all_completions[ch_idx:ch_idx+sub_batch_size]
                ch_ref_logprobs = ref_logprobs[ch_idx:ch_idx+sub_batch_size]
                ch_advantages = torch.tensor(
                    advantages[ch_idx:ch_idx+sub_batch_size],
                    device=ref_logprobs.device,
                    dtype=ref_logprobs.dtype
                )

                ch_policy_logprobs = compute_mixin.compute_logprobs(
                    self.model, self.tokenizer,
                    ch_prompts, ch_comps, is_policy=True
                )

                ratio = torch.exp(ch_policy_logprobs - ch_ref_logprobs)
                
                # Bounded proxy KL divergence calculation
                kl_ratio = torch.exp(ch_ref_logprobs - ch_policy_logprobs)
                kl_div = (kl_ratio - 1) - (ch_ref_logprobs - ch_policy_logprobs)

                clip_eps = getattr(self.config, "clip_eps", 0.2)
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                surr1 = ratio * ch_advantages
                surr2 = clipped * ch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                chunk_loss = policy_loss + self.config.beta * kl_div.mean()

                weight = len(ch_prompts) / total_items
                loss = (chunk_loss * weight) / self.config.gradient_accumulation_steps

                if loss.requires_grad:
                    loss.backward()
                else:
                    print(f"  [Warning] Step {global_step + 1} | Chunk loss does not require grad. Skipping backward pass.")
                
                accum_step_loss += chunk_loss.item() * weight

            total_loss += accum_step_loss
            
            # Print temporary progress mapping to current batch update tracker
            print(f"  [Accumulation Step {accum_step + 1}/{len(stored_batches)}] Accum Loss: {accum_step_loss:.4f}")

        # Execute optimization pass across full accumulated tracking state
        torch.nn.utils.clip_grad_norm_(
            self.lora_manager.get_trainable_parameters(),
            max_norm=getattr(self.config, "max_grad_norm", 1.0)
        )
        self.optimizer.step()

        # Increment global step treating the entire macro-batch update window as a single step unit
        global_step += len(stored_batches)

        return total_loss / len(stored_batches), global_step

    def unload_training(self):
        """Move training model and optimizer GPU → CPU."""
        if self.model is None or next(self.model.parameters()).device.type == "cpu":
            return

        print("[SwapManager] Unloading training model...")

        self.model.to("cpu")
        self._move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"[VRAM] After training unload: {allocated:.2f}GB allocated")
        print("[SwapManager] Training model on CPU.")

    # ==========================================================================
    # Utilities
    # ==========================================================================
    def _move_optimizer_to_device(self, device: str):
        """Moves optimizer running averages to maintain momentum across swaps."""
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _verify_vram_freed(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"[VRAM] Cleared/Verified Allocated: {allocated:.2f}GB")

    def _print_vram(self, tag: str):
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        print(f"[VRAM {tag}] Allocated: {a:.2f}GB | Reserved: {r:.2f}GB")

    def export_lora_for_vllm(self, path: str):
        if self.lora_manager is None:
            raise RuntimeError("No CPU LoRA!")
        self.lora_manager.export_peft_format(path)
        print(f"[SwapManager] LoRA exported: {path}")