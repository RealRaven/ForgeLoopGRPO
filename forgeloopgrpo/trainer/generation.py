"""Generation logic — vLLM only. HF fallback removed."""

import os
import torch
import hashlib
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from vllm.lora.request import LoRARequest
from ..utils.model_utils import init_vllm_and_extract_model


class GenerationMixin:
    """Handles group completion generation with vLLM only."""

    _LORA_INT_ID = 1
    _LORA_NAME = "forge_loop_live_adapter"

    STANDARD_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Extended projection names for model-family detection
    GDN_PROJECTION_NAMES = [
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
        "x_proj", "dt_proj", "out_proj", "conv1d", "linear_attn",
        "embed_tokens_per_layer", "ple_input", "ple_output",
    ]

    # Modules that vLLM cannot load LoRA weights for (multimodal towers, etc.)
    NON_LORA_MODULE_PREFIXES = [
        "audio_tower", "vision_tower", "vision_model", "image_encoder",
        "speech_encoder", "multimodal_projector", "mm_projector",
    ]

    def __init__(self, model, tokenizer, config, vllm_engine=None, sampling_params_cls=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.vllm_engine = vllm_engine
        self.sampling_params_cls = sampling_params_cls
        self.global_step = 0
        self._lora_loaded_ok = False
        self._last_lora_error = None

        if self.vllm_engine is None:
            result = init_vllm_and_extract_model(config.model_path, config)
            if result[0] is not None:
                self.vllm_engine, self.sampling_params_cls, _ = result
                print("[ForgeLoop] vLLM initialized for fast generation")
            else:
                raise RuntimeError("vLLM enabled but failed to initialize.")
        else:
            print("[ForgeLoop] Using provided vLLM engine")

    def generate_group(self, prompt: str, num_generations: int = 8, lora_path: Optional[str] = None) -> List[str]:
        """Generate G completions for a single prompt."""
        return self._generate_vllm_batched([prompt], num_generations, lora_path=lora_path)[0]

    def generate_groups(self, prompts: List[str], num_generations: int, lora_path: Optional[str] = None) -> List[List[str]]:
        """Batch multiple prompts through vLLM."""
        if not prompts:
            return []
        return self._generate_vllm_batched(prompts, num_generations, lora_path=lora_path)

    def _evict_lora(self, lora_request: LoRARequest):
        """Explicitly remove a LoRA adapter from vLLM's internal cache."""
        try:
            engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
            if hasattr(engine, "remove_lora"):
                engine.remove_lora(lora_request.lora_int_id)
                return True
            if hasattr(engine, "lora_manager") and engine.lora_manager is not None:
                engine.lora_manager.remove_adapter(lora_request.lora_int_id)
                return True
            core = getattr(engine, "engine_core", None)
            if core and hasattr(core, "remove_lora"):
                core.remove_lora(lora_request.lora_int_id)
                return True
        except Exception as exc:
            print(f"[GENERATION] LoRA evict warning (may be harmless): {exc}")
        return False

    def _get_vllm_model_dims(self) -> Optional[Dict[str, int]]:
        """Extract expected projection dimensions from the vLLM-loaded model config."""
        try:
            engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
            model_config = getattr(engine, "model_config", None)
            if model_config is None:
                model_config = getattr(getattr(engine, "engine_core", None), "model_config", None)
            if model_config is None:
                return None

            hf_config = getattr(model_config, "hf_config", model_config)
            hidden_size = getattr(hf_config, "hidden_size", None)
            intermediate_size = getattr(hf_config, "intermediate_size", None)
            num_heads = getattr(hf_config, "num_attention_heads", None)
            num_kv_heads = getattr(hf_config, "num_key_value_heads", None)

            if hidden_size is None or num_heads is None:
                return None

            head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)
            num_kv_heads = num_kv_heads or num_heads
            kv_dim = num_kv_heads * head_dim

            global_head_dim = getattr(hf_config, "global_head_dim", head_dim)
            num_kv_shared_layers = getattr(hf_config, "num_kv_shared_layers", 0)

            if num_kv_shared_layers > 0:
                print(f"[GENERATION] Gemma 4 detected: head_dim={head_dim}, "
                      f"global_head_dim={global_head_dim}, "
                      f"num_kv_shared_layers={num_kv_shared_layers}")

            return {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size or hidden_size * 4,
                "q_dim": hidden_size,
                "k_dim": kv_dim,
                "v_dim": kv_dim,
                "o_dim": hidden_size,
                "head_dim": head_dim,
                "global_head_dim": global_head_dim,
                "num_kv_shared_layers": num_kv_shared_layers,
            }
        except Exception as e:
            print(f"[GENERATION WARNING] Could not extract model dims from vLLM: {e}")
            return None

    def _filter_multimodal_weights(self, state_dict: Dict[str, torch.Tensor], lora_path: str) -> Dict[str, torch.Tensor]:
        """Strip audio/vision tower weights that vLLM cannot load as LoRA."""
        filtered = {}
        removed_keys = []
        for key, tensor in state_dict.items():
            if any(prefix in key for prefix in self.NON_LORA_MODULE_PREFIXES):
                removed_keys.append(key)
                continue
            filtered[key] = tensor

        if removed_keys:
            print(f"[GENERATION] Filtered {len(removed_keys)} multimodal weights from LoRA adapter:")
            for k in removed_keys[:5]:
                print(f"    - {k}")
            if len(removed_keys) > 5:
                print(f"    ... and {len(removed_keys) - 5} more")

            # Write filtered weights back to disk so vLLM loads the clean version
            weights_path_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
            weights_path_bin = os.path.join(lora_path, "adapter_model.bin")

            try:
                if os.path.isfile(weights_path_safetensors):
                    from safetensors.torch import save_file
                    save_file(filtered, weights_path_safetensors)
                elif os.path.isfile(weights_path_bin):
                    torch.save(filtered, weights_path_bin)
                else:
                    # Fallback: write safetensors if neither exists (shouldn't happen)
                    from safetensors.torch import save_file
                    save_file(filtered, os.path.join(lora_path, "adapter_model.safetensors"))
            except Exception as e:
                raise RuntimeError(f"Failed to write filtered LoRA weights: {e}")

        return filtered

    def _validate_lora_compatibility(self, lora_path: str) -> Tuple[bool, str]:
        """Deep validation of LoRA adapter compatibility supporting alternating layouts."""
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            return False, f"LoRA adapter config not found at {config_path}"

        with open(config_path, "r") as f:
            adapter_config = json.load(f)

        lora_r = adapter_config.get("r", adapter_config.get("lora_alpha", 16))
        lora_target_modules = adapter_config.get("target_modules", [])
        base_model_name = adapter_config.get("base_model_name_or_path", "unknown")

        print(f"[GENERATION] LoRA config: rank={lora_r}, target_modules={len(lora_target_modules)}, base_model={base_model_name}")

        if isinstance(lora_target_modules, str):
            lora_target_modules = [lora_target_modules]

        normalized_targets = []
        for t in lora_target_modules:
            normalized_targets.append(t.split(".")[-1] if "." in t else t)

        gdn_targets_found = [t for t in normalized_targets if any(gdn in t for gdn in self.GDN_PROJECTION_NAMES)]
        if gdn_targets_found:
            return False, (
                f"Adapter contains non-standard projection targets: {gdn_targets_found}. "
                f"These may be Gemma 4 PLE embeddings or unsupported projections. "
                f"Regenerate with target_modules = {self.STANDARD_TARGET_MODULES}"
            )

        weights_path_bin = os.path.join(lora_path, "adapter_model.bin")
        weights_path_safetensors = os.path.join(lora_path, "adapter_model.safetensors")

        weight_files = []
        if os.path.isfile(weights_path_safetensors):
            weight_files.append(("safetensors", weights_path_safetensors))
        if os.path.isfile(weights_path_bin):
            weight_files.append(("bin", weights_path_bin))

        if not weight_files:
            return False, f"No adapter weights found in {lora_path}"

        state_dict = None
        for fmt, wpath in weight_files:
            try:
                if fmt == "safetensors":
                    from safetensors.torch import load_file
                    state_dict = load_file(wpath)
                else:
                    state_dict = torch.load(wpath, map_location="cpu", weights_only=True)
                break
            except Exception as e:
                return False, f"Failed to load adapter weights from {wpath}: {e}"

        if state_dict is None:
            return False, "Failed to load any adapter weight file"

        # === Strip multimodal tower weights before vLLM sees them ===
        state_dict = self._filter_multimodal_weights(state_dict, lora_path)
        # =================================================================

        expected_dims = self._get_vllm_model_dims()
        if expected_dims is not None:
            dim_map = {
                "q_proj": expected_dims["q_dim"],
                "k_proj": expected_dims["k_dim"],
                "v_proj": expected_dims["v_dim"],
                "o_proj": expected_dims["o_dim"],
                "gate_proj": expected_dims["intermediate_size"],
                "up_proj": expected_dims["intermediate_size"],
                "down_proj": expected_dims["hidden_size"],
            }

            mismatches = []
            for key, tensor in state_dict.items():
                if "lora_B" not in key and "lora_b" not in key:
                    continue
                parts = key.split(".")
                target_name = None
                for part in parts:
                    if part in dim_map:
                        target_name = part
                        break
                if target_name is None:
                    continue

                actual_d_out = tensor.shape[0]
                expected_d_out = dim_map[target_name]

                # GEMMA 4 LENIENCY: For q/k/v_proj, also accept global_head_dim variant
                if target_name in ("q_proj", "k_proj", "v_proj") and expected_dims.get("num_kv_shared_layers", 0) > 0:
                    global_expected = expected_dims["global_head_dim"] * (
                        expected_dims["q_dim"] // expected_dims["head_dim"] if target_name == "q_proj" 
                        else expected_dims["k_dim"] // expected_dims["head_dim"]
                    )
                    if actual_d_out == global_expected:
                        continue 

                if actual_d_out != expected_d_out:
                    mismatches.append({
                        "key": key,
                        "target": target_name,
                        "actual": actual_d_out,
                        "expected": expected_d_out,
                    })

            if mismatches:
                diag_lines = [f"    {m['key']}: actual={m['actual']}, expected={m['expected']} (for {m['target']})" for m in mismatches[:5]]
                if len(mismatches) > 5:
                    diag_lines.append(f"    ... and {len(mismatches) - 5} more mismatches")

                return False, (
                    f"\n🚨 ADAPTER DIMENSION MISMATCH 🚨\n"
                    f"Mismatched weights:\n{'.'.join(diag_lines)}\n\n"
                    f"vLLM expects: hidden_size={expected_dims['hidden_size']}, "
                    f"intermediate_size={expected_dims['intermediate_size']}\n"
                )

        return True, ""

    def _generate_vllm_batched(self, prompts: List[str], num_generations: int, lora_path: Optional[str] = None) -> List[List[str]]:
        """True batch generation utilizing vLLM's native shared prefix KV-caching."""
        sampling_params = self.sampling_params_cls(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=num_generations
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Fall back gracefully if an explicit path isn't provided by orchestration
        if lora_path is None:
            lora_path = getattr(self.config, "lora_staging_path", "/dev/shm/forge_loop_lora_live")

        if not os.path.isdir(lora_path):
            raise RuntimeError(f"LoRA adapter path does not exist in environment scope: {lora_path}")

        is_valid, err_msg = self._validate_lora_compatibility(lora_path)
        if not is_valid:
            self._lora_loaded_ok = False
            self._last_lora_error = err_msg
            raise RuntimeError(f"LoRA validation critical failure during policy pass: {err_msg}")

        lora_request = LoRARequest(
            lora_name=self._LORA_NAME,
            lora_int_id=self._LORA_INT_ID,
            lora_path=lora_path
        )

        self._evict_lora(lora_request)

        try:
            outputs = self.vllm_engine.generate(
                prompts,
                sampling_params,
                lora_request=lora_request
            )
            self._lora_loaded_ok = True
            self._last_lora_error = None
        except RuntimeError as e:
            self._lora_loaded_ok = False
            self._last_lora_error = str(e)
            raise RuntimeError(f"vLLM engine execution failed with adapter configurations loaded: {e}")

        if outputs is None:
            raise RuntimeError("vLLM returned None outputs.")

        grouped = []
        for output in outputs:
            group_texts = [completion.text for completion in output.outputs]
            if len(group_texts) > num_generations:
                group_texts = group_texts[:num_generations]
            while len(group_texts) < num_generations:
                group_texts.append(group_texts[-1] if group_texts else "")
            grouped.append(group_texts)

        first_text = grouped[0][0] if grouped and grouped[0] else ""
        text_hash = hashlib.md5(first_text.encode()).hexdigest()[:8]
        lora_flag = "lora=OK" if self._lora_loaded_ok else "lora=BASE"
        print(f"[GENERATION HASH] step={self.global_step} {lora_flag} hash={text_hash}")

        self.global_step += 1
        return grouped