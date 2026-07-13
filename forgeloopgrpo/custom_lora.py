"""Custom LoRA — zero dependencies on HF, PEFT, or any model loader except vLLM."""

import torch
import torch.nn as nn
import math
import os
import json
from typing import Dict, List, Tuple
from safetensors.torch import save_file

class CustomLoRALayer(nn.Module):
    """Single LoRA layer: W' = W + (A @ B) * (alpha / r)"""

    def __init__(self, in_features: int, out_features: int, r: int = 64, alpha: int = 128, dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.lora_enabled = True

    def compute_lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = x @ self.lora_A
        up_proj = down_proj @ self.lora_B
        return up_proj * self.scaling

    def forward(self, x, base_output):
        if not self.lora_enabled:
            return base_output

        lora_modifier = self.compute_lora_delta(x)

        if isinstance(base_output, tuple):
            main_tensor = base_output[0]
            modified_tensor = main_tensor + lora_modifier
            return (modified_tensor,) + base_output[1:]

        return base_output + lora_modifier


def _is_linear_like(module) -> bool:
    """Duck-type check: does this module behave like a linear layer or wrap one?"""
    if hasattr(module, 'base_layer'):
        return _is_linear_like(module.base_layer)

    typename = type(module).__name__
    if "Linear" in typename:
        return True

    weight = getattr(module, 'weight', None)
    if weight is not None and hasattr(weight, 'shape') and len(weight.shape) == 2:
        return True

    return False


def _get_linear_dimensions(module) -> Tuple[int, int]:
    """Extract local (in_features, out_features) from a linear-like module."""
    if hasattr(module, 'base_layer'):
        return _get_linear_dimensions(module.base_layer)

    weight = getattr(module, 'weight', None)
    if weight is not None and hasattr(weight, 'shape') and len(weight.shape) == 2:
        return weight.shape[1], weight.shape[0]

    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        return module.in_features, module.out_features

    if hasattr(module, 'input_size') and hasattr(module, 'output_size'):
        return module.input_size, module.output_size

    raise ValueError(f"Cannot infer linear dimensions from {type(module).__name__}")


def _expand_fused_targets(target_modules: List[str]) -> List[str]:
    """Map split projection names to fused projection names used by Qwen3.5, Gemma 4, and other models."""
    expanded = set(target_modules)

    has_qkv_split = any(t in expanded for t in ["q_proj", "k_proj", "v_proj"])
    if has_qkv_split:
        expanded.add("qkv_proj")

    has_gate_up_split = any(t in expanded for t in ["gate_proj", "up_proj"])
    if has_gate_up_split:
        expanded.add("gate_up_proj")

    return list(expanded)


class CustomLoRAManager:
    """Inject custom LoRA into model and manage trainable parameters."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.lora_layers: Dict[str, CustomLoRALayer] = {}
        self.trainable_params: List[nn.Parameter] = []
        self._hooks: List[Tuple[str, torch.utils.hooks.RemovableHandle]] = []

    def inject(self, target_modules: List[str] = None):
        if target_modules is None:
            target_modules = self.config.lora.target_modules

        target_modules = _expand_fused_targets(target_modules)
        print(f"[LoRA INJECT] Starting injection. Target modules (expanded): {target_modules}")

        total_named_modules = 0
        matched_target_names = 0
        successful_injections = 0

        for name, module in self.model.named_modules():
            total_named_modules += 1

            is_matched = any(target in name for target in target_modules)
            if not is_matched:
                continue

            if ".base_layer" in name or hasattr(module, "custom_lora_adapter") or "custom_lora_adapter" in name:
                continue

            matched_target_names += 1

            linear_module = None
            if hasattr(module, 'linear') and _is_linear_like(module.linear):
                linear_module = module.linear
            elif hasattr(module, 'qkv_proj') and _is_linear_like(module.qkv_proj):
                linear_module = module.qkv_proj
            elif hasattr(module, 'gate_up_proj') and _is_linear_like(module.gate_up_proj):
                linear_module = module.gate_up_proj
            elif hasattr(module, 'o_proj') and _is_linear_like(module.o_proj):
                linear_module = module.o_proj
            elif hasattr(module, 'down_proj') and _is_linear_like(module.down_proj):
                linear_module = module.down_proj

            if linear_module is None and _is_linear_like(module):
                linear_module = module

            if linear_module is None:
                print(f"[LoRA INJECT] Skipped '{name}': Could not find nested linear structure.")
                continue

            try:
                in_f, out_f = _get_linear_dimensions(linear_module)
            except ValueError as val_err:
                print(f"[LoRA INJECT] Skipped '{name}': Dimension parsing failed. Error: {str(val_err)}")
                continue

            lora = CustomLoRALayer(
                in_f, out_f,
                r=self.config.lora.r,
                alpha=self.config.lora.alpha,
                dropout=self.config.lora.dropout
            )

            try:
                base_device = next(linear_module.parameters()).device
                base_dtype = next(linear_module.parameters()).dtype
            except StopIteration:
                base_device = torch.device("cuda:0")
                base_dtype = torch.float32

            lora = lora.to(device=base_device, dtype=base_dtype)

            setattr(linear_module, "custom_lora_adapter", lora)

            self.lora_layers[name] = lora
            self.trainable_params.extend([lora.lora_A, lora.lora_B])

            def make_hook(lora_layer):
                def _hook(module, input, output):
                    x = input[0] if isinstance(input, tuple) else input
                    if not lora_layer.lora_enabled:
                        return output

                    delta = lora_layer.compute_lora_delta(x)

                    if isinstance(output, tuple):
                        return (output[0] + delta,) + output[1:]
                    else:
                        return output + delta
                return _hook

            handle = linear_module.register_forward_hook(make_hook(lora))
            self._hooks.append((name, handle))
            successful_injections += 1

        # Freeze base, unfreeze LoRA
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.trainable_params:
            p.requires_grad = True

        total_requires_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
        print("\n" + "="*50)
        print("[POST-INJECTION REPORT]")
        print(f"  -> Scanned Modules: {total_named_modules} | Name Matches: {matched_target_names}")
        print(f"  -> Successfully Patched Layers: {successful_injections}")
        print(f"  -> Total Active Training Parameters (requires_grad=True): {total_requires_grad}")
        print("="*50 + "\n")

    def set_enabled(self, enabled: bool):
        """Toggle LoRA on/off for all layers."""
        for name, layer in self.lora_layers.items():
            layer.lora_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"[LoRA TOGGLE] All layers {status} ({len(self.lora_layers)} layers)")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return self.trainable_params

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: {
                'lora_A': layer.lora_A.detach().cpu(),
                'lora_B': layer.lora_B.detach().cpu()
            }
            for name, layer in self.lora_layers.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, torch.Tensor]]):
        for name, params in state_dict.items():
            if name in self.lora_layers:
                self.lora_layers[name].lora_A.data = params['lora_A'].to(self.lora_layers[name].lora_A.device)
                self.lora_layers[name].lora_B.data = params['lora_B'].to(self.lora_layers[name].lora_B.device)

    def export_peft_format(self, output_dir: str, base_model_path: str = None):
        """Export to standard HuggingFace PEFT format with corrected dimension slicing.

        Supports both Qwen3.5 (standard GQA) and Gemma 4 (with num_kv_shared_layers
        and global_head_dim for alternating attention layers).
        """
        import shutil
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump({
                "r": self.config.lora.r,
                "lora_alpha": self.config.lora.alpha,
                "target_modules": self.config.lora.target_modules,
                "lora_dropout": self.config.lora.dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "peft_type": "LORA",
                "base_model_name_or_path": base_model_path or "",
            }, f, indent=2)

        peft_state = {}

        model_config = None
        for candidate in [self.model, getattr(self.model, "model", None), 
                          getattr(self.model, "pretrained_model", None), 
                          getattr(self.model, "policy", None)]:
            if candidate is not None and hasattr(candidate, "config") and candidate.config is not None:
                model_config = candidate.config
                break

        for name, layer in self.lora_layers.items():
            base_key = f"base_model.model.{name}"

            # Detach, move to CPU, and transpose to match standard weights layout: 
            # lora_A_T: [r, in_features]
            # lora_B_T: [out_features, r]
            lora_A_T = layer.lora_A.detach().cpu().T.contiguous()
            lora_B_T = layer.lora_B.detach().cpu().T.contiguous()

            if "qkv_proj" in name:
                hidden_size = layer.lora_A.shape[0]
                if model_config is not None:
                    hidden_size = getattr(model_config, "hidden_size", hidden_size)
                    num_heads = getattr(model_config, "num_attention_heads", 
                                       getattr(model_config, "num_heads", 32))
                    num_kv_heads = getattr(model_config, "num_key_value_heads", 
                                          getattr(model_config, "num_kv_heads", num_heads))
                    head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)

                    # GEMMA 4 SPECIFIC: Handle num_kv_shared_layers and global_head_dim
                    num_kv_shared_layers = getattr(model_config, "num_kv_shared_layers", 0)
                    global_head_dim = getattr(model_config, "global_head_dim", head_dim)
                else:
                    total_actual = lora_B_T.shape[0]
                    num_heads = 32
                    num_kv_heads = 32
                    head_dim = total_actual // 96 
                    num_kv_shared_layers = 0
                    global_head_dim = head_dim

                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                total_expected = q_size + kv_size + kv_size
                total_actual = lora_B_T.shape[0]

                # Alternating Global Attention Layer Mapping for Gemma 4
                if num_kv_shared_layers > 0:
                    global_q_size = num_heads * global_head_dim
                    global_kv_size = num_kv_heads * global_head_dim
                    global_total = global_q_size + global_kv_size + global_kv_size

                    if total_actual == global_total:
                        q_size = global_q_size
                        kv_size = global_kv_size
                        print(f"[LoRA Export] Detected Gemma 4 global layer: {name} "
                              f"(global_head_dim={global_head_dim})")
                    elif total_expected != total_actual:
                        gqa_ratio = q_size / kv_size if kv_size > 0 else 1.0
                        single_kv_block = total_actual // (int(gqa_ratio) + 2)
                        kv_size = single_kv_block
                        q_size = total_actual - (2 * kv_size)
                elif total_expected != total_actual:
                    gqa_ratio = q_size / kv_size if kv_size > 0 else 1.0
                    single_kv_block = total_actual // (int(gqa_ratio) + 2)
                    kv_size = single_kv_block
                    q_size = total_actual - (2 * kv_size)

                lora_B_q = lora_B_T[:q_size, :].contiguous()
                lora_B_k = lora_B_T[q_size:q_size + kv_size, :].contiguous()
                lora_B_v = lora_B_T[q_size + kv_size:, :].contiguous()

                peft_state[f"{base_key.replace('qkv_proj', 'q_proj')}.lora_A.weight"] = lora_A_T.clone()
                peft_state[f"{base_key.replace('qkv_proj', 'q_proj')}.lora_B.weight"] = lora_B_q
                peft_state[f"{base_key.replace('qkv_proj', 'k_proj')}.lora_A.weight"] = lora_A_T.clone()
                peft_state[f"{base_key.replace('qkv_proj', 'k_proj')}.lora_B.weight"] = lora_B_k
                peft_state[f"{base_key.replace('qkv_proj', 'v_proj')}.lora_A.weight"] = lora_A_T.clone()
                peft_state[f"{base_key.replace('qkv_proj', 'v_proj')}.lora_B.weight"] = lora_B_v

            elif "gate_up_proj" in name:
                total_out = lora_B_T.shape[0]
                gate_dim = total_out // 2

                lora_B_gate = lora_B_T[:gate_dim, :].contiguous()
                lora_B_up = lora_B_T[gate_dim:, :].contiguous()

                peft_state[f"{base_key.replace('gate_up_proj', 'gate_proj')}.lora_A.weight"] = lora_A_T.clone()
                peft_state[f"{base_key.replace('gate_up_proj', 'gate_proj')}.lora_B.weight"] = lora_B_gate
                peft_state[f"{base_key.replace('gate_up_proj', 'up_proj')}.lora_A.weight"] = lora_A_T.clone()
                peft_state[f"{base_key.replace('gate_up_proj', 'up_proj')}.lora_B.weight"] = lora_B_up

            else:
                peft_state[f"{base_key}.lora_A.weight"] = lora_A_T
                peft_state[f"{base_key}.lora_B.weight"] = lora_B_T

        save_file(peft_state, os.path.join(output_dir, "adapter_model.safetensors"))

        if base_model_path and os.path.isdir(base_model_path):
            tokenizer_files = [
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.json", "merges.txt", "tokenizer.model", 
                "preprocessor_config.json", "chat_template.json",
            ]
            copied = []
            for fname in tokenizer_files:
                src = os.path.join(base_model_path, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_dir, fname))
                    copied.append(fname)
            if copied:
                print(f"[ForgeLoop LoRA] Copied tokenizer files: {copied}")

        print(f"[ForgeLoop LoRA] Exported PEFT adapter to {output_dir}")