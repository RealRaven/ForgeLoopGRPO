"""Utility to merge a PEFT LoRA adapter into a base model.

Supports:
  - Qwen3.5-4B (composite multimodal → text-only causal LM)
  - google/gemma-4-E2B-it (native multimodal, keep processor config)
"""

import torch
import json
import shutil
import types
from pathlib import Path


def _copy_tokenizer_files(base_model_path: str, output_path: str):
    """Copy tokenizer / processor files from base model to output directory."""
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "image_processor_config.json",
        "audio_processor_config.json",
    ]

    base_path = Path(base_model_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in files_to_copy:
        src = base_path / fname
        if src.exists():
            shutil.copy2(src, out_path / fname)
            copied.append(fname)

    if copied:
        print(f"[Merge] Copied files: {copied}")


def _convert_dict_subconfigs(config):
    """AutoConfig loads nested sub-configs as plain dicts.
    transformers' dtype setter crashes on dicts, so wrap them."""
    if not hasattr(config, "sub_configs"):
        return
    keys = config.sub_configs
    if isinstance(keys, dict):
        keys = list(keys.keys())
    for key in keys:
        sub = getattr(config, key, None)
        if isinstance(sub, dict):
            ns = types.SimpleNamespace(**sub)
            setattr(config, key, ns)
            if hasattr(config, "sub_configs") and isinstance(config.sub_configs, dict):
                config.sub_configs[key] = ns


# =============================================================================
# QWEN 3.5 PATH
# =============================================================================

def _fix_qwen35_config(config_path: str):
    """Flatten Qwen3.5 text_config into top-level and drop vision/audio."""
    config_file = Path(config_path) / "config.json"
    if not config_file.exists():
        return None

    with open(config_file, "r") as f:
        cfg = json.load(f)

    if "text_config" not in cfg:
        return None

    text_cfg = cfg["text_config"]
    merged = dict(cfg)

    excluded = {"model_type"}
    for key, value in text_cfg.items():
        if key not in excluded:
            merged[key] = value

    merged.pop("text_config", None)
    merged.pop("vision_config", None)
    merged.pop("audio_config", None)

    if "architectures" in merged:
        archs = merged["architectures"]
        if isinstance(archs, list):
            clean = []
            for a in archs:
                if "ConditionalGeneration" in a:
                    clean.append("Qwen2ForCausalLM")
                else:
                    clean.append(a)
            merged["architectures"] = list(dict.fromkeys(clean))

    print(f"[Merge/Qwen] Fixed config: model_type={merged.get('model_type')}, "
          f"vocab_size={merged.get('vocab_size')}, hidden_size={merged.get('hidden_size')}, "
          f"num_hidden_layers={merged.get('num_hidden_layers')}")
    return merged


def _load_qwen35_checkpoint_strip_prefix(checkpoint_path: str, prefix: str = "model.language_model."):
    """Load Qwen3.5 composite ckpt, strip vision keys and the language_model prefix."""
    ckpt_path = Path(checkpoint_path)

    safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
    if safetensor_files:
        import safetensors.torch
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(safetensors.torch.load_file(str(f), device="cpu"))
    else:
        bin_files = sorted(ckpt_path.glob("pytorch_model*.bin"))
        if bin_files:
            state_dict = {}
            for f in bin_files:
                state_dict.update(torch.load(str(f), map_location="cpu"))
        else:
            raise ValueError(f"No checkpoint files found in {checkpoint_path}")

    new_state = {}
    vision_prefix = "model.visual."

    for key, value in state_dict.items():
        if key.startswith(vision_prefix):
            continue
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            if not new_key.startswith("model.") and not new_key.startswith("lm_head."):
                new_key = "model." + new_key
            new_state[new_key] = value
        else:
            new_state[key] = value

    print(f"[Merge/Qwen] Stripped '{prefix}' and vision keys: {len(state_dict)} -> {len(new_state)} keys")
    return new_state


def _merge_qwen35(base_model_path: str, adapter_path: str, output_path: str, dtype: torch.dtype):
    """Qwen3.5-specific merge: flatten to text-only causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import PeftModel
    from huggingface_hub import snapshot_download

    local_base = base_model_path
    if not Path(base_model_path).exists():
        print(f"[Merge/Qwen] Downloading from HF Hub: {base_model_path}")
        local_base = snapshot_download(repo_id=base_model_path)
        print(f"[Merge/Qwen] Local cache: {local_base}")

    # 1. Fix config to text-only
    config = AutoConfig.from_pretrained(local_base, trust_remote_code=True)
    _convert_dict_subconfigs(config)

    fixed_dict = _fix_qwen35_config(local_base)
    if fixed_dict is not None:
        for key, value in fixed_dict.items():
            if key != "model_type":
                setattr(config, key, value)
        _convert_dict_subconfigs(config)

    if hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
        print(f"[Merge/Qwen] Using text backbone config: {text_config.__class__.__name__}")
    else:
        text_config = config
        print("[Merge/Qwen] Warning: get_text_config() not found, using full config")

    for drop in ("vision_config", "audio_config", "image_token_id", "video_token_id"):
        if hasattr(text_config, drop):
            setattr(text_config, drop, None)
        if isinstance(text_config, dict) and drop in text_config:
            text_config.pop(drop, None)

    # 2. Detect composite checkpoint
    has_composite = False
    ckpt_path = Path(local_base)
    index_file = ckpt_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})
            has_composite = any(k.startswith("model.language_model.") for k in weight_map)
    else:
        st_files = list(ckpt_path.glob("*.safetensors"))
        if st_files:
            import safetensors
            for sf in st_files:
                with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                    if any(k.startswith("model.language_model.") for k in f.keys()):
                        has_composite = True
                        break
        else:
            pt_files = list(ckpt_path.glob("pytorch_model*.bin"))
            for pt in pt_files:
                sd = torch.load(pt, map_location="cpu")
                if any(k.startswith("model.language_model.") for k in sd):
                    has_composite = True
                    break

    if has_composite:
        print("[Merge/Qwen] Detected composite checkpoint. Loading text weights only...")
        base_model = AutoModelForCausalLM.from_config(
            text_config,
            dtype=dtype,
            trust_remote_code=True,
        )
        state_dict = _load_qwen35_checkpoint_strip_prefix(local_base, "model.language_model.")
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
            print("[Merge/Qwen] Tied lm_head.weight to embed_tokens.weight")

        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Merge/Qwen] Missing keys: {len(missing)} (first 5 shown)")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"[Merge/Qwen] Unexpected keys: {len(unexpected)} (first 5 shown)")
            for k in unexpected[:5]:
                print(f"  - {k}")

        if torch.cuda.is_available():
            base_model = base_model.to("cuda")
    else:
        print("[Merge/Qwen] Standard checkpoint. Loading directly...")
        base_model = AutoModelForCausalLM.from_pretrained(
            local_base,
            config=text_config,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(local_base, trust_remote_code=True)

    # 3. Merge adapter
    print(f"[Merge/Qwen] Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("[Merge/Qwen] Merging adapter...")
    model = model.merge_and_unload()
    model.config = text_config

    # 4. Save
    print(f"[Merge/Qwen] Saving to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    _copy_tokenizer_files(local_base, output_path)
    print("[Merge/Qwen] Done. Text-only Unsloth-compatible model saved.")
    return output_path


# =============================================================================
# GEMMA 4 PATH
# =============================================================================

def _merge_gemma4(base_model_path: str, adapter_path: str, output_path: str, dtype: torch.dtype):
    """Gemma 4 merge: manually apply LoRA weights since PEFT doesn't support Gemma4ClippableLinear."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
    )
    from safetensors.torch import load_file, save_file
    from huggingface_hub import snapshot_download

    local_base = base_model_path
    if not Path(base_model_path).exists():
        print(f"[Merge/Gemma] Downloading from HF Hub: {base_model_path}")
        local_base = snapshot_download(repo_id=base_model_path)
        print(f"[Merge/Gemma] Local cache: {local_base}")

    # 1. Load base model
    print("[Merge/Gemma] Loading Gemma 4 base model...")
    model = AutoModelForCausalLM.from_pretrained(
        local_base,
        torch_dtype=dtype,
        device_map="cpu",  # Load to CPU for merge, save VRAM
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 2. Load adapter weights manually
    print(f"[Merge/Gemma] Loading adapter from {adapter_path}...")
    
    adapter_weights = {}
    adapter_safetensors = Path(adapter_path) / "adapter_model.safetensors"
    adapter_bin = Path(adapter_path) / "adapter_model.bin"
    
    if adapter_safetensors.exists():
        adapter_weights = load_file(str(adapter_safetensors), device="cpu")
    elif adapter_bin.exists():
        adapter_weights = torch.load(str(adapter_bin), map_location="cpu")
    else:
        raise ValueError(f"No adapter weights found in {adapter_path}")

    # 3. Parse LoRA weights and merge into base state dict
    # LoRA format: base_key.lora_A.weight, base_key.lora_B.weight
    # Merged weight = base_weight + (lora_B @ lora_A) * (alpha / r)
    
    merged_count = 0
    skipped_keys = []
    
    # Try to read alpha/r from adapter_config.json
    config_path = Path(adapter_path) / "adapter_config.json"
    alpha, rank = 128, 64  # defaults
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        alpha = cfg.get("lora_alpha", alpha)
        rank = cfg.get("r", rank)
    
    scale = alpha / rank

    state_dict = model.state_dict()
    lora_keys = list(adapter_weights.keys())
    
    # Group A and B pairs
    lora_groups = {}
    for key in lora_keys:
        if ".lora_A.weight" in key:
            base_key = key.replace(".lora_A.weight", "")
            lora_groups.setdefault(base_key, {})["A"] = adapter_weights[key]
        elif ".lora_B.weight" in key:
            base_key = key.replace(".lora_B.weight", "")
            lora_groups.setdefault(base_key, {})["B"] = adapter_weights[key]

    for base_key, parts in lora_groups.items():
        if "A" not in parts or "B" not in parts:
            skipped_keys.append(base_key)
            continue

        lora_A = parts["A"]  # (r, in_features)
        lora_B = parts["B"]  # (out_features, r)
        
        # Compute delta: lora_B @ lora_A
        delta = (lora_B @ lora_A) * scale
        
        # Find matching base weight key
        # Adapter keys may use different naming than state_dict
        # Try exact match, then common prefixes
        weight_key = base_key + ".weight"
        
        # Handle naming mismatches: e.g. adapter uses "model.layers.0.self_attn.q_proj"
        # but state_dict uses "model.model.layers.0.self_attn.q_proj.linear.weight" for Gemma4ClippableLinear
        if weight_key not in state_dict:
            # Try with "linear" suffix for Gemma4ClippableLinear
            alt_key = base_key + ".linear.weight"
            if alt_key in state_dict:
                weight_key = alt_key
            else:
                # Try adding "model." prefix if missing
                alt_key2 = "model." + weight_key
                if alt_key2 in state_dict:
                    weight_key = alt_key2
                else:
                    alt_key3 = "model." + base_key + ".linear.weight"
                    if alt_key3 in state_dict:
                        weight_key = alt_key3

        if weight_key not in state_dict:
            skipped_keys.append(base_key)
            print(f"[Merge/Gemma] Warning: Could not find base weight for {base_key}, tried {weight_key}")
            continue

        # Apply merge
        state_dict[weight_key] = state_dict[weight_key].to(delta.dtype) + delta.to(state_dict[weight_key].device)
        merged_count += 1

    # Load merged weights back
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Merge/Gemma] Missing keys after merge: {len(missing)}")
    if unexpected:
        print(f"[Merge/Gemma] Unexpected keys: {len(unexpected)}")

    print(f"[Merge/Gemma] Merged {merged_count} LoRA adapters. Skipped: {len(skipped_keys)}")
    if skipped_keys:
        for k in skipped_keys[:5]:
            print(f"  - {k}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")

    # 4. Save
    tokenizer = AutoTokenizer.from_pretrained(local_base, trust_remote_code=True)
    
    print(f"[Merge/Gemma] Saving to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    _copy_tokenizer_files(local_base, output_path)

    try:
        processor = AutoProcessor.from_pretrained(local_base, trust_remote_code=True)
        processor.save_pretrained(output_path)
        print("[Merge/Gemma] Processor config saved.")
    except Exception as e:
        print(f"[Merge/Gemma] Warning: could not save processor config: {e}")

    print("[Merge/Gemma] Done. Gemma 4 merged model saved.")
    return output_path

# =============================================================================
# ENTRY POINT
# =============================================================================

def merge_peft_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """
    Merge a PEFT LoRA adapter into a base model.

    Auto-detects Qwen3.5 vs Gemma 4 and runs the appropriate path.
    """
    try:
        from transformers import AutoConfig
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Requires transformers, peft, huggingface_hub. "
            "Install: pip install transformers peft huggingface_hub"
        ) from e

    local_base = base_model_path
    if not Path(base_model_path).exists():
        local_base = snapshot_download(repo_id=base_model_path)

    config = AutoConfig.from_pretrained(local_base, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    archs = getattr(config, "architectures", [])
    arch_name = archs[0] if archs else ""

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"[Merge] Detected model_type={model_type}, architecture={arch_name}")

    # Route to correct handler
    if "qwen3" in model_type or "qwen2" in model_type.lower():
        print("[Merge] Routing to Qwen3.5 merge path.")
        return _merge_qwen35(local_base, adapter_path, output_path, dtype)
    elif "gemma4" in model_type or "gemma" in model_type:
        print("[Merge] Routing to Gemma 4 merge path.")
        return _merge_gemma4(local_base, adapter_path, output_path, dtype)
    else:
        print(f"[Merge] Unknown model type '{model_type}'. Falling back to generic merge.")
        return _merge_gemma4(local_base, adapter_path, output_path, dtype)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapter into base model")
    parser.add_argument("--base_model", required=True, help="Base model ID or local path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    merge_peft_adapter(args.base_model, args.adapter, args.output)