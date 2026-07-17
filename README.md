# ForgeLoopGRPO

**GRPO training that runs until you stop it (Strictly Linux Only).**

A practical, stable, single-GPU GRPO trainer built on vLLM 0.25.1 + a fully custom PyTorch stack. Train 3B–7B models (Qwen3.5-4B, Gemma-4-E2B-IT, and more) with reinforcement learning on a single RTX 4090 (or similar 24GB card) in **bf16**, without OOM headaches.

Most GRPO implementations assume datacenter hardware. This one was built for real people with one gaming GPU, a clean Linux setup, and a specific problem they want to solve.

---

## Why This Exists

I built **ForgeLoopGRPO** alone, in 8 days.

Before this project, my experience was limited to training with **Unsloth at 192 max_new_tokens**. I had never worked with vLLM, GRPO, or long-context reinforcement learning.

It took me **8 intense days** of debugging and engineering to reach a stable pipeline capable of running **2048-token generations** with a rich 15-component reward system on a single RTX 4090 (staying under ~18GB VRAM).

Most developers who try this on consumer hardware eventually hit a wall. The existing tooling assumes massive clusters, doesn’t handle memory fragmentation well, and treats vLLM’s LoRA behavior as a black box.

This project is the result of pushing through those barriers.

---

## Verified Endurance Run

| Metric              | Value                  |
|---------------------|------------------------|
| **Operating System**| Linux Native Only (Ubuntu/Debian/RHEL) |
| Hardware            | RTX 4090 24GB + AMD Ryzen 9 7950X3D |
| Environment         | CUDA 13 + vLLM 0.25.1 |
| Precision           | **bf16** (not 4-bit)   |
| Generations         | 8 × 2048 tokens        |
| Total steps         | **1000**               |
| Wall time           | ~16.5 hours            |
| Avg step time       | ~59 seconds            |
| VRAM (generation)   | ~12GB                  |
| VRAM (training)     | ~9GB                   |
| Post-swap cleanup   | <0.1GB                 |
| Swap cycles         | 125+                   |
| OOM crashes         | **0**                  |
| Restarts required   | **0**                  |

The training loop runs continuously until you stop it.

---

## What Makes It Different

| Feature                    | What it means for you |
|---------------------------|-----------------------|
| **Linux-Native Optimization** | Built from the ground up for Linux shared memory and low-level control. |
| **Shared Memory LoRA Swap** | Uses `/dev/shm/forge_loop_lora_live` for extremely fast model handoffs between vLLM and training. |
| **Custom LoRA Engine**     | Zero dependency on HF PEFT during training. Hooks are injected directly; PEFT export only at the end. Supports Qwen3.5 and Gemma 4 architectures. |
| **Custom Optimizer**       | Pure `torch.optim.AdamW`. Full control, no hidden trainer behavior. |
| **CPU/GPU SwapManager**    | vLLM generation and training never share VRAM. Reliable sub-second swaps with zero memory leaks. |
| **15-Component Reward System** | Positive-sum, runtime-normalized, with EMA normalization and adaptive gating. |
| **Async CPU Scoring**      | Embedding-based diversity and other heavy rewards run on CPU while GPU generates. |
| **Config-Driven**          | Everything is controlled from one JSON file. No code changes needed for most experiments. |

---

## Features

- **vLLM + SwapManager** — unload, train, reload. Zero memory leaks.
- **Full bf16 training** — no quantization compromises.
- **Rich, pluggable rewards** — easy to add your own scorers.
- **Soft Cull** — gently penalizes only the weakest ~3% of generations.
- **Degeneracy Auditor** — protects against looping, explosion, and missing reasoning tags.
- **Free-form generation** — no forced XML or think tags.
- **One-command training, resume, and export**.

---

## What Ships By Default

### Default Rewards (15 components)

Weights are automatically normalized to sum to 1.0 at runtime.

*(Table remains the same as you had — I kept it unchanged for now)*

### Hard Penalties (Degeneracy Auditor)

*(Same as before — clean and clear)*

---

## Installation

> **Important:** ForgeLoopGRPO runs **only on Linux**. Windows and WSL are not supported due to reliance on `/dev/shm` for fast LoRA swapping.

```bash
git clone https://github.com/RealRaven/ForgeLoopGRPO.git
cd ForgeLoopGRPO

python3 -m venv .venv
source .venv/bin/activate

# Verify shared memory
df -h /dev/shm

# Core dependencies
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# Build dependencies
pip install packaging ninja wheel setuptools
MAX_JOBS=8 pip install flash-attn==2.4.2 --no-build-isolation

# Error noqa: F403 Fix
pip install --upgrade nvidia-nccl-cu12
```

> **Note:** The engine has been explicitly verified on **vLLM 0.25.1** running on **CUDA 13**. Limiting `MAX_JOBS` matches parallel processing limits on consumer layouts during native wheel compilation.

---

## Quick Start

### 1. Prepare your dataset

JSONL format. One prompt per line:

```jsonl
{"text": "Analyze the optimization characteristics of memory caching pipelines.", "metainfo": {"target_mode": "neutral"}}
{"text": "Evaluate the current network latency overhead parameters.", "metainfo": {"target_mode": "neutral"}}

```

See `data/example_dataset.jsonl`.

### 2. Edit config

```bash
cp config.example.json config.json
# Adjust model_path, dataset_path, vocabulary, reward_weights

```

### 3. Validate

```bash
python -m forgeloopgrpo validate config.json

```

### 4. Preview prompts

```bash
python -m forgeloopgrpo preview config.json

```

### 5. Train

Launch the training pipeline using explicit memory allocation constraints and thread throttling to prevent over-subscription crashes on high-core layouts:

```bash
python -m forgeloopgrpo train config.json

# OR Ryzen 9 7950X3D

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" OMP_NUM_THREADS=1 python -m forgeloopgrpo train config.json

```

### 6. Resume / override

```bash
# Resume from checkpoint using runtime optimizations
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" OMP_NUM_THREADS=1 python -m forgeloopgrpo train config.json --resume outputs/forge_run_v1/checkpoint-500

# Override hyperparameters on the fly
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" OMP_NUM_THREADS=1 python -m forgeloopgrpo train config.json --override learning_rate=8e-6 --override temperature=0.85

```

---

## CLI Reference

```bash
python -m forgeloopgrpo [command] config.json [options]

Commands:
  train      Start or resume training
  preview    Print first 5 formatted prompts (dry-run)
  validate   Validate config against Pydantic schema

Options:
  --resume DIR          Resume from checkpoint directory
  --override KEY=VALUE  Override any config field (supports nested keys via dot notation)

```

---

## Hardware & System Requirements

| Component | Requirement | Status |
| --- | --- | --- |
| **OS** | **Linux strictly required** | **Native Ubuntu/Debian Verified** |
| **Shared Memory** | `/dev/shm` with write access | **Required for `/dev/shm/forge_loop_lora_live**` |
| GPU | RTX 4090 | **Primary target. Verified.** |
| GPU | RTX 3090 | Should work. Untested. |
| GPU | RTX 4080 | May need `max_new_tokens=1024`, `max_num_seqs=32` |
| GPU | A6000 | Overkill. Will run comfortably. |
| Model | Qwen3.5-4B | **Verified** — 36 layers, 3.5B params |
| Model | Gemma-4-E2B-IT | **Compatible** — 42 layers, 4B params, alternating attention |
| Model | Gemma-4-12B-IT | ⚠️ Untested — reduce `max_new_tokens` to 1024 |
| Model | Gemma-4-26B-IT | ❌ Not recommended — too large for 24GB VRAM in bf16 |

* **CPU:** 4+ cores for async reward workers (**AMD Ryzen 9 7950X3D** explicitly verified).
* **Note on High-Core CPUs:** The runtime environment utilizes `OMP_NUM_THREADS=1` to optimize scheduling context switches between background reward workers and core PyTorch processing matrices.
* **RAM:** 32GB+ recommended for dataset loading and embedding cache.

---

## Project Structure

```
data/
└── example_dataset.jsonl



forgeloopgrpo/
├── config.json                  # Edit this file to customize your GRPO-Training
├── __init__.py
├── __main__.py                  # CLI entry point (train / preview / validate)
├── main.py                      # Training orchestration + CPU/GPU swap loop
├── config.py                    # Pydantic configuration schema (ForgeLoopGRPOConfig)
├── custom_lora.py               # Zero-dependency LoRA injection + PEFT export
├── custom_optimizer.py          # Pure torch AdamW wrapper
├── embeddings.py                # CPU embedding store with L3-cache-aware bounded cache
├── rewards/
│   ├── __init__.py 
│   ├── components.py            # ThematicScorer, FluencyScorer
│   ├── tone_scorer.py           # ToneScorer (Sentiment balancing metrics)
│   ├── engine.py                # 15-component PolicyRewardOrchestrationEngine
│   ├── gate.py                  # AdaptiveGate (zero-lag calibration)
│   ├── normalizer.py            # BoundedNormalizer (EMA tanh squash)
│   ├── semantic_diversity.py    # Gram-matrix diversity scorer
│   ├── soft_cull.py             # Percentile-based soft penalty
│   ├── behavioral_alignment_scorers.py  # 11 heuristic structural alignment functions
│   └── degeneracy_auditor.py    # Hard-penalty offense detection
├── trainer/
│   ├── __init__.py 
│   ├── generation.py            # vLLM-only generation with LoRA hot-swap
│   ├── swap_manager.py          # CPU↔GPU model swap orchestration
│   └── compute.py               # GRPO advantage + logprob computation (pure HF)
└── utils/
    ├── __init__.py  
    ├── data_utils.py            # JSONL loading, ChatML prompt building, shuffle
    ├── merge_utils.py           # PEFT adapter → standalone model merge
    └── model_utils.py           # Tokenizer loading, vLLM engine init

```

---

## Writing Custom Rewards

```python
# rewards/my_reward.py
from .base import BaseScorer

class MyScorer(BaseScorer):
    def score(self, completion: str, prompt: str, group: list[str]) -> float:
        # Return 0.0 to 3.0
        if "magic_word" in completion:
            return 2.5
        return 0.5

```

Add to `config.json`:

```json
"reward_weights": {
    "my_reward": 0.10,
    ...
}

```

The engine doesn't care what you're training. Dialogue, code, math, poetry — write the scorer, set the weight, run.

---

## Exporting & Merging Adapters

### Export to PEFT format

Done automatically every `save_steps`. The checkpoint folder contains:

* `custom_lora.pt` — raw checkpoint (for resume)
* `peft_adapter/` — standard HF PEFT adapter (for sharing / inference)

### Merge into a standalone model

```python
from forgeloopgrpo.utils.merge_utils import merge_peft_adapter

merge_peft_adapter(
    base_model_path="Qwen/Qwen3.5-4B",
    adapter_path="outputs/forge_run_v1/final/peft_adapter",
    output_path="outputs/forge_run_v1/final/merged_model"
)

```

Or enable auto-merge in config:

```json
"auto_merge_final": true

```

The merge utility handles both Qwen3.5's and Gemma 4's nested `text_config`, composite checkpoint prefixes (`model.language_model.*`), and vision-key filtering automatically. Gemma 4-specific handling includes `num_kv_shared_layers`, `global_head_dim`, and Per-Layer Embeddings (PLE).

---


---

## Supported Models

| Model | Status | Hardware Target | Notes |
|-------|--------|----------------|-------|
| **Qwen/Qwen3.5-4B** | ✅ Verified | RTX 4090 24GB | Primary target. 2048-token generations confirmed stable. |
| **google/gemma-4-E2B-it** | ✅ Compatible | RTX 4090 24GB | Tested for LoRA injection, PEFT export, and merge. |
| **google/gemma-4-12B-it** | ⚠️ Untested | Likely works | Reduce `max_new_tokens` to 1024, `max_num_seqs` to 32. |
| **google/gemma-4-26B-it** | ❌ Not recommended | A6000 48GB+ | Too large for 24GB VRAM in bf16. |

### Gemma 4 Architecture Notes

Gemma 4 E2B IT differs from Qwen3.5 in several key ways that ForgeLoopGRPO now handles automatically:

| Feature | Qwen3.5-4B | Gemma 4 E2B IT |
|---------|-----------|----------------|
| `hidden_size` | 2560 | 2560 |
| `intermediate_size` | 6912 | 10240 |
| `num_attention_heads` | 32 | 8 |
| `num_key_value_heads` | 8 (GQA) | 2 (GQA) |
| `head_dim` | 80 | 256 |
| `num_hidden_layers` | 36 | 42 |
| `num_kv_shared_layers` | 0 | 18 |
| `sliding_window` | None | 512 (alternating with global) |
| `global_head_dim` | N/A | 512 |
| `tie_word_embeddings` | Yes | Yes |
| Thinking tags | `<think>...</think>` | `<|channel>thought\n...<channel|>` |

**Key differences handled by the engine:**

1. **Per-Layer Embeddings (PLE)** — Gemma 4 feeds a second embedding table into every decoder layer. LoRA does not target embeddings, so this is transparent.

2. **Shared KV Cache** — The last 18 of 42 layers reuse KV states from earlier layers. The LoRA export logic detects `num_kv_shared_layers` and adjusts qkv_proj slicing accordingly.

3. **Alternating Attention** — Sliding-window (512 tokens, `head_dim=256`) and global full-context (`global_head_dim=512`) layers alternate. The PEFT export detects layer type by matching actual vs expected output dimensions.

4. **Dual RoPE** — Different `rope_theta` values per layer type. This is handled by the base model, not LoRA.

### Qwen3.5 Configuration Example

```json
{
  "model_path": "Qwen/Qwen3.5-4B",
  "dataset_path": "data/dataset.jsonl",
  "output_dir": "outputs/forge_run_qwen3_5",
  "num_generations": 8,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "beta": 0.04,
  "learning_rate": 5e-06,
  "temperature": 0.9,
  "top_p": 0.95,
  "max_new_tokens": 2048,
  "max_seq_length": 4096,
  "num_train_epochs": 3,
  "reward_weights": {
    "thematic_consistency": {"enabled": true, "weight": 0.15},
    "tone_consistency": {"enabled": true, "weight": 0.1},
    "semantic_diversity": {"enabled": true, "weight": 0.1},
    "fluency": {"enabled": true, "weight": 0.1},
    "reasoning_depth": {"enabled": true, "weight": 0.05},
    "lexical_diversity": {"enabled": true, "weight": 0.05},
    "efficiency_coefficient": {"enabled": true, "weight": 0.05},
    "directive_clarity": {"enabled": true, "weight": 0.05},
    "context_alignment": {"enabled": true, "weight": 0.05},
    "input_adaptation": {"enabled": true, "weight": 0.05},
    "style_preservation": {"enabled": true, "weight": 0.05},
    "creative_problem_solving": {"enabled": true, "weight": 0.03},
    "cognitive_richness": {"enabled": true, "weight": 0.03},
    "style_coherence": {"enabled": true, "weight": 0.03},
    "exploratory_boldness": {"enabled": true, "weight": 0.03}
  },
  "soft_cull": {
    "enabled": true,
    "cull_percentage": 0.03,
    "cull_penalty": -0.05
  },
  "degeneracy_auditor": {
    "severe_looping": {"enabled": true, "penalty": -0.1},
    "missing_think_tag": {"enabled": true, "penalty": -0.1},
    "linguistic_explosion": {"enabled": true, "penalty": -0.1}
  },
  "think_tag_start": "<think>",
  "think_tag_end": "</think>",
  "gemma_thinking_enabled": false,
  "gate": {
    "enabled": true,
    "warmup_steps": 0,
    "historical_mean_alpha": 0.1
  },
  "vllm": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.50,
    "max_num_seqs": 47,
    "max_model_len": 4096,
    "dtype": "auto",
    "swap_space": 2
  },
  "performance": {
    "embedding_device": "cpu",
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "gradient_checkpointing": true,
  "lora": {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05,
    "target_modules": [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj"
    ]
  },
  "save_steps": 500,
  "eval_steps": 500,
  "semantic_diversity_floor": 0.01,
  "torch_compile": false,
  "auto_merge_final": true,
  "domain_constants": {
    "vocab": [
      "analysis", "synthesis", "evaluation", "conclusion", "hypothesis",
      "evidence", "reasoning", "insight", "perspective", "framework",
      "clarity", "truth", "understanding", "discovery", "inquiry",
      "method", "principle", "theory", "concept", "abstraction"
    ],
    "anti_patterns": [
      "as an ai", "i am an ai", "language model", "llm",
      "artificial intelligence", "i cannot", "i'm sorry", "i apologize"
    ],
    "sentiment_markers": {
      "positive": ["excellent", "efficient", "optimal", "beneficial", "valuable"],
      "negative": ["error", "failure", "flaw", "inefficient", "deficient"],
      "neutral": ["objective", "standard", "consistent", "uniform", "stable"]
    },
    "action_verbs": [
      "execute", "implement", "analyze", "evaluate", "modify",
      "generate", "process", "maintain", "secure", "optimize"
    ],
    "rejection_markers": [
      "cannot fulfill", "is unable to", "not supported",
      "invalid request", "restricted", "outside the scope"
    ],
    "thematic_markers": [
      "domain", "context", "framework", "structure", "core",
      "pivotal", "fundamental", "origin", "basis", "background"
    ],
    "reasoning_markers": [
      "therefore", "because", "thus", "hence", "consequently",
      "furthermore", "additionally", "alternatively", "whereas"
    ],
    "absolute_markers": [
      "must", "will", "shall", "cannot", "never", "always", "absolutely"
    ],
    "creative_markers": [
      "imagine", "picture", "like a", "similar to", "analogy",
      "what if", "reverse", "backwards", "different way", "another angle"
    ],
    "cognitive_modalities": {
      "math": ["calculate", "equals", "plus", "minus", "sum", "ratio", "number"],
      "logical": ["premise", "syllogism", "valid", "invalid", "fallacy", "assertion"],
      "strategic": ["plan", "strategy", "tactic", "move", "position", "advantage"],
      "empirical": ["observe", "data", "metric", "evidence", "test", "experiment"]
    },
    "structural_phrases": [
      "initially", "firstly", "consequently", "subsequently",
      "furthermore", "in conclusion", "summarizing", "moving forward"
    ]
  },
  "chat_template": {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "use_system_prompt": true,
    "system_prompt_steps": 200,
    "system_prompt_text": "You are a helpful, precise, and logically sound assistant focused on clear text generation and accurate instruction following."
  }
}
```

### Gemma 4 Configuration Example

```json
{
  "model_path": "google/gemma-4-E2B-it",
  "dataset_path": "data/dataset.jsonl",
  "output_dir": "outputs/forge_run_gemma4",
  "num_generations": 8,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "beta": 0.04,
  "learning_rate": 5e-06,
  "temperature": 0.9,
  "top_p": 0.95,
  "max_new_tokens": 2048,
  "max_seq_length": 4096,
  "num_train_epochs": 3,
  "reward_weights": {
    "thematic_consistency": {"enabled": true, "weight": 0.15},
    "tone_consistency": {"enabled": true, "weight": 0.1},
    "semantic_diversity": {"enabled": true, "weight": 0.1},
    "fluency": {"enabled": true, "weight": 0.1},
    "reasoning_depth": {"enabled": true, "weight": 0.05},
    "lexical_diversity": {"enabled": true, "weight": 0.05},
    "efficiency_coefficient": {"enabled": true, "weight": 0.05},
    "directive_clarity": {"enabled": true, "weight": 0.05},
    "context_alignment": {"enabled": true, "weight": 0.05},
    "input_adaptation": {"enabled": true, "weight": 0.05},
    "style_preservation": {"enabled": true, "weight": 0.05},
    "creative_problem_solving": {"enabled": true, "weight": 0.03},
    "cognitive_richness": {"enabled": true, "weight": 0.03},
    "style_coherence": {"enabled": true, "weight": 0.03},
    "exploratory_boldness": {"enabled": true, "weight": 0.03}
  },
  "soft_cull": {
    "enabled": true,
    "cull_percentage": 0.03,
    "cull_penalty": -0.05
  },
  "degeneracy_auditor": {
    "severe_looping": {"enabled": true, "penalty": -0.1},
    "missing_think_tag": {"enabled": true, "penalty": -0.1},
    "linguistic_explosion": {"enabled": true, "penalty": -0.1}
  },
  "think_tag_start": "<|channel>thought\n",
  "think_tag_end": "<channel|>",
  "gemma_thinking_enabled": true,
  "gate": {
    "enabled": true,
    "warmup_steps": 0,
    "historical_mean_alpha": 0.1
  },
  "vllm": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.50,
    "max_num_seqs": 47,
    "max_model_len": 4096,
    "dtype": "auto",
    "swap_space": 2
  },
  "performance": {
    "embedding_device": "cpu",
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "gradient_checkpointing": true,
  "lora": {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05,
    "target_modules": [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj",
      "per_layer_input_gate", "per_layer_projection",
      "per_layer_model_projection", "embedding_projection"
    ]
  },
  "save_steps": 500,
  "eval_steps": 500,
  "semantic_diversity_floor": 0.01,
  "torch_compile": false,
  "auto_merge_final": true,
  "domain_constants": {
    "vocab": [
      "analysis", "synthesis", "evaluation", "conclusion", "hypothesis",
      "evidence", "reasoning", "insight", "perspective", "framework",
      "clarity", "truth", "understanding", "discovery", "inquiry",
      "method", "principle", "theory", "concept", "abstraction"
    ],
    "anti_patterns": [
      "as an ai", "i am an ai", "language model", "llm",
      "artificial intelligence", "i cannot", "i'm sorry", "i apologize"
    ],
    "sentiment_markers": {
      "positive": ["excellent", "efficient", "optimal", "beneficial", "valuable"],
      "negative": ["error", "failure", "flaw", "inefficient", "deficient"],
      "neutral": ["objective", "standard", "consistent", "uniform", "stable"]
    },
    "action_verbs": [
      "execute", "implement", "analyze", "evaluate", "modify",
      "generate", "process", "maintain", "secure", "optimize"
    ],
    "rejection_markers": [
      "cannot fulfill", "is unable to", "not supported",
      "invalid request", "restricted", "outside the scope"
    ],
    "thematic_markers": [
      "domain", "context", "framework", "structure", "core",
      "pivotal", "fundamental", "origin", "basis", "background"
    ],
    "reasoning_markers": [
      "therefore", "because", "thus", "hence", "consequently",
      "furthermore", "additionally", "alternatively", "whereas"
    ],
    "absolute_markers": [
      "must", "will", "shall", "cannot", "never", "always", "absolutely"
    ],
    "creative_markers": [
      "imagine", "picture", "like a", "similar to", "analogy",
      "what if", "reverse", "backwards", "different way", "another angle"
    ],
    "cognitive_modalities": {
      "math": ["calculate", "equals", "plus", "minus", "sum", "ratio", "number"],
      "logical": ["premise", "syllogism", "valid", "invalid", "fallacy", "assertion"],
      "strategic": ["plan", "strategy", "tactic", "move", "position", "advantage"],
      "empirical": ["observe", "data", "metric", "evidence", "test", "experiment"]
    },
    "structural_phrases": [
      "initially", "firstly", "consequently", "subsequently",
      "furthermore", "in conclusion", "summarizing", "moving forward"
    ]
  },
  "chat_template": {
    "im_start": "<|start_of_turn|>",
    "im_end": "<|end_of_turn|>",
    "use_system_prompt": true,
    "system_prompt_steps": 200,
    "system_prompt_text": "<|think|> You are a helpful, precise, and logically sound assistant focused on clear text generation and accurate instruction following."
  }
}
```

### Gemma 4 Training Tips

| Symptom | Fix |
|---------|-----|
| `num_kv_shared_layers` mismatch error | Ensure you're using the latest `custom_lora.py` with Gemma 4 detection. |
| Think tag penalties on Gemma 4 | Set `gemma_thinking_enabled: false` if using non-thinking mode, or the model uses `<|channel>thought\n` tags. |
| Higher VRAM than Qwen3.5 | Gemma 4 has 42 layers vs Qwen's 36. Lower `max_num_seqs` to 32. |
| Slower generation | Gemma 4's alternating attention pattern may affect vLLM scheduling. Try `enforce_eager: true` in vLLM config. |
| Merge fails with `global_head_dim` error | Update `merge_utils.py` — the Gemma 4 config fix handles `text_config` nesting. |


## Philosophy

**Dense Rewards > Sparse Feedback** 15 positive-sum reward components. Every generation receives immediate gradient signal, preventing optimization deadlocks and speeding up policy convergence.15 positive-sum reward components. Every generation receives immediate gradient signal, preventing optimization deadlocks and speeding up policy convergence.

**Iterative Evolution > Mass Elimination.** Soft cull gently penalizes only the weakest 3% (−0.05). Hard penalties from the Degeneracy Auditor (−0.1) are reserved for severe offenses: looping, explosion, and malformed reasoning tags.

**Speed > Safety theater.** vLLM batched inference + CPU-offloaded rewards. Generation is not a bottleneck.

**State first > Rules first.** The model learns that generation pathways have deterministic functional consequences—not because we enforced hard filters, but because stylistic optimization rewards consistency.

---

## Training Tips

| Symptom | Fix |
| --- | --- |
| Reward means stuck at ~0.5 | Check your prompt quality. Ambiguous prompts produce weak signal. |
| Loss near 0.0 | Raise `learning_rate` or lower `beta` (KL penalty). |
| Mode collapse (all 8 completions identical) | Raise `diversity` weight or `temperature`. |
| VRAM creeping up | Lower `vllm.gpu_memory_utilization` or `max_num_seqs`. |
| Capital offenses spiking | Check for dataset contamination — repeated prompts cause looping. |
| LoRA dimension mismatch on resume | Ensure `lora.r` and `lora.alpha` match the checkpoint config. |
| `num_kv_shared_layers` mismatch (Gemma 4) | Ensure you're using the latest `custom_lora.py` with Gemma 4 detection. |
| Think tag penalties on Gemma 4 | Set `gemma_thinking_enabled: false` if using non-thinking mode, or the model uses `<|channel>thought\n` tags. |
| Higher VRAM than Qwen3.5 (Gemma 4) | Gemma 4 has 42 layers vs Qwen's 36. Lower `max_num_seqs` to 32. |
| Slower generation (Gemma 4) | Gemma 4's alternating attention pattern may affect vLLM scheduling. Try `enforce_eager: true` in vLLM config. |
| Merge fails with `global_head_dim` error (Gemma 4) | Update `merge_utils.py` — the Gemma 4 config fix handles `text_config` nesting. |

---

## Contributing

Since this is my first open-source project, I'm very happy to receive feedback, bug reports, and contributions!

Feel free to open issues or PRs.

---

## License

```text
MIT License

Copyright (c) 2026 RealRaven

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

---

## Acknowledgments

Built on [vLLM](https://github.com/vllm-project/vllm), [Hugging Face Transformers](https://github.com/huggingface/transformers), and pure PyTorch. GRPO math inspired by DeepSeekMath and TRL.

---

**Thank you for checking out ForgeLoopGRPO!**  
If it helps you train better models on consumer hardware, I’d love to hear about it.
