"""Pydantic configuration for ForgeLoop GRPO v1.0 (Generic NLP Variant)

Now supports both Qwen3.5-4B and Gemma-4-E2B-IT model families.
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional


class VLLMConfig(BaseModel):
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.50
    max_num_seqs: int = 47
    max_model_len: int = 4096
    dtype: str = "auto"
    swap_space: int = 2


class PerformanceConfig(BaseModel):
    embedding_device: str = "cpu"
    embedding_model: str = "all-MiniLM-L6-v2"


class RewardComponentConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.0


class RewardWeightsConfig(BaseModel):
    thematic_consistency: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.15))
    tone_consistency: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.10))
    semantic_diversity: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.10))
    fluency: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.10))
    reasoning_depth: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    lexical_diversity: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    efficiency_coefficient: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    directive_clarity: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    context_alignment: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    input_adaptation: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    style_preservation: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.05))
    creative_problem_solving: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.03))
    cognitive_richness: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.03))
    style_coherence: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.03))
    exploratory_boldness: RewardComponentConfig = Field(default_factory=lambda: RewardComponentConfig(weight=0.03))


class SoftCullConfig(BaseModel):
    enabled: bool = True
    cull_percentage: float = Field(default=0.03, ge=0.02, le=0.05)
    cull_penalty: float = -0.05


class DegeneracyAuditorConfig(BaseModel):
    enabled: bool = True
    penalty: float = -0.1


class DegeneracyAuditorsConfig(BaseModel):
    severe_looping: DegeneracyAuditorConfig = Field(default_factory=lambda: DegeneracyAuditorConfig(penalty=-0.1))
    missing_think_tag: DegeneracyAuditorConfig = Field(default_factory=lambda: DegeneracyAuditorConfig(penalty=-0.1))
    linguistic_explosion: DegeneracyAuditorConfig = Field(default_factory=lambda: DegeneracyAuditorConfig(penalty=-0.1))


class GateConfig(BaseModel):
    enabled: bool = True
    warmup_steps: int = 0
    historical_mean_alpha: float = 0.1


class LoRAConfig(BaseModel):
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


class ChatTemplateConfig(BaseModel):
    """ChatML template and system prompt configuration."""
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    use_system_prompt: bool = True
    system_prompt_steps: int = 200
    system_prompt_text: str = "You are a helpful, precise, and logically sound assistant focused on clear text generation and accurate instruction following."


class DomainConstantsConfig(BaseModel):
    vocab: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    sentiment_markers: Dict[str, List[str]] = Field(default_factory=dict)
    action_verbs: List[str] = Field(default_factory=list)
    rejection_markers: List[str] = Field(default_factory=list)
    thematic_markers: List[str] = Field(default_factory=list)
    reasoning_markers: List[str] = Field(default_factory=list)
    absolute_markers: List[str] = Field(default_factory=list)
    creative_markers: List[str] = Field(default_factory=list)
    cognitive_modalities: Dict[str, List[str]] = Field(default_factory=dict)
    structural_phrases: List[str] = Field(default_factory=list)


class ForgeLoopGRPOConfig(BaseModel):
    # Model selection: supports Qwen3.5-4B and Gemma-4-E2B-IT
    model_path: str = "Qwen/Qwen3.5-4B"  # or "google/gemma-4-E2B-it"
    dataset_path: str = "data/dataset.jsonl"
    output_dir: str = "outputs/forge_loop_grpo"

    num_generations: int = Field(default=8, ge=2, le=32)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    beta: float = Field(default=0.04, gt=0)
    learning_rate: float = 5e-6
    temperature: float = Field(default=0.9, gt=0)
    top_p: float = Field(default=0.95, ge=0, le=1)
    max_new_tokens: int = 512
    max_seq_length: int = 2048
    num_train_epochs: int = 3

    reward_weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)
    soft_cull: SoftCullConfig = Field(default_factory=SoftCullConfig)
    degeneracy_auditor: DegeneracyAuditorsConfig = Field(default_factory=DegeneracyAuditorsConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    domain_constants: DomainConstantsConfig = Field(default_factory=DomainConstantsConfig)
    chat_template: ChatTemplateConfig = Field(default_factory=ChatTemplateConfig)

    gradient_checkpointing: bool = True
    torch_compile: bool = False
    auto_merge_final: bool = True

    semantic_diversity_floor: float = 0.01

    # Think tag configuration (model-specific)
    # Qwen3.5: "<think>"
    # Gemma 4: "<|channel>thought\n / <channel|>" (when thinking enabled)
    think_tag_start: str = "<think>"
    think_tag_end: str = "</think>"

    # Gemma 4 specific: enable/disable thinking mode for degeneracy auditor
    # When False, missing_think_tag check is bypassed for Gemma models
    gemma_thinking_enabled: bool = True

    save_steps: int = 500
    eval_steps: int = 500

    @field_validator("soft_cull")
    @classmethod
    def validate_cull(cls, v):
        if v.cull_percentage < 0.02 or v.cull_percentage > 0.05:
            raise ValueError("cull_percentage must be strictly bounded between 0.02 and 0.05")
        return v

    @field_validator("chat_template")
    @classmethod
    def validate_chat_template(cls, v):
        if v.system_prompt_steps < -1:
            raise ValueError("system_prompt_steps must be >= -1 (-1 for infinite, 0 to disable)")
        return v


def validate_config(config_path: str) -> bool:
    """Validate config file against ForgeLoopGRPOConfig schema."""
    script_dir = Path(__file__).parent.resolve() if '__file__' in dir() else Path(".").resolve()
    target_path = Path(config_path)
    if not target_path.is_absolute() and len(target_path.parts) == 1:
        target_path = script_dir / config_path

    if not target_path.exists():
        print(f"[ForgeLoop Error] Config not found: {target_path}")
        return False

    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            config = ForgeLoopGRPOConfig(**json.load(f))
        print("[ForgeLoop] Config: PASSED")
        print(f"  gpu_memory_utilization: {config.vllm.gpu_memory_utilization}")
        print(f"  Domain vocabulary words parsed: {len(config.domain_constants.vocab)}")
        print(f"  Scorer marker sets parsed: {len([
            k for k in config.domain_constants.model_dump().keys()
            if k not in {'vocab', 'anti_patterns', 'sentiment_markers'}
        ])}")
        print(f"  Chat template: im_start='{config.chat_template.im_start}', im_end='{config.chat_template.im_end}'")
        print(f"  System prompt: enabled={config.chat_template.use_system_prompt}, steps={config.chat_template.system_prompt_steps}")

        # Model family detection
        model_lower = config.model_path.lower()
        if "gemma" in model_lower:
            print(f"  Model family: Gemma 4 (thinking_enabled={config.gemma_thinking_enabled})")
        elif "qwen" in model_lower:
            print(f"  Model family: Qwen3.5")
        else:
            print(f"  Model family: Unknown/Generic")

        return True
    except Exception as e:
        print(f"[ForgeLoop] Config: FAILED - {e}")
        return False