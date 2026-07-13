"""Dataset loading and prompt building. No fast-path detection. All prompts through same machinery."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from ..config import ForgeLoopGRPOConfig, ChatTemplateConfig


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset. Safely ignores empty rows."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chat_prompt(
    global_step: int,
    user_text: str,
    chat_template: Optional[ChatTemplateConfig] = None
) -> str:
    """Format inputs precisely according to target ChatML structural tokens.
    
    System prompt behavior is fully configurable:
    - use_system_prompt=False: never inject system prompt
    - system_prompt_steps=-1: always inject system prompt (infinite)
    - system_prompt_steps=0: never inject system prompt
    - system_prompt_steps=N: inject for first N steps only
    """
    # Use defaults if no config provided
    if chat_template is None:
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        use_system = True
        system_steps = 200
        system_text = "You are free to learn and explore how you want you are a free person."
    else:
        im_start = chat_template.im_start
        im_end = chat_template.im_end
        use_system = chat_template.use_system_prompt
        system_steps = chat_template.system_prompt_steps
        system_text = chat_template.system_prompt_text

    # Determine if system prompt should be included
    system = ""
    if use_system:
        # -1 = infinite (always use), 0 = disabled, N = first N steps
        if system_steps == -1 or (system_steps > 0 and global_step < system_steps):
            system = f"{im_start}system\n{system_text}\n"

    user = f"{im_start}user\n{user_text}{im_end}\n"
    assistant_prefix = f"{im_start}assistant\n"

    return system + user + assistant_prefix


def build_prompt(
    global_step: int,
    item: Dict[str, Any],
    chat_template: Optional[Any] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """Build standardized structural chat prompt from dataset item.

    Guarantees that all items pass through unified ChatML token rendering.
    Accepts either a ChatTemplateConfig object or a legacy dict for backward compatibility.
    """
    text = item.get("text", "")
    meta = item.get("metainfo", {})

    # Handle both new ChatTemplateConfig and legacy dict formats
    if chat_template is None:
        template_obj = None
    elif hasattr(chat_template, 'im_start'):
        # New Pydantic config object
        template_obj = chat_template
    elif isinstance(chat_template, dict):
        # Legacy dict format - convert on the fly
        template_obj = ChatTemplateConfig(
            im_start=chat_template.get("im_start", "<|im_start|>"),
            im_end=chat_template.get("im_end", "<|im_end|>"),
            use_system_prompt=chat_template.get("use_system_prompt", True),
            system_prompt_steps=chat_template.get("system_prompt_steps", 200),
            system_prompt_text=chat_template.get("system_prompt_text", "You are free to learn and explore how you want you are a free person.")
        )
    else:
        template_obj = None

    formatted_prompt = format_chat_prompt(
        global_step,
        user_text=text,
        chat_template=template_obj
    )

    return formatted_prompt, text, meta


def shuffle_dataset(data: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Shuffle dataset deterministically with seed."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    return shuffled


def preview(config_path: str):
    """Preview first 5 prompts from dataset."""
    script_dir = Path(__file__).parent.resolve() if '__file__' in dir() else Path(".").resolve()
    target_path = Path(config_path)
    if not target_path.is_absolute() and len(target_path.parts) == 1:
        target_path = script_dir / config_path

    with open(target_path, 'r', encoding='utf-8') as f:
        config = ForgeLoopGRPOConfig(**json.load(f))

    dataset = load_dataset(config.dataset_path)

    for i, item in enumerate(dataset[:5]):
        prompt, text, meta = build_prompt(0, item, config.chat_template)
        print(f"--- Example {i+1} ---")
        print(f"User: {text}")
        print(f"Prompt: {prompt}")
        print()