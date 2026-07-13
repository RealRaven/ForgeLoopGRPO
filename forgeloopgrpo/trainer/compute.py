"""Reward, advantage, logprob computation — pure HF Transformers, no vLLM."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

from ..rewards.engine import PolicyRewardOrchestrationEngine


class ComputeMixin:
    def __init__(self, reward_engine: PolicyRewardOrchestrationEngine, config, model=None, tokenizer=None):
        self.reward_engine = reward_engine
        self.config = config
        self.beta = config.beta
        self.model = model
        self.tokenizer = tokenizer

    def compute_rewards(self, texts: List[str], prompts: List[str],
                        metainfos: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Compute rewards for a group of completions."""
        rewards, diagnostics = self.reward_engine.score_group(
            texts, prompts, metainfos
        )
        return rewards, diagnostics

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """GRPO advantages from rewards."""
        return self.reward_engine.compute_advantages(rewards)

    def compute_logprobs(self, model, tokenizer, prompts, completions,
                         is_policy=True, micro_batch=1) -> torch.Tensor:
        """
        Compute sequence log probabilities using strict token-boundary matching.
        """
        model = model or self.model
        tokenizer = tokenizer or self.tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("Model and tokenizer must be provided")

        device = next(model.parameters()).device
        
        # Resolve padding token to ensure right-padded operations match standard configurations
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        model_vocab_size = getattr(unwrapped.config, "vocab_size", len(tokenizer))
        all_computed_logprobs = []
        batch_size = 4  

        for b_start in range(0, len(prompts), batch_size):
            b_end = b_start + batch_size
            b_prompts = prompts[b_start:b_end]
            b_completions = completions[b_start:b_end]

            batch_input_ids = []
            batch_attention_mask = []
            batch_comp_mask = []
            max_len = 0

            # Tokenize separately to eliminate context-dependent boundary shifts
            for p, c in zip(b_prompts, b_completions):
                p_ids = tokenizer(p, add_special_tokens=True)["input_ids"]
                c_ids = tokenizer(c, add_special_tokens=False)["input_ids"]
                
                combined_ids = p_ids + c_ids
                max_len = max(max_len, len(combined_ids))
                
                # Maintain parallel tracking for completion masks
                c_mask = [0.0] * len(p_ids) + [1.0] * len(c_ids)
                
                batch_input_ids.append(combined_ids)
                batch_comp_mask.append(c_mask)

            # Manual right-padding to construct aligned execution blocks
            for idx in range(len(batch_input_ids)):
                curr_len = len(batch_input_ids[idx])
                pad_needed = max_len - curr_len
                
                batch_attention_mask.append([1] * curr_len + [0] * pad_needed)
                batch_input_ids[idx] = batch_input_ids[idx] + [tokenizer.pad_token_id] * pad_needed
                batch_comp_mask[idx] = batch_comp_mask[idx] + [0.0] * pad_needed

            input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)
            comp_mask = torch.tensor(batch_comp_mask, dtype=torch.float32, device=device)

            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            with torch.set_grad_enabled(is_policy):
                with torch.amp.autocast(device_type='cuda', dtype=target_dtype):
                    if is_policy:
                        model.train()
                    else:
                        model.eval()

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            B, S, V = logits.shape
            effective_vocab_size = min(V, model_vocab_size)

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_comp_mask = comp_mask[:, 1:]

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)

            # Zero out positions pointing to out-of-bounds vocab tokens
            invalid_label_mask = (shift_labels >= effective_vocab_size) | (shift_labels < 0)
            safe_labels = shift_labels.clone()
            safe_labels[invalid_label_mask] = 0
            
            # Switch off computation flag for invalid positions
            shift_comp_mask[invalid_label_mask] = 0.0

            gathered = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

            for i in range(B):
                seq_comp_mask = shift_comp_mask[i]
                seq_logprobs = gathered[i]

                denom = seq_comp_mask.sum()
                if denom > 0:
                    avg_logprob = (seq_logprobs * seq_comp_mask).sum() / denom
                else:
                    avg_logprob = torch.tensor(0.0, device=device)

                if torch.isnan(avg_logprob) or torch.isinf(avg_logprob):
                    avg_logprob = torch.tensor(0.0, device=device)

                if not is_policy:
                    avg_logprob = avg_logprob.detach()

                all_computed_logprobs.append(avg_logprob)

            del logits, shift_logits, log_probs, gathered, comp_mask, shift_comp_mask
            if not is_policy:
                torch.cuda.empty_cache()

        return torch.stack(all_computed_logprobs)