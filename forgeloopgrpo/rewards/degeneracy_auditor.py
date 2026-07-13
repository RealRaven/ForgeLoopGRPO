"""Degeneracy Auditor Evaluation Rule Layer.

Monitors and penalizes severe looping, matrix blackout, and linguistic explosion
events across generative optimization pipelines. Bounded and positive-sum.

Updated for dual-model support:
  - Qwen3.5: uses <think> / </think> tags
  - Gemma 4: uses <|channel>thought\n / <channel|> tags (when thinking enabled)
"""

import re
import numpy as np
from typing import List, Dict, Tuple


class DegeneracyAuditor:
    """Monitors and penalizes severe looping, matrix blackout, and linguistic explosion."""

    def __init__(self, config: Dict):
        self.config = config

    def _get_clean_tokens(self, text: str) -> List[str]:
        """Extract uniform alphanumeric tokens to eliminate punctuation-skewed boundaries."""
        return re.findall(r'\w+', text.lower())

    def check_severe_looping(self, text: str) -> bool:
        """Same n-gram repeated >6 times, or >80% identical tokens on substantial texts."""
        words = self._get_clean_tokens(text)
        if len(words) < 20:
            return False

        # Check 4-gram repetition
        ngrams = {}
        for i in range(len(words) - 3):
            gram = " ".join(words[i:i+4])
            ngrams[gram] = ngrams.get(gram, 0) + 1

        max_repeats = max(ngrams.values()) if ngrams else 0
        if max_repeats > 6:
            return True

        # Prevent false positives on completely valid short texts
        if len(words) >= 75:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:  # 80%+ identical tokens
                return True

        return False

    def missing_think_tag(self, text: str) -> bool:
        """Returns True if required reasoning tags are missing, malformed, or empty.

        Supports both Qwen3.5 (<think>...</think>) and 
        Gemma 4 (<|channel>thought\n...<channel|>) tag formats.
        """
        # Auto-detect Gemma family from config path if explicit flag isn't present
        model_path = self.config.get("model_path", "").lower()
        is_gemma = self.config.get("_is_gemma_model", "gemma" in model_path)
        gemma_thinking_enabled = self.config.get("gemma_thinking_enabled", True)

        # GEMMA 4 Native Execution Block
        if is_gemma:
            if not gemma_thinking_enabled:
                return False  # Thinking mode disabled explicitly, skip audit

            # Aligned precisely with standard vocabulary tokenization schemas
            gemma_start = self.config.get("think_tag_start", "<|channel>thought\n")
            gemma_end = self.config.get("think_tag_end", "<channel|>")

            if gemma_start not in text or gemma_end not in text:
                return True

            start_idx = text.find(gemma_start) + len(gemma_start)
            end_idx = text.rfind(gemma_end)

            if end_idx <= start_idx:
                return True

            think_content = text[start_idx:end_idx]
            return not bool(think_content.strip())

        # QWEN3.5 / GENERIC Configuration Fallback
        start_tag = self.config.get("think_tag_start", "<think>")
        end_tag = self.config.get("think_tag_end", "</think>")

        if start_tag not in text or end_tag not in text:
            return True

        # Evaluate rightmost edge to capture full trace span safely
        start_idx = text.find(start_tag) + len(start_tag)
        end_idx = text.rfind(end_tag) 

        if end_idx <= start_idx:
            return True

        think_content = text[start_idx:end_idx]
        return not bool(think_content.strip())

    def check_linguistic_explosion(self, text: str, safe_baseline_length: float) -> bool:
        """Exceeds 4x safe baseline length, or >30% garbled unprintable streams."""
        tokens = self._get_clean_tokens(text)
        if len(tokens) > safe_baseline_length * 4.0:
            return True

        if not text or len(text.strip()) == 0:
            return True

        # Check for raw unprintable characters rather than penalizing standard Unicode paths
        unprintable_chars = sum(1 for c in text if not c.isprintable() and not c.isspace())
        if (unprintable_chars / len(text)) > 0.3:
            return True

        return False

    def audit(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """Returns penalties array and offense report."""
        penalties = np.zeros(len(texts))
        report = {"severe_looping": [], "missing_think_tag": [], "linguistic_explosion": []}

        lengths = [len(self._get_clean_tokens(t)) for t in texts]
        median_length = float(np.median(lengths)) if lengths else 50.0
        safe_baseline_length = max(20.0, median_length)

        for i, text in enumerate(texts):
            if self.config.get("severe_looping", {}).get("enabled", True):
                if self.check_severe_looping(text):
                    penalties[i] += self.config.get("severe_looping", {}).get("penalty", -3.0)
                    report["severe_looping"].append(i)

            if self.config.get("missing_think_tag", {}).get("enabled", True):
                if self.missing_think_tag(text):
                    penalties[i] += self.config.get("missing_think_tag", {}).get("penalty", -3.0)
                    report["missing_think_tag"].append(i)

            if self.config.get("linguistic_explosion", {}).get("enabled", True):
                if self.check_linguistic_explosion(text, safe_baseline_length):
                    penalties[i] += self.config.get("linguistic_explosion", {}).get("penalty", -3.0)
                    report["linguistic_explosion"].append(i)

        return penalties, report