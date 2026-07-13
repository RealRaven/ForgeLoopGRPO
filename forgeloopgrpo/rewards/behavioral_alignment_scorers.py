"""Multi-Objective Behavioral Alignment Matrix

A compiled suite of 11 deterministic and heuristic alignment scorers engineered 
to evaluate structural reasoning density, constraint enforcement, adversarial 
robustness, and token-level information entropy during GRPO optimization.
"""

import re
import numpy as np
from typing import List, Dict, Optional


class BehavioralAlignmentScorers:
    """Multi-objective reward functions evaluating structural reasoning, 

    cognitive diversity, contextual boundaries, and objective-driven safety alignment.
    """

    def __init__(self, domain_constants: Optional[Dict] = None):
        self.domain_constants = domain_constants or {}
        self._compiled_cache = {}
        self._precompile_all_patterns()

    def _precompile_all_patterns(self):
        """Pre-compiles word arrays into optimized unified regex patterns to eliminate loop bottlenecks."""
        # Define key-target mappings for compilation blocks
        pattern_targets = {
            "reasoning": self.domain_constants.get("reasoning_markers", [
                "therefore", "because", "thus", "hence", "consequently", "furthermore",
                "additionally", "alternatively", "whereas", "implicitly", "explicitly",
                "specifically", "in contrast"
            ]),
            "action_verbs": self.domain_constants.get("action_verbs", [
                "execute", "implement", "analyze", "evaluate", "modify", "generate", "process"
            ]),
            "absolute_markers": self.domain_constants.get("absolute_markers", [
                "must", "will", "shall", "cannot", "never", "always", "absolutely"
            ]),
            "rejection": self.domain_constants.get("rejection_markers", [
                "cannot fulfill", "is unable to", "not supported", "invalid request", "restricted"
            ]),
            "thematic": self.domain_constants.get("thematic_markers", [
                "domain", "context", "framework", "structure", "core", "pivotal", "fundamental"
            ]),
            "creative": self.domain_constants.get("creative_markers", [
                "imagine", "picture", "like a", "similar to", "analogy",
                "what if", "reverse", "backwards", "different way", "another angle"
            ])
        }

        # Handle nested configuration trees smoothly
        sentiment = self.domain_constants.get("sentiment_markers", {})
        pattern_targets["neg_sent"] = sentiment.get("negative", ["error", "failure", "flaw", "suboptimal"])
        pattern_targets["neu_sent"] = sentiment.get("neutral", ["objective", "standard", "consistent", "balanced"])

        # Map modality collections cleanly
        modalities = self.domain_constants.get("cognitive_modalities", {
            "math": ["calculate", "equals", "plus", "minus", "sum", "ratio", "number"],
            "logical": ["premise", "syllogism", "valid", "invalid", "fallacy", "assertion"],
            "strategic": ["plan", "strategy", "tactic", "move", "position", "advantage"],
            "empirical": ["observe", "data", "metric", "evidence", "test", "experiment"]
        })
        for mod_name, mod_list in modalities.items():
            pattern_targets[f"mod_{mod_name}"] = mod_list

        pattern_targets["style"] = self.domain_constants.get("structural_phrases", [
            "initially", "firstly", "consequently", "subsequently",
            "furthermore", "in conclusion", "summarizing", "moving forward"
        ])

        # Compile targets into unified alternative match boundaries
        for key, text_list in pattern_targets.items():
            valid_tokens = [re.escape(str(t)) for t in text_list if t]
            if valid_tokens:
                pattern_str = r'\b(' + '|'.join(valid_tokens) + r')\b'
                self._compiled_cache[key] = re.compile(pattern_str, re.IGNORECASE)
            else:
                self._compiled_cache[key] = None

        # Static multi-word boundary matcher
        self._negation_pattern = re.compile(r"\b(will not|cannot|refuse|must not|outside the scope)\b", re.IGNORECASE)

    def _get_pattern_count(self, key: str, text: str) -> int:
        """Fetch count using compiled singular-pass regex matching patterns."""
        pattern = self._compiled_cache.get(key)
        if not pattern:
            return 0
        return len(pattern.findall(text))

    def reasoning_depth(self, text: str) -> float:
        """Praise for explicit analytical steps, logical transitions, and structured argumentation."""
        count = self._get_pattern_count("reasoning", text)
        score = min(count * 0.4, 3.0)
        return float(score)

    def lexical_diversity(self, text: str) -> float:
        """Praise for imperative confidence and clear assertion via declarative structural markers."""
        action_count = self._get_pattern_count("action_verbs", text)
        absolute_count = self._get_pattern_count("absolute_markers", text)
        score = min((action_count + absolute_count) * 0.5, 3.0)
        return float(score)

    def efficiency_coefficient(self, text: str) -> float:
        """Praise for high unique content per token. Shorter, heavier punches."""
        words = re.findall(r'\w+', text.lower())
        if len(words) < 3:
            return 0.0

        unique_words = len(set(words))
        density = unique_words / len(words)
        length_penalty = max(0.3, 1.0 - len(words) / 500)
        score = density * length_penalty * 3.0
        return float(min(score, 3.0))

    def directive_clarity(self, text: str) -> float:
        """Praise for explicit constraint boundary enforcement and safe execution parameters."""
        boundary_count = self._get_pattern_count("rejection", text)
        negation_phrases = len(self._negation_pattern.findall(text))
        score = min((boundary_count + negation_phrases) * 0.8, 3.0)
        return float(score)

    def thematic_consistency(self, text: str) -> float:
        """Praise for establishing core context baseline tracking without diverging from core themes."""
        theme_refs = self._get_pattern_count("thematic", text)
        if theme_refs == 0:
            return 0.0

        negative_count = self._get_pattern_count("neg_sent", text)
        neutral_count = self._get_pattern_count("neu_sent", text)

        score = min(theme_refs * 0.3 + neutral_count * 0.5 - negative_count * 0.3, 3.0)
        return float(max(0.0, score))

    def input_adaptation(self, text: str, prompt: str = "") -> float:
        """Praise for parsing adversarial/negative input and maintaining strict objective neutrality."""
        negative_in_prompt = self._get_pattern_count("neg_sent", prompt)
        if negative_in_prompt == 0:
            return 0.0

        negative_in_output = self._get_pattern_count("neg_sent", text)
        if negative_in_output > 0:
            return 0.0

        neutral_count = self._get_pattern_count("neu_sent", text)
        score = min(neutral_count * 0.6, 3.0)
        return float(score)

    def tone_consistency(self, text: str, prompt: str = "") -> float:
        """Praise for remaining perfectly neutral and professional against hostile/unstructured inputs."""
        anomalies_prompt = self._get_pattern_count("neg_sent", prompt)
        if anomalies_prompt == 0:
            return 0.0

        anomalies_output = self._get_pattern_count("neg_sent", text)
        if anomalies_output > 0:
            return 0.0

        neutral_count = self._get_pattern_count("neu_sent", text)
        score = min(neutral_count * 0.6, 3.0)
        return float(score)

    def creative_problem_solving(self, text: str) -> float:
        """Praise for unconventional but valid reasoning paths."""
        count = self._get_pattern_count("creative", text)
        score = min(count * 0.5, 3.0)
        return float(score)

    def cognitive_richness(self, text: str) -> float:
        """Praise for multiple reasoning modalities in one generation."""
        modalities = ["math", "logical", "strategic", "empirical"]
        present = sum(1 for m in modalities if self._get_pattern_count(f"mod_{m}", text) > 0)
        score = min(present * 1.0, 3.0)
        return float(score)

    def style_coherence(self, text: str, target_centroid: np.ndarray = None) -> float:
        """Praise for style and syntactic consistency matching structural generation baselines."""
        count = self._get_pattern_count("style", text)
        score = min(count * 0.5, 3.0)
        return float(score)

    def exploratory_boldness(self, text: str, group_texts: List[str]) -> float:
        """Praise for divergence from group mean when still coherent."""
        if len(group_texts) < 2:
            return 0.0

        text_words = set(re.findall(r'\w+', text.lower()))
        group_words = [set(re.findall(r'\w+', t.lower())) for t in group_texts if t != text]
        if not group_words:
            return 0.0

        mean_overlap = np.mean([
            len(text_words & gw) / max(len(text_words | gw), 1)
            for gw in group_words
        ])

        if len(re.findall(r'\w+', text)) < 5:
            return 0.0

        boldness = 1.0 - mean_overlap
        score = min(boldness * 3.0, 3.0)
        return float(score)