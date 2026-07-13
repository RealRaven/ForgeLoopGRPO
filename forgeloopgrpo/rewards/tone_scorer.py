"""Tone Consistency Scorer

Evaluates text sequence alignment against target profile markers specified 
in runtime metadata configurations. Bounded and positive-sum.
"""

import re


class ToneScorer:
    """Praise for natural, flowing target keyword density distributions."""

    def __init__(self, config):
        # Dynamically pull markers from the updated generic config tree structure
        self.markers = {}
        if hasattr(config, "domain_constants"):
            constants = config.domain_constants
            raw_markers = getattr(constants, "sentiment_markers", {})
            if hasattr(raw_markers, "model_dump"):
                self.markers = raw_markers.model_dump()
            elif isinstance(raw_markers, dict):
                self.markers = raw_markers
        
        # Pre-compile unified pattern matchers for enhanced runtime speed
        self._compiled_modes = {}
        self._global_pattern = None
        self._build_compiled_cache()

    def _build_compiled_cache(self):
        """Compile list components into unified singular-pass regex matchers."""
        all_markers = []
        for mode, marker_list in self.markers.items():
            if not marker_list:
                continue
            # Escape and wrap individual markers in word boundary expressions
            escaped_markers = [re.escape(str(m)) for m in marker_list if m]
            if escaped_markers:
                pattern_str = r'\b(' + '|'.join(escaped_markers) + r')\b'
                self._compiled_modes[mode] = re.compile(pattern_str, re.IGNORECASE)
                all_markers.extend(escaped_markers)
        
        if all_markers:
            global_str = r'\b(' + '|'.join(all_markers) + r')\b'
            self._global_pattern = re.compile(global_str, re.IGNORECASE)

    def _word_count(self, text: str) -> int:
        """Consistent word-token count aligned with regex \\b boundary semantics."""
        return len(re.findall(r'\w+', text))

    def score(self, text: str, target_mode: str = "") -> float:
        """Calculates a density distribution match score between the generated text 

        and the token targets specified by the objective mode configuration.
        """
        total_words = self._word_count(text)
        if total_words == 0:
            return 0.0

        # If no explicit target is passed or target_mode is missing, evaluate globally
        if not target_mode or target_mode not in self.markers:
            if not self._global_pattern:
                return 0.0
            all_hits = len(self._global_pattern.findall(text))
            density = all_hits / total_words
            score = min(density * 20.0, 3.0)
            return float(score)

        # Retrieve pre-compiled specific targeted pattern engine
        compiled_regex = self._compiled_modes.get(target_mode)
        if not compiled_regex:
            return 1.5  # Neutral structural midpoint

        hits = len(compiled_regex.findall(text))
        density = hits / total_words

        # Saturation curve optimization: 0.15 density distribution maps to max praise (3.0)
        score = min(density * 20.0, 3.0)
        return float(score)