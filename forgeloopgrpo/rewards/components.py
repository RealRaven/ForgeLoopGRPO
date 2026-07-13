"""Base reward scorers — Structural Vocabulary Alignment and Fluency. All positive-sum."""

import re
import numpy as np


class VocabularyAlignmentScorer:
    """Soft vocabulary density distribution tracking. Rewards core domain lexical overlap."""

    def __init__(self, config):
        # Accessing vocab and anti_patterns from the Pydantic config model parameters
        self.vocab = set(config.domain_constants.vocab)
        self.anti_patterns = config.domain_constants.anti_patterns
        
        # Pre-compile anti-pattern regexes to enforce strict word boundaries and protect performance
        self._compiled_anti = [
            re.compile(r'\b' + re.escape(p.lower()) + r'\b') for p in self.anti_patterns if p
        ]

    def score(self, text: str) -> float:
        text_lower = text.lower()
        tokens = re.findall(r'\b\w+\b', text_lower)
        if not tokens:
            return 0.0

        vocab_hits = sum(1 for token in tokens if token in self.vocab)
        vocab_ratio = vocab_hits / len(tokens)

        # Enforce boundary checking on structural anti-patterns to prevent accidental substring leakage
        anti_hits = sum(1 for pattern in self._compiled_anti if pattern.search(text_lower))

        score = vocab_ratio * 3.0 - anti_hits * 0.5
        return float(max(0.0, min(3.0, score)))


class FluencyScorer:
    """Praise for sequence syntactic excellence. Validates clean, structured prose distributions."""

    def __init__(self, config=None):
        self.config = config

    def score(self, text: str) -> float:
        if not text or len(text) < 5:
            return 0.0

        # Length normalization score tracking
        length = len(text.split())
        if length < 10:
            length_score = length / 10.0
        elif length < 200:
            length_score = 1.0
        else:
            length_score = max(0.3, 1.0 - (length - 200) / 500)

        # Punctuation ratio consistency
        punct_count = sum(1 for c in text if c in '.,!?;:')
        punct_ratio = punct_count / max(len(text), 1)
        punct_score = 1.0 if 0.02 < punct_ratio < 0.15 else 0.5

        # Fixes Sentence Splitting: split on punctuation boundaries followed immediately by whitespace
        # This keeps floating-point numbers or short abbreviations from breaking sentence counts
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return 0.0

        cleaned_sentences = []
        for s in sentences:
            first_alpha_match = re.search(r'[a-zA-Z]', s)
            if first_alpha_match:
                cleaned_sentences.append(s[first_alpha_match.start():])
            else:
                cleaned_sentences.append(s)

        cap_count = sum(1 for s in cleaned_sentences if s and s[0].isupper())
        cap_score = cap_count / len(sentences)

        # Evaluation of variation distributions in token sequence counts
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            std_dev = np.std(sent_lengths)
            if std_dev < 1.0:
                variety = 0.2
            else:
                variety = min(std_dev / 8.0, 1.0) if std_dev <= 12.0 else max(0.4, 1.0 - (std_dev - 12.0) / 20.0)
        else:
            variety = 0.5

        score = (length_score + punct_score + cap_score + variety) / 4.0 * 3.0
        return float(max(0.0, min(3.0, score)))