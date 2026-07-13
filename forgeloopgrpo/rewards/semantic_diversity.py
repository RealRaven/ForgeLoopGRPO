"""Gram matrix semantic diversity scorer — natural anti-collapse via cosine distance."""

import numpy as np
from typing import List
from ..embeddings import CPUEmbeddingStore


class SemanticDiversityScorer:
    """Rewards unique phrasing. Low floor — let collapse happen and recover naturally."""

    def __init__(self, embedding_store: CPUEmbeddingStore, floor: float = 0.01):
        self.embeddings = embedding_store
        self.floor = floor

    def score(self, texts: List[str]) -> List[float]:
        if len(texts) < 2:
            return [1.5] * len(texts)

        # Retrieve cross-similarity Gram matrix [N, N]
        gram = self.embeddings.gram_matrix(texts)

        # Bound similarities to [0, 1] to preserve linear semantic_diversity scaling
        gram = np.clip(gram, 0.0, 1.0)

        # Force diagonal to absolute zero to handle half-precision rounding variations safely
        np.fill_diagonal(gram, 0.0)

        # Compute row sums directly over group variations
        adjusted_sums = np.sum(gram, axis=1)

        # Divide by N-1 so the mean reflects only *other* texts
        n_others = len(texts) - 1
        mean_sims = adjusted_sums / n_others

        # Semantic Diversity = dissimilarity to the group mean
        semantic_diversity = 1.0 - mean_sims

        # Scale to [0, 3]
        scores = np.clip(semantic_diversity * 3.0, 0.0, 3.0)
        scores = np.maximum(scores, self.floor * 3.0)  # Gentle floor

        return [float(s) for s in scores]