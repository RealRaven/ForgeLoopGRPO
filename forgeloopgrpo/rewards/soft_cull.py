"""Soft cull advantage adjustment engine.

Modifies reinforcement learning advantages to widen the contrast gap against 
the lowest-performing generation trajectories. Bounded and positive-sum.
"""

import numpy as np
from typing import List, Tuple


class SoftCull:
    """Reward shaping filter that applies a negative penalty to the lowest-performing completions."""

    def __init__(self, percentage: float = 0.03, penalty: float = -0.05):
        self.percentage = percentage
        self.penalty = penalty

    def apply_to_advantages(self, advantages: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Identifies the bottom N% of advantages per group and applies a step-down penalty.

        Modifying advantages directly ensures group standard deviation calculations 
        are not structurally distorted for normal trajectories.
        """
        if len(advantages) < 3:
            return advantages, []

        # Guard: if all advantages are identical, culling is redundant noise
        if np.isclose(np.min(advantages), np.max(advantages)):
            return advantages, []

        n_cull = max(1, int(len(advantages) * self.percentage))
        sorted_indices = np.argsort(advantages)
        culled_indices = sorted_indices[:n_cull].tolist()

        modified = advantages.copy()
        # Direct scalar shift applied past baseline distribution mapping
        modified[culled_indices] += self.penalty

        return modified, culled_indices