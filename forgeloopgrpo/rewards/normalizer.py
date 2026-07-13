"""Bounded EMA normalizer — true outlier preservation via nonlinear squashing."""

import numpy as np
from typing import Dict, Any, Union


class BoundedNormalizer:
    """Normalizes rewards to [0, 3] using mathematically sound EMA moments.

    Preserves outliers dynamically using a true nonlinear soft-squash curve.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.mean = 1.5
        # Track second moment E[X^2] instead of raw variance.
        # Initialized so that initial variance = 1.0  (3.25 - 1.5^2 = 1.0)
        self.m2 = 3.25
        self.count = 0

    @property
    def var(self) -> float:
        """Dynamically compute variance from parallel first and second moments."""
        # Guard against precision underflow producing negative variance
        return max(1e-6, self.m2 - (self.mean ** 2))

    def update(self, values: np.ndarray):
        if len(values) == 0:
            return

        batch_mean = float(np.mean(values))
        # Update second moment directly
        batch_m2 = float(np.mean(values ** 2))

        self.mean = self.alpha * batch_mean + (1 - self.alpha) * self.mean
        self.m2 = self.alpha * batch_m2 + (1 - self.alpha) * self.m2
        self.count += len(values)

    def normalize(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Maps rewards smoothly to [0, 3]. Supports both scalar floats and NumPy vectors."""
        std = np.sqrt(self.var)
        z = (value - self.mean) / std

        # True soft squash via tanh instead of hard linear clamp.
        # Maps (-inf, +inf) -> (-1, 1) smoothly, keeping outliers distinguishable.
        if isinstance(value, np.ndarray):
            soft_z = np.tanh(z / 2.0)
            normalized = 1.5 + soft_z * 1.5
            return normalized
        else:
            soft_z = np.tanh(z / 2.0)
            normalized = 1.5 + soft_z * 1.5
            return float(normalized)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "m2": self.m2,
            "count": self.count,
            "alpha": self.alpha
        }

    def load_state_dict(self, state: Dict[str, Any]):
        # Fetch state checkpoint attributes cleanly first to maintain historical alignment
        incoming_mean = state.get("mean", state.get("ema_mean", 1.5))
        self.mean = incoming_mean
        self.count = state.get("count", 0)
        self.alpha = state.get("alpha", self.alpha)  # Fallback to current alpha if missing

        # Graceful backward compatibility with old checkpoints using synchronized states
        if "m2" in state:
            self.m2 = state["m2"]
        else:
            legacy_var = state.get("var", state.get("ema_var", 1.0))
            # Calculate the second moment strictly relative to the incoming historical mean,
            # not the default initialized 1.5 values.
            self.m2 = legacy_var + (incoming_mean ** 2)