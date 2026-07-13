"""Adaptive gate — zero-lag calibration. Continuous policy pressure from step one."""

import math
from typing import Dict, Any


class AdaptiveGate:
    """A zero-drag baseline filter tracking real-time performance fluctuations."""

    def __init__(self, alpha: float = 0.1, enabled: bool = True, scale_factor: float = 0.5):
        self.enabled = enabled
        self.alpha = alpha
        self.scale_factor = max(1e-5, scale_factor)  # Prevent division by zero
        self.historical_mean = 0.0
        self.step_count = 0

    def update(self, batch_mean: float):
        if not self.enabled:
            return
        
        # Eliminate zero-initialization cold-start drag
        if self.step_count == 0:
            self.historical_mean = batch_mean
        else:
            self.historical_mean = self.alpha * batch_mean + (1 - self.alpha) * self.historical_mean
        
        self.step_count += 1

    def weight(self, class_mean: float) -> float:
        if not self.enabled:
            return 1.0
        
        diff = class_mean - self.historical_mean
        scaled = diff / self.scale_factor
        
        # Native math sigmoid — no torch tensor creation / sync overhead
        # Clamp to prevent overflow/underflow in extreme edge cases
        scaled = max(-20.0, min(20.0, scaled))
        return 1.0 / (1.0 + math.exp(-scaled))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "historical_mean": self.historical_mean,
            "step_count": self.step_count,
            "alpha": self.alpha,
            "scale_factor": self.scale_factor,
            "enabled": self.enabled
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.historical_mean = state.get("historical_mean", 0.0)
        self.step_count = state.get("step_count", 0)
        self.alpha = state.get("alpha", 0.1)
        self.scale_factor = state.get("scale_factor", 0.5)
        self.enabled = state.get("enabled", True)