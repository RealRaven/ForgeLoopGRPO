"""Custom optimizer — no HF, no PEFT, just torch."""

import torch
from typing import List


class CustomOptimizer:
    """Simple AdamW wrapper for our custom LoRA parameters."""

    def __init__(self, parameters: List[torch.nn.Parameter], lr: float = 5e-6):
        self.optimizer = torch.optim.AdamW(parameters, lr=lr)

    @property
    def state(self):
        """Expose the underlying PyTorch optimizer state directly for device swapping."""
        return self.optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)