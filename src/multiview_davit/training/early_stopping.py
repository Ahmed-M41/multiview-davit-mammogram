"""Composite-score-based early stopping."""

from __future__ import annotations

import copy

import torch.nn as nn


class CompositeEarlyStopper:
    """Stops training when the composite score fails to improve for `patience` epochs."""

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0
        self.best_state_dict: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Update state. Returns True if training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best weights back into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
