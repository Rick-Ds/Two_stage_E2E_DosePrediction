# -*- coding: utf-8 -*-
"""
train_utils.py

Lightweight training utilities.

Currently provides:
- EarlyStopping: monitors a validation metric (lower is better) and triggers early stop after a
  patience window without sufficient improvement (min_delta).

Author: Boda Ning
"""


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0

    def step(self, val: float) -> bool:
        improved = (self.best - val) > self.min_delta
        if improved:
            self.best = val
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience


