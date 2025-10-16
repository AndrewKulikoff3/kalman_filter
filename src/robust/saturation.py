from __future__ import annotations
import numpy as np

class Saturation:
    """
    "Жёсткое" ограничение инновации по Махаланобис-радиусу:
      nu' = min(1, c / r) * nu
    """
    def __init__(self, c: float = 3.0):
        self.c = float(c)

    def modify_innovation(self, nu: np.ndarray, S: np.ndarray):
        r = np.sqrt(max(float(nu.T @ np.linalg.solve(S, nu)), 1e-12))
        scale = 1.0 if r <= self.c else self.c / r
        return scale * nu, True
