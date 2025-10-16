from __future__ import annotations
import numpy as np

class HuberRobust:
    """
    Мягкое подавление: w = 1 при r<=c, иначе w = c/r, где r = sqrt(nu^T S^{-1} nu)
    """
    def __init__(self, c: float = 2.5):
        self.c = float(c)

    def modify_innovation(self, nu: np.ndarray, S: np.ndarray):
        r = np.sqrt(max(float(nu.T @ np.linalg.solve(S, nu)), 1e-12))
        w = 1.0 if r <= self.c else self.c / r
        return w * nu, True
