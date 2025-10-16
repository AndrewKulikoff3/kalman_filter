from __future__ import annotations
import numpy as np

class ChiSquareGate:
    """
    Если d^2 = nu^T S^{-1} nu > threshold -> пропускаем update (skip).
    threshold: квантиль χ^2(m, p), где m = размерность измерения.
    """
    def __init__(self, threshold: float):
        self.th = float(threshold)

    def modify_innovation(self, nu: np.ndarray, S: np.ndarray) -> tuple[None | np.ndarray, bool]:
        r = np.linalg.solve(S, nu)
        d2 = float(nu.T @ r)
        if d2 > self.th:
            return None, False  # skip update
        return nu, True        # use as-is
