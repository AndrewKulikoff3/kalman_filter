from __future__ import annotations
import numpy as np

class NormBallProjector:
    """
    Ограничение нормы по индексу компонент: || x[idx] || ≤ r.
    Проецирует только выбранный подвектор.
    """
    def __init__(self, idx: list[int], r: float):
        self.idx = list(idx)
        self.r = float(r)

    def apply(self, mean: np.ndarray, cov: np.ndarray, backend):
        sub = mean[self.idx]
        norm = float(np.linalg.norm(sub))
        if norm > self.r and norm > 0:
            mean2 = mean.copy()
            mean2[self.idx] = (self.r / norm) * sub
            return mean2, cov
        return mean, cov
