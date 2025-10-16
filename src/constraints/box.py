from __future__ import annotations
import numpy as np

class BoxProjector:
    """
    Боксовые границы: lb ≤ x ≤ ub.
    По умолчанию — лёгкая версия: клипим mean, ковариацию оставляем.
    (при желании можно добавить 'mode="truncmoments"' позже)
    """
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        self.lb = lb
        self.ub = ub

    def apply(self, mean: np.ndarray, cov: np.ndarray, backend):
        m2 = np.clip(mean, self.lb, self.ub)
        return m2, cov
