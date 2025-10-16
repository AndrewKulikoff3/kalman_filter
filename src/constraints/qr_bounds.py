from __future__ import annotations
import numpy as np

def _psd_clip(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = 0.5*(M+M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

class QRBounder:
    """
    Жёсткие пределы на элементы Q и R (матрицы PSD).
    Применяется каждый шаг, независимо от того, какие адаптации стоят.
    """
    def __init__(
        self,
        q_min: float = 1e-10, q_max: float = 1e6,
        r_min: float = 1e-10, r_max: float = 1e12,
        eps: float = 1e-12,
    ):
        self.q_min, self.q_max = float(q_min), float(q_max)
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.eps = float(eps)

    def apply(self, mean: np.ndarray, cov: np.ndarray, backend):
        # клип Q
        Q = np.clip(backend.Q, self.q_min, self.q_max)
        backend.set_Q(_psd_clip(Q, eps=self.eps))
        # клип R
        R = np.clip(backend.R, self.r_min, self.r_max)
        backend.set_R(_psd_clip(R, eps=self.eps))
        # состояние не меняем
        return mean, cov
