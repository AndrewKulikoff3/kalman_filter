from __future__ import annotations
import numpy as np

def _psd_clip(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

class InnovationAdaptiveR:
    """
    Innovation-based Adaptive Estimation (IAE) для матрицы R.
    Идея: S = H P H^T + R; оценим эмпирическую S и вычислим R ≈ S_emp - H P H^T.
    S_ema — экспоненциальное скользящее среднее для устойчивости.
    """
    def __init__(self, alpha: float = 0.05, r_min: float = 1e-6, r_max_scale: float = 1e6):
        self.alpha = float(alpha)
        self.S_ema = None
        self.last_H = None
        self.last_P = None
        self.r_min = r_min
        self.r_max_scale = r_max_scale

    def on_innovation(self, nu: np.ndarray, S: np.ndarray, H: np.ndarray, P: np.ndarray) -> None:
        # экспоненциальное сглаживание ковариации инноваций
        if self.S_ema is None:
            self.S_ema = S.copy()
        else:
            self.S_ema = (1.0 - self.alpha) * self.S_ema + self.alpha * S
        self.last_H = H
        self.last_P = P

    def adjust_backend(self, backend) -> None:
        if self.S_ema is None or self.last_H is None or self.last_P is None:
            return
        H, P = self.last_H, self.last_P
        R_hat = self.S_ema - H @ P @ H.T
        R_hat = _psd_clip(R_hat, eps=self.r_min)

        # ограничим «раздувание» R (защита от выбросов/нестабильности)
        base = backend.R
        try:
            # в простейшем виде — элементно
            max_allowed = np.maximum(self.r_min, np.minimum(np.abs(base) * self.r_max_scale, 1e12))
            R_hat = np.clip(R_hat, self.r_min, max_allowed)
        except Exception:
            pass

        backend.set_R(R_hat)
