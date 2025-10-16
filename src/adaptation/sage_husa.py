from __future__ import annotations
import numpy as np

def _psd_clip(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

class SageHusaR:
    """
    Онлайн-обновление R (шум измерений) с забыванием λ:
      R_k = (1-λ) R_{k-1} + λ * (ν_k ν_k^T - H P_k H^T)
    Формула классическая, но на практике:
      - клип/PSD-очистка обязательны,
      - λ ~ 0.01..0.1 в зависимости от динамики изменений шума.
    """
    def __init__(self, lam: float = 0.05, r_min: float = 1e-8, r_max: float = 1e12):
        self.lam = float(lam)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.last_H = None
        self.last_P = None
        self.last_nu = None
        self.R_inited = False

    def on_innovation(self, nu, S, H, P):
        self.last_H, self.last_P, self.last_nu = H, P, nu.copy()

    def adjust_backend(self, backend):
        if any(v is None for v in (self.last_H, self.last_P, self.last_nu)):
            return
        H, P, nu = self.last_H, self.last_P, self.last_nu

        R_emp = np.outer(nu, nu) - H @ P @ H.T
        R_emp = _psd_clip(R_emp, eps=self.r_min)

        if not self.R_inited:
            R_new = R_emp  # можно смешать с backend.R, если хочешь плавнее
            self.R_inited = True
        else:
            R_new = (1.0 - self.lam) * backend.R + self.lam * R_emp

        # клип по масштабу
        R_new = np.clip(R_new, self.r_min, self.r_max)
        R_new = _psd_clip(R_new, eps=self.r_min)
        backend.set_R(R_new)