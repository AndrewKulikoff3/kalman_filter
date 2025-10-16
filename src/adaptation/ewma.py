from __future__ import annotations
import numpy as np

def _psd_clip(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    M = 0.5*(M+M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

class EWMAAdaptive:
    """
    EWMA адаптация дисперсий для R и/или Q.
    - R: R <- (1-α) R + α * (νν^T), опционально игнорируя выбросы по χ²-гейтингу.
    - Q: Q <- (1-α) Q + α * (w w^T), где w = x_k - F x_{k-1} (линейный KF).
    Предохранители: psd-клиппинг + пределы (min/max).
    """
    def __init__(
        self,
        mode: str = "r",                 # 'r' | 'q' | 'both'
        alpha: float = 0.05,
        chi2_gate: float | None = None,  # если задано, R не обновляем при d^2 > gate
        accel_idx: list[int] | None = None,  # если хотим считать Q по подмножеству компонент (например, аксель)
    ):
        assert mode in ("r","q","both")
        self.mode = mode
        self.alpha = float(alpha)
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.q_min, self.q_max = float(q_min), float(q_max)
        self.chi2_gate = chi2_gate
        self.prev_x = None
        self.accel_idx = accel_idx

    def on_innovation(self, nu: np.ndarray, S: np.ndarray, H: np.ndarray, P: np.ndarray) -> None:
        # ничего — всё делаем в adjust_backend, чтобы иметь доступ к backend.{R,Q} и F,x
        self._last = (nu.copy(), S.copy())

    def adjust_backend(self, backend) -> None:
        nu, S = getattr(self, "_last", (None, None))
        # --- R: EWMA по инновациям, с гейтингом ---
        if self.mode in ("r","both") and nu is not None:
            if self.chi2_gate is not None:
                d2 = float(nu.T @ np.linalg.solve(S, nu))
                if d2 <= self.chi2_gate:
                    R_emp = np.outer(nu, nu)
                    R_new = (1.0 - self.alpha) * backend.R + self.alpha * R_emp
                    R_new = np.clip(R_new, self.r_min, self.r_max)
                    backend.set_R(_psd_clip(R_new, eps=self.r_min))
            else:
                R_emp = np.outer(nu, nu)
                R_new = (1.0 - self.alpha) * backend.R + self.alpha * R_emp
                R_new = np.clip(R_new, self.r_min, self.r_max)
                backend.set_R(_psd_clip(R_new, eps=self.r_min))

        # --- Q: EWMA по процессным остаткам (только линейный KF: нужен F и x) ---
        if self.mode in ("q","both") and hasattr(backend, "kf"):
            F = backend.kf.F
            x_cur = backend.kf.x.copy()
            if self.prev_x is not None:
                w = x_cur - F @ self.prev_x
                if self.accel_idx is not None:
                    w_sub = w[self.accel_idx]
                    wwT = np.zeros_like(backend.Q)
                    wwT[np.ix_(self.accel_idx, self.accel_idx)] = np.outer(w_sub, w_sub)
                else:
                    wwT = np.outer(w, w)
                Q_new = (1.0 - self.alpha) * backend.Q + self.alpha * wwT
                Q_new = np.clip(Q_new, self.q_min, self.q_max)
                backend.set_Q(_psd_clip(Q_new, eps=self.q_min))
            self.prev_x = x_cur.copy()
