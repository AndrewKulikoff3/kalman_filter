from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from filterpy.kalman import rts_smoother  # fixed-interval RTS

Array = np.ndarray

@dataclass
class EMResult:
    Q: Array
    R: Array
    x0: Array
    P0: Array
    history: list[dict]

class EMEstimator:
    """
    Батч-EM для оценки Q,R (и при желании x0,P0) в LTI-модели:
      x_k = F x_{k-1} + w_k,  w~N(0,Q)
      z_k = H x_k       + v_k, v~N(0,R)
    Упрощённый M-шаг по Q: через 'process residuals' (без кросс-ковариаций).
    Это даёт хороший практичный старт; позже можно добавить точные формулы.
    """
    def __init__(self, F: Array, H: Array, Q_init: Array, R_init: Array,
                 x0: Array, P0: Array):
        self.F, self.H = F.copy(), H.copy()
        self.Q, self.R = Q_init.copy(), R_init.copy()
        self.x0, self.P0 = x0.copy(), P0.copy()

    def _kf_forward(self, Z: Array) -> tuple[list[Array], list[Array], KalmanFilter]:
        n, m = self.F.shape[0], self.H.shape[0]
        kf = KalmanFilter(dim_x=n, dim_z=m, dim_u=0)
        kf.F, kf.H, kf.Q, kf.R = self.F.copy(), self.H.copy(), self.Q.copy(), self.R.copy()
        kf.x, kf.P = self.x0.copy(), self.P0.copy()
        Xf, Pf = [], []
        for z in Z:
            kf.predict()
            kf.update(z)
            Xf.append(kf.x.copy())
            Pf.append(kf.P.copy())
        return Xf, Pf, kf

    def _smooth(self, Xf: list[Array], Pf: list[Array]) -> tuple[Array, Array]:
        # filterpy.rts_smoother принимает массивы (T,n) и (T,n,n)
        Xf_arr = np.asarray(Xf)         # (T,n)
        Pf_arr = np.asarray(Pf)         # (T,n,n)
        Xs, Ps, _, _ = rts_smoother(Xf_arr, Pf_arr, self.F, self.Q)
        return Xs, Ps  # (T,n), (T,n,n)

    def fit(self, Z: Array, n_iter: int = 10, clip_Q: float = 1e12, clip_R: float = 1e12) -> EMResult:
        """
        Z: (T, m)
        """
        history: list[dict] = []
        for _ in range(n_iter):
            # E-step: forward KF + RTS smoothing
            Xf, Pf, _ = self._kf_forward(Z)
            Xs, Ps = self._smooth(Xf, Pf)
            T = len(Z)

            # M-step (R): классическая формула через сглаженные x_k
            R_num = np.zeros_like(self.R)
            for k in range(T):
                zk = Z[k][:, None]            # (m,1)
                xk = Xs[k][:, None]           # (n,1)
                R_num += (zk - self.H @ xk) @ (zk - self.H @ xk).T + self.H @ Ps[k] @ self.H.T
            R_new = R_num / T

            # M-step (Q): практичная аппроксимация через 'process residuals'
            Q_num = np.zeros_like(self.Q)
            for k in range(1, T):
                xk   = Xs[k][:, None]
                xkm1 = Xs[k-1][:, None]
                wk = xk - self.F @ xkm1
                Q_num += wk @ wk.T + Ps[k] + self.F @ Ps[k-1] @ self.F.T  # без кросс-ковариаций
            Q_new = Q_num / (T - 1)

            # симметризация и клиппинг
            def psd(M):
                M = 0.5*(M+M.T)
                w, V = np.linalg.eigh(M)
                w = np.clip(w, 1e-12, clip_Q if M.shape==self.Q.shape else clip_R)
                return (V * w) @ V.T
            self.Q, self.R = psd(Q_new), psd(R_new)

            history.append({"Q": self.Q.copy(), "R": self.R.copy()})

            # (опционально можно обновлять x0,P0 по сглаженным первым моментам)
            self.x0 = Xs[0].copy()
            self.P0 = Ps[0].copy()

        return EMResult(Q=self.Q, R=self.R, x0=self.x0, P0=self.P0, history=history)
