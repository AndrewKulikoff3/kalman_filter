from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    from filterpy.kalman import KalmanFilter as _FKF
except ImportError as e:
    raise RuntimeError("pip install filterpy") from e


class FilterPyBackend:
    """
    Тонкая обёртка над filterpy.KalmanFilter для линейного случая.
    Держит матрицы модели и состояние; предоставляет шаги predict/innovate/update.
    """
    def __init__(
        self,
        F: np.ndarray, Q: np.ndarray,
        H: np.ndarray, R: np.ndarray,
        x0: np.ndarray, P0: np.ndarray,
        B: Optional[np.ndarray] = None,
    ):
        n = F.shape[0]
        m = H.shape[0]
        dim_u = 0 if B is None else B.shape[1]

        kf = _FKF(dim_x=n, dim_z=m, dim_u=dim_u)
        kf.F, kf.Q, kf.H, kf.R = F.copy(), Q.copy(), H.copy(), R.copy()
        kf.x, kf.P = x0.copy(), P0.copy()
        kf.B = None if B is None else B.copy()
        self.kf = kf

    # --- модель/параметры ---
    @property
    def F(self): return self.kf.F
    @property
    def Q(self): return self.kf.Q
    @property
    def H(self): return self.kf.H
    @property
    def R(self): return self.kf.R

    def set_F(self, F: np.ndarray): self.kf.F = F.copy()
    def set_Q(self, Q: np.ndarray): self.kf.Q = Q.copy()
    def set_H(self, H: np.ndarray): self.kf.H = H.copy()
    def set_R(self, R: np.ndarray): self.kf.R = R.copy()

    # --- шаги ---
    def predict(self, u: Optional[np.ndarray] = None):
        self.kf.predict(u=u)

    def innovate(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает (nu, S, K) для использования политиками.
        """
        H, R = self.kf.H, self.kf.R
        x, P = self.kf.x, self.kf.P
        z_pred = H @ x
        nu = z - z_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        return nu, S, K

    def update_with_gain(self, K: np.ndarray, nu: np.ndarray, joseph: bool = True):
        x, P, H, R = self.kf.x, self.kf.P, self.kf.H, self.kf.R
        self.kf.x = x + K @ nu
        I = np.eye(P.shape[0])
        if joseph:
            Pn = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        else:
            Pn = (I - K @ H) @ P
        self.kf.P = 0.5 * (Pn + Pn.T)  # симметризация

    # --- доступ к состоянию ---
    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kf.x.copy(), self.kf.P.copy()


### TODO: добавить EKF, UKF backend