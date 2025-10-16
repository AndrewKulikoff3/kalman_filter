from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def _log_likelihood_gaussian(nu: np.ndarray, S: np.ndarray) -> float:
    m = nu.shape[0]
    # ln N(nu; 0, S) = -0.5*(m ln(2π) + ln|S| + d^2)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        # деградация ковариации — защитный ход
        logdet = np.log(np.linalg.det(S + 1e-9*np.eye(m)))
    d2 = float(nu.T @ np.linalg.solve(S, nu))
    return -0.5 * (m*np.log(2*np.pi) + logdet + d2)

@dataclass
class ModelProfile:
    Q: np.ndarray
    R: np.ndarray
    name: str = ""

class MMAE:
    """
    Multiple-Model Adaptive Estimation (упрощённый):
      - есть K профилей (Q_i, R_i)
      - на каждом шаге считаем likelihood измерения при текущей инновации (nu,S_i)
      - обновляем веса (байесовски), нормализуем
      - стратегия выбора: argmax или усреднение параметров
    """
    def __init__(self,
                 profiles: list[ModelProfile],
                 select: str = "argmax",    # "argmax" | "weighted"
                 forgetting: float = 0.0):  # 0..1, экспоненциальный "prior" к прошлым весам
        assert select in ("argmax", "weighted")
        self.profiles = profiles
        self.weights = np.ones(len(profiles)) / len(profiles)
        self.select = select
        self.forgetting = float(forgetting)
        self._last_nu = None

    # интеграция с PluggableKF (адаптация-политика)
    def on_innovation(self, nu: np.ndarray, S: np.ndarray, H: np.ndarray | None, P: np.ndarray) -> None:
        self._last_nu = (nu.copy(), S.copy())

    def adjust_backend(self, backend) -> None:
        if self._last_nu is None:
            return
        nu, _ = self._last_nu

        # обновляем веса по правдоподобию для К профилей
        logw = np.log(np.clip(self.weights, 1e-15, 1.0))
        new_logw = np.zeros_like(logw)
        for i, prof in enumerate(self.profiles):
            # S_i = H P H^T + R_i  (линейный KF; H берём из backend)
            H = backend.H
            S_i = H @ backend.kf.P @ H.T + prof.R
            ll = _log_likelihood_gaussian(nu, S_i)
            new_logw[i] = (1.0 - self.forgetting) * ll + self.forgetting * logw[i]

        # нормализация в log-сумме
        maxl = np.max(new_logw)
        w = np.exp(new_logw - maxl)
        self.weights = w / np.sum(w)

        # выбор параметров
        if self.select == "argmax":
            j = int(np.argmax(self.weights))
            backend.set_Q(self.profiles[j].Q)
            backend.set_R(self.profiles[j].R)
        else:  # weighted
            Qw = sum(wi * pi.Q for wi, pi in zip(self.weights, self.profiles))
            Rw = sum(wi * pi.R for wi, pi in zip(self.weights, self.profiles))
            backend.set_Q(Qw)
            backend.set_R(Rw)
