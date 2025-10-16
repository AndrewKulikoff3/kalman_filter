from __future__ import annotations
import numpy as np

class StudentTRobust:
    """
    Аппроксимация t-правдоподобия via веса:
      w_t = sqrt((ν + m) / (ν + d^2)), где d^2 = nu^T S^{-1} nu, m=dim(z).
      nu' = w_t * nu
    """
    def __init__(self, dof: float = 5.0):
        assert dof > 0
        self.v = float(dof)

    def modify_innovation(self, nu: np.ndarray, S: np.ndarray):
        m = nu.shape[0]
        d2 = float(nu.T @ np.linalg.solve(S, nu))
        w = np.sqrt((self.v + m) / (self.v + max(d2, 1e-12)))
        return w * nu, True
