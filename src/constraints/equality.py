from __future__ import annotations
import numpy as np

class EqualityProjector:
    """
    Жёсткая проекция апостериора на множество {x : A x = b}.
    Формулы ККТ:
      Kc = P A^T (A P A^T)^{-1}
      x' = x - Kc (A x - b)
      P' = P - Kc A P
    """
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

    def apply(self, mean: np.ndarray, cov: np.ndarray, backend):
        A, b = self.A, self.b
        APAT = A @ cov @ A.T
        # численно устойчиво — через pinv (или cho_solve при PSD)
        Kc = cov @ A.T @ np.linalg.pinv(APAT)
        m2 = mean - Kc @ (A @ mean - b)
        P2 = cov - Kc @ A @ cov
        P2 = 0.5*(P2 + P2.T)
        return m2, P2
