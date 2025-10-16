from __future__ import annotations
import numpy as np

def fixed_point_const(Xf: np.ndarray, Pf: np.ndarray, F: np.ndarray, Q: np.ndarray, t_idx: int):
    """
    Fixed-Point smoothing для одного момента t_idx (константные F,Q).
    Возвращает (x_s, P_s) только для указанного k=t_idx.
    Реализовано как частичный RTS-бекпасс от конца до t_idx (без лишних аллокаций).
    """
    T, n = Xf.shape
    assert 0 <= t_idx < T

    # вперёд: нужные предсказанные ковариации (только до конца)
    P_pred = np.empty_like(Pf)
    P_pred[0] = Pf[0]
    for k in range(1, T):
        P_pred[k] = F @ Pf[k-1] @ F.T + Q

    # начнём с хвоста (на конце сглаженные = фильтрованные)
    x_s = Xf[-1].copy()
    P_s = Pf[-1].copy()

    # идём назад до t_idx
    Ft = F.T
    for k in range(T-2, t_idx-1, -1):
        Ck = Pf[k] @ Ft @ np.linalg.inv(P_pred[k+1])
        x_s = Xf[k] + Ck @ (x_s - F @ Xf[k])
        P_s = Pf[k] + Ck @ (P_s - P_pred[k+1]) @ Ck.T
        P_s = 0.5 * (P_s + P_s.T)

    return x_s, P_s

def fixed_point_var(Xf: np.ndarray, Pf: np.ndarray, F_list: list[np.ndarray], Q_list: list[np.ndarray], t_idx: int):
    """
    Fixed-Point smoothing для одного момента t_idx при покадровых F_k, Q_k.
    """
    T, n = Xf.shape
    assert 0 <= t_idx < T
    assert len(F_list) == T-1 and len(Q_list) == T-1

    P_pred = np.empty_like(Pf)
    P_pred[0] = Pf[0]
    for k in range(1, T):
        Fk, Qk = F_list[k-1], Q_list[k-1]
        P_pred[k] = Fk @ Pf[k-1] @ Fk.T + Qk

    x_s = Xf[-1].copy()
    P_s = Pf[-1].copy()

    for k in range(T-2, t_idx-1, -1):
        Fk = F_list[k]
        Ck = Pf[k] @ Fk.T @ np.linalg.inv(P_pred[k+1])
        x_s = Xf[k] + Ck @ (x_s - Fk @ Xf[k])
        P_s = Pf[k] + Ck @ (P_s - P_pred[k+1]) @ Ck.T
        P_s = 0.5 * (P_s + P_s.T)

    return x_s, P_s
