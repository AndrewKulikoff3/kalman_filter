from __future__ import annotations
import numpy as np

def rts_smooth(xs: np.ndarray, Ps: np.ndarray,
               Fs: np.ndarray, Qs: np.ndarray,
               lag: int) -> np.ndarray:
    """
    Fixed-lag RTS smoother.

    This function performs a local RTS backwards pass over a window
    of size `lag` for each time index i. In other words, to smooth
    estimate at index i it uses information only up to index i+lag.
    When lag=0, no smoothing is applied (the output equals xs).

    Parameters
    ----------
    xs : np.ndarray
        Filtered state estimates, shape (N, dim_x).
    Ps : np.ndarray
        Filtered covariance matrices, shape (N, dim_x, dim_x).
    Fs : np.ndarray
        State transition matrices, shape (N, dim_x, dim_x).
        Fs[k] propagates state from step k to k+1.
    Qs : np.ndarray
        Process noise covariance matrices, shape (N, dim_x, dim_x).
        Qs[k] is the process noise used at step k.
    lag : int
        Fixed lag (>=0). For each i, information from steps up to i+lag
        will be used to refine x[i].

    Returns
    -------
    np.ndarray
        Smoothed state estimates of shape (N, dim_x).
    """
    N, dim_x = xs.shape
    # Copies to hold smoothed states and covariances
    x_out = xs.copy()
    P_out = Ps.copy()

    # Iterate backwards over all time indices except the last one
    for i in range(N - 2, -1, -1):
        # Determine the furthest index we are allowed to look ahead
        j_max = min(N - 1, i + lag)

        # Initialize the forward value in the smoothing window:
        # start from the filtered estimate at j_max
        x_curr = xs[j_max].copy()
        P_curr = Ps[j_max].copy()

        # Now walk backwards from j_max - 1 down to i
        for k in range(j_max - 1, i - 1, -1):
            # Predict covariance from k to k+1
            Fkp1 = Fs[k + 1]
            Qkp1 = Qs[k + 1]
            P_pred = Fkp1 @ Ps[k] @ Fkp1.T + Qkp1

            # Smoothing gain
            K = Ps[k] @ Fkp1.T @ np.linalg.inv(P_pred)

            # RTS backward updates
            x_sm_k = xs[k] + K @ (x_curr - Fkp1 @ xs[k])
            P_sm_k = Ps[k] + K @ (P_curr - P_pred) @ K.T

            # Update "current" smoothed estimate for the next iteration
            x_curr, P_curr = x_sm_k, P_sm_k

        # Assign the smoothed value at position i
        x_out[i] = x_curr
        P_out[i] = P_curr

    return x_out
