from __future__ import annotations
from typing import Optional, Protocol, Tuple, Iterable, Union, List
import numpy as np

class AdaptationPolicy(Protocol):
    def on_innovation(self, nu: np.ndarray, S: np.ndarray, H: np.ndarray, P: np.ndarray) -> None: ...
    def adjust_backend(self, backend) -> None: ...

class RobustPolicy(Protocol):
    # Возвращает (исправленная_инновация, использовать_измерение?)
    def modify_innovation(self, nu: np.ndarray, S: np.ndarray) -> tuple[Optional[np.ndarray], bool]: ...

class ConstraintPolicy(Protocol):
    # Возвращает скорректированные (mean, cov). Может менять Q/R внутри backend по желанию.
    def apply(self, mean: np.ndarray, cov: np.ndarray, backend) -> tuple[np.ndarray, np.ndarray]: ...

def _as_iter(x: Union[AdaptationPolicy, Iterable[AdaptationPolicy], None]) -> Iterable[AdaptationPolicy]:
    if x is None: return ()
    if isinstance(x, Iterable) and not isinstance(x, (bytes, str)):
        return x
    return (x,)  # одиночный -> кортеж

def _set_state(backend, x: np.ndarray, P: np.ndarray) -> None:
    # поддержка линейного KF (backend.kf), EKF (backend.ekf), UKF (backend.ukf)
    if hasattr(backend, "kf"):
        backend.kf.x = x.copy()
        backend.kf.P = 0.5*(P+P.T)
    elif hasattr(backend, "ekf"):
        backend.ekf.x = x.copy()
        backend.ekf.P = 0.5*(P+P.T)
    elif hasattr(backend, "ukf"):
        backend.ukf.x = x.copy()
        backend.ukf.P = 0.5*(P+P.T)
    else:
        raise RuntimeError("Unknown backend type: cannot set state")


class PluggableKF:
    """
    Унифицированный «хост»:
      - backend предоставляет: predict(u), innovate(z)->(nu,S,H,z_pred), update_with_innovation(nu')
      - robust: список политик (применяются по очереди; любая может "завернуть" измерение)
      - adaptation: список политик (видят статистику и затем могут крутить R/Q у backend)
      - constraints: список политик (проекция mean/cov; ограничители Q/R)
    """
    def __init__(
        self,
        backend,
        adaptation: Union[AdaptationPolicy, Iterable[AdaptationPolicy], None] = None,
        robust: Union[RobustPolicy, Iterable[RobustPolicy], None] = None,
        constraints: Union[ConstraintPolicy, Iterable[ConstraintPolicy], None] = None,
    ):
        self.b = backend
        self.adapt_all: List[AdaptationPolicy] = list(_as_iter(adaptation))
        self.robust_all: List[RobustPolicy]   = list(_as_iter(robust))
        self.cons_all:  List[ConstraintPolicy]= list(_as_iter(constraints))

    def step(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # 1) predict
        self.b.predict(u=u)
        # 2) innovate
        nu, S, H, _ = self.b.innovate(z)

        # 3) адаптации: видят nu,S,H,P
        _, P = self.b.state()
        for ad in self.adapt_all:
            ad.on_innovation(nu, S, H, P)

        # 4) робастная цепочка: последовательно модифицируем инновацию
        nu_prime: Optional[np.ndarray] = nu
        use = True
        for rb in self.robust_all:
            if not use: break
            nu_prime, use = rb.modify_innovation(nu_prime, S)  # type: ignore
        if not use or nu_prime is None:
            # пропускаем update, но всё равно позволяем адаптациям применить изменения
            for ad in self.adapt_all:
                ad.adjust_backend(self.b)
            return self.b.state()

        # 5) стандартный update
        self.b.update_with_innovation(nu_prime)

        # 6) constraints — после апдейта проецируем mean/cov
        x, P = self.b.state()
        for cs in self.cons_all:
            x, P = cs.apply(x, P, self.b)
        _set_state(self.b, x, P)

        # 7) применяем изменения у адаптаций (Q/R)
        for ad in self.adapt_all:
            ad.adjust_backend(self.b)

        return self.b.state()