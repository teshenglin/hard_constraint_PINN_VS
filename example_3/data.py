import numpy as np
import torch
from typing import Tuple, Optional, Union, Sequence

def latin_hypercube(dimensions: int, count: int, rng: np.random.Generator):
    """Draw one point in each stratum of every coordinate."""
    if count < 0:
        raise ValueError("LHS point count must be nonnegative")
    if count == 0:
        return np.empty((0, dimensions))
    strata = np.column_stack([rng.permutation(count) for _ in range(dimensions)])
    return (strata + rng.random((count, dimensions))) / count


def _log_viscosity_with_endpoints(unit_samples, nu_min, nu_max, *, uniform_nu):
    """Map LHS strata to log viscosity and pin the two endpoints."""
    eta_min, eta_max = np.log([nu_min, nu_max])
    if uniform_nu:
        nu = nu_min + (nu_max - nu_min) * unit_samples
        eta = np.log(nu)
    else:
        eta = eta_min + (eta_max - eta_min) * unit_samples
    eta[unit_samples.argmin()] = eta_min
    eta[unit_samples.argmax()] = eta_max
    return eta


def sample_periodic_training_points_nu(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    t_min: float, t_max: float,
    nu_min: float, nu_max: float,
    N_interior: int,
    N_initial: int,
    seed: Optional[int] = None,
    dtype=np.float64,
    *,
    uniform_nu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate periodic collocation points including both viscosity endpoints in each set.

    With uniform_nu=True, stratify in physical viscosity before returning log(nu).

    Returns:
        X_inner:  (N_interior, 4) interior points in (x,y,t,log(nu))
        X_initial:(N_initial, 4) initial points with t=t_min
    """

    if min(N_interior, N_initial) < 2:
        raise ValueError("Each training set needs at least two points for the viscosity endpoints")

    rng = np.random.default_rng(seed)
    sampling = latin_hypercube(4, N_interior, rng)

    x_in = x_min + (x_max - x_min) * sampling[:, 0:1]
    y_in = y_min + (y_max - y_min) * sampling[:, 1:2]
    t_in = t_min + (t_max - t_min) * sampling[:, 2:3]
    eta = _log_viscosity_with_endpoints(
        sampling[:, 3:4], nu_min, nu_max, uniform_nu=uniform_nu,
    )
    X_inner = np.hstack((x_in, y_in, t_in, eta))

    # ---- initial points: (x,y,log(nu)) LHS, t=t_min
    x0 = x_min + (x_max - x_min) * latin_hypercube(1, N_initial, rng)
    y0 = y_min + (y_max - y_min) * latin_hypercube(1, N_initial, rng)
    t0 = np.full((N_initial, 1), t_min)
    eta = _log_viscosity_with_endpoints(
        latin_hypercube(1, N_initial, rng), nu_min, nu_max, uniform_nu=uniform_nu,
    )
    X_initial = np.hstack((x0, y0, t0, eta))

    # ---- cast dtype
    X_inner = X_inner.astype(dtype, copy=False)
    X_initial = X_initial.astype(dtype, copy=False)

    total = N_interior + N_initial
    print(f"Total number of training points: {total}")

    return X_inner, X_initial

def sample_gauge_points_nu(
    x_min, x_max, y_min, y_max, t_min, t_max, nu_min, nu_max,
    N_times: int, N_viscosities: int, N_space: int, seed: int | None = None,
):
    """Reuse a spatial LHS design at each fixed time and log viscosity."""
    if min(N_times, N_viscosities, N_space) < 1:
        raise ValueError("Gauge time, viscosity, and spatial point counts must be positive")

    rng = np.random.default_rng(seed)
    xy = latin_hypercube(2, N_space, rng)
    xy = np.array([x_min, y_min]) + xy * np.array([x_max - x_min, y_max - y_min])
    times = np.linspace(t_min, t_max, N_times)
    eta = np.linspace(np.log(nu_min), np.log(nu_max), N_viscosities)
    points = np.empty((N_times, N_viscosities, N_space, 4))
    points[:, :, :, :2] = xy
    points[:, :, :, 2] = times[:, None, None]
    points[:, :, :, 3] = eta[None, :, None]
    return points.reshape(N_times * N_viscosities, N_space, 4)


def to_torch(*arrays, device=None, dtype=torch.float64, requires_grad: Union[bool, Sequence[bool]] = True):
    import torch

    if isinstance(requires_grad, (list, tuple)):
        assert len(requires_grad) == len(arrays)
        flags = requires_grad
    else:
        flags = [requires_grad] * len(arrays)

    tensors = []
    for a, rg in zip(arrays, flags):
        t = torch.as_tensor(a, device=device, dtype=dtype)
        t.requires_grad_(bool(rg))
        tensors.append(t)
    return tensors
    
