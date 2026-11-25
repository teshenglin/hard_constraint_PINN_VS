import numpy as np
import torch
from typing import Tuple, Optional, Union, Sequence

try:
    # pyDOE / pyDOE2 style
    from pyDOE import lhs  # type: ignore
except Exception:
    from pyDOE2 import lhs  # type: ignore


def sample_periodic_training_points(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    t_min: float, t_max: float,
    N_interior: int,
    N_initial: int,
    N_boundary: int,
    seed: Optional[int] = None,
    dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate collocation points for a 2D periodic domain in space and time interval [t_min, t_max].

    Returns:
        X_inner:  (N_interior, 3) interior points in (x,y,t)
        X_initial:(N_initial, 3) initial points with t=t_min
        X0_bd,X1_bd: (N_boundary, 3) paired periodic points on x=x_min and x=x_max sharing (y,t)
        Y0_bd,Y1_bd: (N_boundary, 3) paired periodic points on y=y_min and y=y_max sharing (x,t)
    """

    # ---- interior points: LHS in (x,y,t)
    # pyDOE lhs uses global RNG; to keep determinism, we generate our own uniform and "lhs" via pyDOE.
    # Practical compromise: set np.random.seed if you need strict repeatability across runs.
    if seed is not None:
        np.random.seed(seed)

    sampling = lhs(3, N_interior)  # (N_interior, 3) in [0,1]^3

    x_in = x_min + (x_max - x_min) * sampling[:, 0:1]
    y_in = y_min + (y_max - y_min) * sampling[:, 1:2]
    t_in = t_min + (t_max - t_min) * sampling[:, 2:3]
    X_inner = np.hstack((x_in, y_in, t_in))

    # ---- initial points: (x,y) LHS, t=t_min
    x0 = x_min + (x_max - x_min) * lhs(1, N_initial)
    y0 = y_min + (y_max - y_min) * lhs(1, N_initial)
    t0 = np.full((N_initial, 1), t_min)
    X_initial = np.hstack((x0, y0, t0))

    # ---- periodic x-faces: share (y,t)
    y_shared = y_min + (y_max - y_min) * lhs(1, N_boundary)
    t_shared_x = t_min + (t_max - t_min) * lhs(1, N_boundary)

    X0_bd = np.hstack((np.full((N_boundary, 1), x_min), y_shared, t_shared_x))
    X1_bd = np.hstack((np.full((N_boundary, 1), x_max), y_shared, t_shared_x))

    # ---- periodic y-faces: share (x,t)
    x_shared = x_min + (x_max - x_min) * lhs(1, N_boundary)
    t_shared_y = t_min + (t_max - t_min) * lhs(1, N_boundary)

    Y0_bd = np.hstack((x_shared, np.full((N_boundary, 1), y_min), t_shared_y))
    Y1_bd = np.hstack((x_shared, np.full((N_boundary, 1), y_max), t_shared_y))

    # ---- cast dtype
    X_inner = X_inner.astype(dtype, copy=False)
    X_initial = X_initial.astype(dtype, copy=False)
    X0_bd = X0_bd.astype(dtype, copy=False)
    X1_bd = X1_bd.astype(dtype, copy=False)
    Y0_bd = Y0_bd.astype(dtype, copy=False)
    Y1_bd = Y1_bd.astype(dtype, copy=False)

    total = N_interior + N_initial + 4 * N_boundary
    print(f"Total number of training points: {total}")

    return X_inner, X_initial, X0_bd, X1_bd, Y0_bd, Y1_bd

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
    
