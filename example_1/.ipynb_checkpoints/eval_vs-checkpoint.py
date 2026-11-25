import numpy as np
import torch
from torch.func import vmap, grad

try:
    from pyDOE import lhs  # type: ignore
except Exception:
    from pyDOE2 import lhs  # type: ignore

def sample_interior_lhs(
    x_min, x_max, y_min, y_max, t_min, t_max,
    N: int,
    t_fixed: float | None = None,
    seed: int | None = None,
    dtype=np.float64,
):
    """
    LHS sampling for evaluation in (x,y,t).
    If t_fixed is not None, sample only (x,y) and set t=t_fixed.
    Returns X_np shape (N,3) and also x,y,t column arrays for exact soln.
    """
    if seed is not None:
        np.random.seed(seed)

    if t_fixed is None:
        X_unit = lhs(3, N)
        x = x_min + (x_max - x_min) * X_unit[:, 0:1]
        y = y_min + (y_max - y_min) * X_unit[:, 1:2]
        t = t_min + (t_max - t_min) * X_unit[:, 2:3]
    else:
        X_unit = lhs(2, N)
        x = x_min + (x_max - x_min) * X_unit[:, 0:1]
        y = y_min + (y_max - y_min) * X_unit[:, 1:2]
        t = np.full((N, 1), t_fixed)

    X = np.hstack([x, y, t]).astype(dtype, copy=False)
    return X, x.astype(dtype), y.astype(dtype), t.astype(dtype)


def rel_l2(a, b, eps=1e-30):
    num = np.linalg.norm(a - b, 2)
    den = np.linalg.norm(b, 2)
    return num / (den + eps)


def rel_linf(a, b, eps=1e-30):
    num = np.linalg.norm(a - b, np.inf)
    den = np.linalg.norm(b, np.inf)
    return num / (den + eps)


def rel_l2_vec(ax, ay, bx, by, eps=1e-30):
    num = np.sqrt(np.linalg.norm(ax - bx, 2)**2 + np.linalg.norm(ay - by, 2)**2)
    den = np.sqrt(np.linalg.norm(bx, 2) ** 2 + np.linalg.norm(by, 2) ** 2)
    return num / (den + eps)


def rel_linf_vec(ax, ay, bx, by, eps=1e-30):
    num = np.max(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
    den = np.max(np.sqrt(bx ** 2 + by ** 2))
    return num / (den + eps)


@torch.no_grad()
def predict_omega_psi(func_model, params, X_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    X_t: (N,3) torch tensor on device
    returns omega_pred, psi_pred as numpy arrays (N,1)
    """
    pred = vmap(func_model, in_dims=(None, 0))(params, X_t)
    omega = pred[:, 0:1].detach().cpu().numpy()
    psi = pred[:, 1:2].detach().cpu().numpy()
    return omega, psi

def predict_gradpsi(func_model, params, X_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂x psi and ∂y psi using autodiff (expensive).
    X_t must require_grad=True.
    returns psix_pred, psiy_pred as numpy arrays (N,1)
    """
    def psi_scalar(p, x):
        return func_model(p, x)[1].reshape(())

    gpsi_fn = grad(psi_scalar, argnums=1)
    gpsi = vmap(gpsi_fn, in_dims=(None, 0))(params, X_t)  # (N,3)
    psix = gpsi[:, 0:1].detach().cpu().numpy()
    psiy = gpsi[:, 1:2].detach().cpu().numpy()
    return psix, psiy

def evaluate_errors(
    *,
    func_model,
    params,
    device,
    dtype_torch,
    # domain
    x_min, x_max,
    y_min, y_max,
    t_min, t_max,
    # NEW: physics parameter
    nu: float,
    # sampling
    N_val: int = 100000,
    N_grad: int = 20000,
    t_fixed: float | None = None,
    seed: int | None = None,
    # exact functions (NumPy)
    exact_omega_np=None,
    exact_psi_np=None,
    exact_psix_np=None,
    exact_psiy_np=None,
):
    """
    Returns:
        error_w_2r, error_w_infr, error_gradpsi_2r, error_gradpsi_infr
    """
    if dtype_torch == torch.float64:
        dtype_np = np.float64
    elif dtype_torch == torch.float32:
        dtype_np = np.float32
    else:
        raise ValueError(f"Unsupported dtype_torch: {dtype_torch}")
    
    assert exact_omega_np is not None
    assert exact_psix_np is not None and exact_psiy_np is not None

    # --- value errors (omega)
    X_np, x_np, y_np, t_np = sample_interior_lhs(
        x_min, x_max, y_min, y_max, t_min, t_max,
        N=N_val, t_fixed=t_fixed, seed=seed, dtype=dtype_np
    )
    X_t = torch.as_tensor(X_np, device=device, dtype=dtype_torch)
    # values do NOT need input grads
    omega_pred, _ = predict_omega_psi(func_model, params, X_t)
    omega_ex = exact_omega_np(x_np, y_np, t_np, nu=nu)

    error_w_2r = rel_l2(omega_pred, omega_ex)
    error_w_infr = rel_linf(omega_pred, omega_ex)

    # --- gradient errors (grad psi) — use smaller N_grad for speed
    Xg_np, xg, yg, tg = sample_interior_lhs(
        x_min, x_max, y_min, y_max, t_min, t_max,
        N=N_grad, t_fixed=t_fixed, seed=None if seed is None else seed + 12345, dtype=dtype_np
    )
    Xg_t = torch.as_tensor(Xg_np, device=device, dtype=dtype_torch)
    Xg_t.requires_grad_(True)

    psix_pred, psiy_pred = predict_gradpsi(func_model, params, Xg_t)
    psix_ex  = exact_psix_np(xg, yg, tg, nu=nu)
    psiy_ex  = exact_psiy_np(xg, yg, tg, nu=nu)

    error_gradpsi_2r = rel_l2_vec(psix_pred, psiy_pred, psix_ex, psiy_ex)
    error_gradpsi_infr = rel_linf_vec(psix_pred, psiy_pred, psix_ex, psiy_ex)

    return error_w_2r, error_w_infr, error_gradpsi_2r, error_gradpsi_infr
