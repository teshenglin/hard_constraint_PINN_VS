import numpy as np
import torch
from torch.func import vmap, grad

from data import latin_hypercube

def sample_interior_lhs(
    x_min, x_max, y_min, y_max, t_min, t_max, nu_min, nu_max, 
    N: int,
    t_fixed: float | None = None,
    seed: int | None = None,
    dtype=np.float64,
):
    """Sample space, time, and log viscosity, or use one fixed time."""
    rng = np.random.default_rng(seed)
    X_unit = latin_hypercube(4 if t_fixed is None else 3, N, rng)
    x = x_min + (x_max - x_min) * X_unit[:, 0:1]
    y = y_min + (y_max - y_min) * X_unit[:, 1:2]
    if t_fixed is None:
        t = t_min + (t_max - t_min) * X_unit[:, 2:3]
    else:
        t = np.full((N, 1), t_fixed)
    eta = np.log(nu_min) + (np.log(nu_max) - np.log(nu_min)) * X_unit[:, -1:]

    X = np.hstack([x, y, t, eta]).astype(dtype, copy=False)
    return X, x.astype(dtype), y.astype(dtype), t.astype(dtype), eta.astype(dtype)


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
    nu_min, nu_max,
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

    value_rng, grad_rng = np.random.SeedSequence(seed).spawn(2)

    # --- value errors (omega)
    X_np, x_np, y_np, t_np, eta_np = sample_interior_lhs(
        x_min, x_max, y_min, y_max, t_min, t_max, nu_min, nu_max, 
        N=N_val, t_fixed=t_fixed,
        seed=int(value_rng.generate_state(1, dtype=np.uint64)[0]), dtype=dtype_np
    )
    X_t = torch.as_tensor(X_np, device=device, dtype=dtype_torch)
    # values do NOT need input grads
    omega_pred, _ = predict_omega_psi(func_model, params, X_t)
    omega_ex = exact_omega_np(x_np, y_np, t_np, eta_np)

    error_w_2r = rel_l2(omega_pred, omega_ex)
    error_w_infr = rel_linf(omega_pred, omega_ex)

    # --- gradient errors (grad psi) — use smaller N_grad for speed
    Xg_np, xg, yg, tg, etag = sample_interior_lhs(
        x_min, x_max, y_min, y_max, t_min, t_max, nu_min, nu_max, 
        N=N_grad, t_fixed=t_fixed, seed=int(grad_rng.generate_state(1, dtype=np.uint64)[0]), dtype=dtype_np
    )
    Xg_t = torch.as_tensor(Xg_np, device=device, dtype=dtype_torch)
    Xg_t.requires_grad_(True)

    psix_pred, psiy_pred = predict_gradpsi(func_model, params, Xg_t)
    psix_ex  = exact_psix_np(xg, yg, tg, etag)
    psiy_ex  = exact_psiy_np(xg, yg, tg, etag)

    error_gradpsi_2r = rel_l2_vec(psix_pred, psiy_pred, psix_ex, psiy_ex)
    error_gradpsi_infr = rel_linf_vec(psix_pred, psiy_pred, psix_ex, psiy_ex)

    return error_w_2r, error_w_infr, error_gradpsi_2r, error_gradpsi_infr
