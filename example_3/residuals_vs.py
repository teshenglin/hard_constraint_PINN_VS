import torch
from torch.func import vmap, grad, jacrev


def omega_scalar(func_model, params, x: torch.Tensor) -> torch.Tensor:
    # force scalar
    return func_model(params, x)[0].reshape(())

def psi_scalar(func_model, params, x: torch.Tensor) -> torch.Tensor:
    # force scalar
    return func_model(params, x)[1].reshape(())

def res_vorticity(func_model, params, x: torch.Tensor) -> torch.Tensor:
    w = lambda x_: omega_scalar(func_model, params, x_)
    psi = lambda x_: psi_scalar(func_model, params, x_)

    g1_w = grad(w)(x)                 # (3,)
    w_x, w_y, w_t = g1_w[0], g1_w[1], g1_w[2]

    g1_psi = grad(psi)(x)             # (3,)
    psi_x, psi_y = g1_psi[0], g1_psi[1]

    H_w = jacrev(grad(w))(x)          # (3,3)
    lap_w = H_w[0, 0] + H_w[1, 1]

    nu = torch.exp(x[3])

    return w_t + (w_x * psi_y - w_y * psi_x) - nu * lap_w

def res_poisson_stream(func_model, params, x: torch.Tensor) -> torch.Tensor:
    r"""Δψ + ω = 0  (since ω = -Δψ)."""
    psi = lambda x_: psi_scalar(func_model, params, x_)

    H_psi = jacrev(grad(psi))(x)      # (3,3)
    lap_psi = H_psi[0, 0] + H_psi[1, 1]

    omega = omega_scalar(func_model, params, x)
    # omega_mean = omega.mean(dim=0, keepdim=True)
    # omega_tilde = omega - omega_mean
    return lap_psi + omega


def res_initial_omega(func_model, params, x: torch.Tensor, omega0: torch.Tensor) -> torch.Tensor:
    omega = omega_scalar(func_model, params, x)
    return omega - omega0.reshape(())

def res_periodic_value_pairs(func_model, params, A, B, field: str):
    """
    A,B: (N,3) paired points.
    field: 'omega' or 'psi'
    returns: (N,) residual vector
    """
    if field == "omega":
        f = lambda x: omega_scalar(func_model, params, x)
    elif field == "psi":
        f = lambda x: psi_scalar(func_model, params, x)
    else:
        raise ValueError("field must be 'omega' or 'psi'")

    fa = vmap(f)(A)  # (N,)
    fb = vmap(f)(B)  # (N,)
    return fa - fb

def res_periodic_normal_derivative_pairs(func_model, params, A, B, field: str, normal_axis: int):
    """
    Enforce periodicity of the normal derivative of a scalar field across paired faces.
    field: 'psi' or 'omega'
    normal_axis: 0 for x-faces (∂x), 1 for y-faces (∂y)
    returns: (N,)
    """
    if field == "psi":
        def s(p, x):
            return func_model(p, x)[1].reshape(())
    elif field == "omega":
        def s(p, x):
            return func_model(p, x)[0].reshape(())
    else:
        raise ValueError("field must be 'psi' or 'omega'")

    gs = grad(s, argnums=1)  # gradient wrt input x -> (3,)
    gA = vmap(lambda x: gs(params, x))(A)  # (N,3)
    gB = vmap(lambda x: gs(params, x))(B)  # (N,3)

    return gA[:, normal_axis] - gB[:, normal_axis]

def res_gauge_mean_psi(func_model, params, Xg: torch.Tensor) -> torch.Tensor:
    """mean(psi)=0 gauge over points Xg (Ng,3). Returns scalar."""
    psi_vals = vmap(lambda x_: psi_scalar(func_model, params, x_))(Xg)  # (Ng,)
    return psi_vals.mean()


def res_gauge_slice(func_model, params, X_slice: torch.Tensor) -> torch.Tensor:
    """Return the spatial mean of psi at fixed time and log viscosity."""
    return vmap(lambda x: psi_scalar(func_model, params, x))(X_slice).mean()
