import torch
from torch.func import vmap, grad, jacrev


def omega_scalar(func_model, params, x: torch.Tensor) -> torch.Tensor:
    # force scalar
    return func_model(params, x)[0].reshape(())


def psi_scalar(func_model, params, x: torch.Tensor) -> torch.Tensor:
    # force scalar
    return func_model(params, x)[1].reshape(())


def res_vorticity(func_model, params, x: torch.Tensor, nu: float) -> torch.Tensor:
    r"""
    ω_t + (u·∇)ω - νΔω = 0,
    u = ∇^⊥ψ = (ψ_y, -ψ_x) so (u·∇)ω = ψ_y ω_x - ψ_x ω_y.
    """
    w = lambda x_: omega_scalar(func_model, params, x_)
    psi = lambda x_: psi_scalar(func_model, params, x_)

    g1_w = grad(w, argnums=0)(x)
    w_x, w_y, w_t = g1_w[0], g1_w[1], g1_w[2]

    g1_psi = grad(psi, argnums=0)(x)
    psi_x, psi_y = g1_psi[0], g1_psi[1]

    H_w = jacrev(grad(w, argnums=0), argnums=0)(x)
    lap_w = H_w[0, 0] + H_w[1, 1]

    return w_t + (w_x * psi_y - w_y * psi_x) - nu * lap_w


def res_poisson_stream(func_model, params, x: torch.Tensor) -> torch.Tensor:
    r"""Δψ + ω = 0  (since ω = -Δψ)."""
    psi = lambda x_: psi_scalar(func_model, params, x_)

    H_psi = jacrev(grad(psi, argnums=0), argnums=0)(x)
    lap_psi = H_psi[0, 0] + H_psi[1, 1]

    omega = omega_scalar(func_model, params, x)
    return lap_psi + omega


def res_initial_omega(func_model, params, x: torch.Tensor, omega0: torch.Tensor) -> torch.Tensor:
    omega = omega_scalar(func_model, params, x)
    return omega - omega0.reshape(())

def periodic_value_jump_scalar(func_model, params, a, b, field: str):
    """
    Scalar periodic jump at a single paired point (a,b).
    Returns a scalar () tensor.
    """
    outA = func_model(params, a)
    outB = func_model(params, b)

    if field == "omega":
        return (outA[0] - outB[0]).reshape(())
    elif field == "psi":
        return (outA[1] - outB[1]).reshape(())
    else:
        raise ValueError("field must be 'omega' or 'psi'")

def periodic_normal_derivative_jump_scalar(func_model, params, a, b, field: str, axis: int):
    """
    Scalar jump of normal derivative at one paired point (a,b).

    field: "psi" or "omega"
    axis:  0 -> d/dx, 1 -> d/dy  (normal direction of the face)
    returns: scalar () tensor
    """
    if field == "psi":
        def s(x):
            return func_model(params, x)[1].reshape(())
    elif field == "omega":
        def s(x):
            return func_model(params, x)[0].reshape(())
    else:
        raise ValueError("field must be 'psi' or 'omega'")

    gs = grad(s)  # gradient wrt input x, returns (3,)
    return (gs(a)[axis] - gs(b)[axis]).reshape(())

def res_periodic_value_pairs(func_model, params, A, B, field: str):
    """
    Vectorized periodic value residual.
    A,B: (N,3)
    returns: (N,)
    """
    return vmap(
        lambda a, b: periodic_value_jump_scalar(func_model, params, a, b, field)
    )(A, B)

def res_periodic_normal_derivative_pairs(func_model, params, A, B, field: str, normal_axis: int):
    """
    Vectorized jump of normal derivative over paired points A,B (N,3).
    returns: (N,)
    """
    return vmap(
        lambda a, b: periodic_normal_derivative_jump_scalar(
            func_model, params, a, b, field, normal_axis
        )
    )(A, B)

def res_gauge_mean_psi(func_model, params, Xg: torch.Tensor) -> torch.Tensor:
    """mean(psi)=0 gauge over points Xg (Ng,3). Returns scalar."""
    psi_vals = vmap(lambda x_: psi_scalar(func_model, params, x_))(Xg)  # (Ng,)
    return psi_vals.mean()


def res_gauge_slice(func_model, params, X_slice: torch.Tensor) -> torch.Tensor:
    """Return the spatial mean of psi at fixed time."""
    return vmap(lambda x: psi_scalar(func_model, params, x))(X_slice).mean()
