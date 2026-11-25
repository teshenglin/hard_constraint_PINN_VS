import math
import torch
from torch.func import vmap, jacrev

from collections.abc import Mapping

from residuals_vs_v2 import (
    res_vorticity,
    res_poisson_stream,
    res_initial_omega,
    res_periodic_value_pairs,
    res_periodic_normal_derivative_pairs, 
    res_gauge_slice,
    periodic_value_jump_scalar,
    periodic_normal_derivative_jump_scalar,
)

def _flatten_param_jacobian(jac_parts, nrows: int, param_keys):
    """
    Flatten a parameter Jacobian into a dense (nrows, P) matrix.

    jac_parts:
      - dict[str, Tensor] if params is a dict
      - list/tuple[Tensor] if params is a list
    """
    cols = []

    if isinstance(jac_parts, Mapping):
        # params is a dict → jac_parts is a dict with same keys
        for k in param_keys:
            g = jac_parts[k]            # (nrows, *param_shape)
            cols.append(g.reshape(nrows, -1))
    else:
        # params is list/tuple (old style)
        for g in jac_parts:
            cols.append(g.reshape(nrows, -1))

    return torch.cat(cols, dim=1)

def lm_train_vs(
    *,
    func_model,
    func_params,                 # list/tuple of tensors (functional params)
    X_inner,                     # torch (Nf,3), requires_grad True
    X_initial,                   # torch (Ni,3), requires_grad True
    omega0_initial,              # torch (Ni,1) or (Ni,)
    # NEW: periodic boundary pairs (optional)
    X0_bd=None, X1_bd=None,        # (Nb,3) torch
    Y0_bd=None, Y1_bd=None,        # (Nb,3) torch
    # optional weights (keep 1.0 for now)
    nu: float = 0.001,
    w_p_val: float = 1.0,
    w_p_der: float = 1.0,
    max_steps: int = 2000,
    mu_init: float = 1e-2,
    mu_update_every: int = 25,
    div_factor: float = 10.0,
    mul_factor: float = 10.0,
    tol: float = 1e-12,
    use_gauge: bool = True,
    gauge_weight: float = 0.01,
    X_gauge: torch.Tensor | None = None,
    verbose_every: int = 100,
):
    """
    Levenberg–Marquardt training for vorticity–stream PINN (ω,ψ).
    Returns: losses(list[float]), updated func_params
    """

    if isinstance(func_params, dict):
        param_keys = list(func_params.keys())
    else:
        param_keys = None

    device = X_inner.device
    dtype = X_inner.dtype

    use_periodic = (X0_bd is not None) and (X1_bd is not None) and (Y0_bd is not None) and (Y1_bd is not None)

    Nf = X_inner.shape[0]
    Ni = X_initial.shape[0]

    sf_f = 1.0 / math.sqrt(Nf)
    sf_i = 1.0 / math.sqrt(Ni)

    if use_gauge:
        if X_gauge is None or X_gauge.ndim != 3 or X_gauge.shape[-1] != X_inner.shape[-1] or X_gauge.numel() == 0:
            raise ValueError("X_gauge must contain nonempty (slice, spatial point, coordinate) coordinates")
        N_gauge = X_gauge.shape[0]
        if gauge_weight < 0:
            raise ValueError("gauge_weight must be nonnegative")
        sf_g = math.sqrt(gauge_weight / N_gauge)

    mu = torch.tensor(mu_init, device=device, dtype=dtype)

    losses = []
    loss_prev = torch.tensor(float("inf"), device=device, dtype=dtype)

    # ---- define per-sample residuals for vmap
    def r1_one(p, x):
        return res_vorticity(func_model, p, x, nu)

    def r2_one(p, x):
        return res_poisson_stream(func_model, p, x)

    def ri_one(p, x, w0):
        return res_initial_omega(func_model, p, x, w0)


    def rg_one(p, points):
        return res_gauge_slice(func_model, p, points)

    for step in range(max_steps):

        # -------- residual vectors (scaled)
        L1 = vmap(lambda x: r1_one(func_params, x))(X_inner) * sf_f   # (Nf,)
        L2 = vmap(lambda x: r2_one(func_params, x))(X_inner) * sf_f   # (Nf,)
        Li = vmap(lambda x, w0: ri_one(func_params, x, w0))(X_initial, omega0_initial) * sf_i  # (Ni,)

        L_list = [L1, L2, Li]

        # -------- Jacobian blocks w.r.t params
        # jacrev of scalar residual gives pytree of params-shaped tensors
        J1_parts = vmap(lambda x: jacrev(lambda p: r1_one(p, x))(func_params))(X_inner)  # pytree with leading Nf
        J2_parts = vmap(lambda x: jacrev(lambda p: r2_one(p, x))(func_params))(X_inner)
        Ji_parts = vmap(lambda x, w0: jacrev(lambda p: ri_one(p, x, w0))(func_params))(X_initial, omega0_initial)
        
        # flatten to dense
        J1 = _flatten_param_jacobian(J1_parts, Nf, param_keys) * sf_f
        J2 = _flatten_param_jacobian(J2_parts, Nf, param_keys) * sf_f
        Ji = _flatten_param_jacobian(Ji_parts, Ni, param_keys) * sf_i

        J_list = [J1, J2, Ji]

        if use_periodic:

            assert X0_bd.shape == X1_bd.shape
            assert Y0_bd.shape == Y1_bd.shape
            assert X0_bd.shape[1] == 3 and Y0_bd.shape[1] == 3

            Nb = X0_bd.shape[0]
            sf_b = 1.0 / math.sqrt(Nb)
            
            # residual vectors
            Lpx_w = res_periodic_value_pairs(func_model, func_params, X0_bd, X1_bd, field="omega") * sf_b * w_p_val
            Lpx_p = res_periodic_value_pairs(func_model, func_params, X0_bd, X1_bd, field="psi")   * sf_b * w_p_val
            Lpy_w = res_periodic_value_pairs(func_model, func_params, Y0_bd, Y1_bd, field="omega") * sf_b * w_p_val
            Lpy_p = res_periodic_value_pairs(func_model, func_params, Y0_bd, Y1_bd, field="psi")   * sf_b * w_p_val
        
            Lpx_dpsi   = res_periodic_normal_derivative_pairs(func_model, func_params, X0_bd, X1_bd, field="psi",   normal_axis=0) * sf_b * w_p_der
            Lpy_dpsi   = res_periodic_normal_derivative_pairs(func_model, func_params, Y0_bd, Y1_bd, field="psi",   normal_axis=1) * sf_b * w_p_der
        
            L_list += [Lpx_w, Lpx_p, Lpy_w, Lpy_p, Lpx_dpsi, Lpy_dpsi]

            Jpx_w_parts = vmap(lambda a, b: jacrev(lambda p: periodic_value_jump_scalar(func_model, p, a, b, "omega"))(func_params))(X0_bd, X1_bd)
            Jpx_p_parts = vmap(lambda a, b: jacrev(lambda p: periodic_value_jump_scalar(func_model, p, a, b, "psi"))(func_params))(X0_bd, X1_bd)
            Jpy_w_parts = vmap(lambda a, b: jacrev(lambda p: periodic_value_jump_scalar(func_model, p, a, b, "omega"))(func_params))(Y0_bd, Y1_bd)
            Jpy_p_parts = vmap(lambda a, b: jacrev(lambda p: periodic_value_jump_scalar(func_model, p, a, b, "psi"))(func_params))(Y0_bd, Y1_bd)
        
            Jpx_w = _flatten_param_jacobian(Jpx_w_parts, Nb, param_keys) * sf_b * w_p_val
            Jpx_p = _flatten_param_jacobian(Jpx_p_parts, Nb, param_keys) * sf_b * w_p_val
            Jpy_w = _flatten_param_jacobian(Jpy_w_parts, Nb, param_keys) * sf_b * w_p_val
            Jpy_p = _flatten_param_jacobian(Jpy_p_parts, Nb, param_keys) * sf_b * w_p_val

            Jpx_dpsi_parts = vmap(lambda a, b: jacrev(lambda p: periodic_normal_derivative_jump_scalar(func_model, p, a, b, "psi", 0))(func_params))(X0_bd, X1_bd)
            Jpy_dpsi_parts = vmap(lambda a, b: jacrev(lambda p: periodic_normal_derivative_jump_scalar(func_model, p, a, b, "psi", 1))(func_params))(Y0_bd, Y1_bd)

            Jpx_dpsi   = _flatten_param_jacobian(Jpx_dpsi_parts, Nb, param_keys) * sf_b * w_p_der
            Jpy_dpsi   = _flatten_param_jacobian(Jpy_dpsi_parts, Nb, param_keys) * sf_b * w_p_der
        
            J_list += [Jpx_w, Jpx_p, Jpy_w, Jpy_p, Jpx_dpsi, Jpy_dpsi]
        
        if use_gauge:
            rg = vmap(rg_one, in_dims=(None, 0))(func_params, X_gauge) * sf_g
            Jg_parts = vmap(jacrev(rg_one), in_dims=(None, 0))(func_params, X_gauge)
            Jg = _flatten_param_jacobian(Jg_parts, N_gauge, param_keys) * sf_g
            L_list += [rg]
            J_list += [Jg]
        
        L = torch.cat(L_list, dim=0)
        J = torch.cat(J_list, dim=0)
        
        loss = L.square().sum()


        # -------- LM step
        P = J.shape[1]
        I = torch.eye(P, device=device, dtype=dtype)

        if step == 0:
            print("J shape:", J.shape)
            print("L shape:", L.shape)
            print("Parameter count P:", J.shape[1])
            print("Residual count M:", J.shape[0])

        with torch.no_grad():
            loss_val = loss.item()
            if loss_val < 1e-13:
                A = torch.cat([J, torch.sqrt(mu) * I], dim=0)
                b = torch.cat([-L, torch.zeros(P, device=device, dtype=dtype)], dim=0)
                dp = torch.linalg.lstsq(A, b).solution
            else:
                JTJ = J.T @ J
                rhs = -(J.T @ L)
                dp = torch.linalg.solve(JTJ + mu * I, rhs)
            

            # if step % 100 == 0 or step == 0:
            #     with torch.no_grad():
            #         A = J.T @ J + mu * I
            #         eigvals = torch.linalg.eigvalsh(A)
            #         lam_min = eigvals.min().clamp_min(1e-30)
            #         cond_A = (eigvals.max() / lam_min).item()
            #         print(f"step {step}: cond(A) = {cond_A:.2e}")

            # update params (in-place)
            offset = 0
            if isinstance(func_params, dict):
                for k in param_keys:
                    p = func_params[k]
                    num = p.numel()
                    p.add_(dp[offset:offset+num].reshape(p.shape))
                    offset += num
            else:
                for p in func_params:
                    num = p.numel()
                    p.add_(dp[offset:offset+num].reshape(p.shape))
                    offset += num

            # μ update
            if (step + 1) % mu_update_every == 0:
                if loss < loss_prev:
                    mu = torch.clamp(mu / div_factor, min=1e-12)
                else:
                    mu = torch.clamp(mu * mul_factor, max=1e12)
                loss_prev = loss

        losses.append(float(loss.item()))

        if verbose_every and (step % verbose_every == 0):
            msg = f"step {step:5d} | loss={loss.item():.5e} | mu={mu.item():.3e}"
            print(msg)

        if loss.item() < tol:
            break

    return losses, func_params
