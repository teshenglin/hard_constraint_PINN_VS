import math
import torch
from torch.func import vmap

from residuals_vs import (
    res_vorticity,
    res_poisson_stream,
    res_initial_omega,
    res_gauge_mean_psi,
)

def lbfgs_train_vs(
    *,
    model,
    make_func_model,
    X_inner,
    X_initial,
    omega0_initial,
    nu: float,
    max_iter: int = 2000,
    max_eval: int | None = None,
    lr: float = 1.0,
    history_size: int = 50,
    line_search_fn: str | None = "strong_wolfe",
    tolerance_grad: float = 1e-12,
    tolerance_change: float = 1e-12,
    weight_decay: float = 0.0,
    use_gauge: bool = True,
    verbose_every: int = 100,
):
    model.train()

    if max_eval is None:
        max_eval = max_iter * 5  # reasonable default

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        line_search_fn=line_search_fn,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )

    Nf = X_inner.shape[0]
    Ni = X_initial.shape[0]
    sf_f = 1.0 / math.sqrt(Nf)
    sf_i = 1.0 / math.sqrt(Ni)

    losses = []
    call = {"k": 0}

    def closure():
        opt.zero_grad(set_to_none=True)

        func_model, func_params = make_func_model(model)

        # ----- residual vectors (match your LM structure)
        L1 = vmap(lambda x: res_vorticity(func_model, func_params, x, nu=nu))(X_inner) * sf_f
        L2 = vmap(lambda x: res_poisson_stream(func_model, func_params, x))(X_inner) * sf_f
        Li = vmap(lambda x, w0: res_initial_omega(func_model, func_params, x, w0))(X_initial, omega0_initial) * sf_i

        loss = (L1.square().sum() + L2.square().sum() + Li.square().sum())

        if use_gauge:
            rg = res_gauge_mean_psi(func_model, func_params, X_inner) * sf_f
            loss = loss + rg.square()

        # ----- weight decay (LBFGS doesn't support it natively)
        if weight_decay > 0.0:
            l2 = torch.zeros((), device=loss.device, dtype=loss.dtype)
            for p in model.parameters():
                if p.requires_grad:
                    l2 = l2 + p.square().sum()
            loss = loss + weight_decay * l2

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in LBFGS closure: {loss.item()}")

        loss.backward()

        losses.append(loss.detach().item())
        if (call["k"] % verbose_every) == 0:
            print(f"[LBFGS] call={call['k']:6d} loss={loss.item():.6e}")
        call["k"] += 1

        return loss

    opt.step(closure)
    return losses


