import math
import torch
from torch.func import vmap

from residuals_vs import (
    res_vorticity,
    res_poisson_stream,
    res_initial_omega,
    res_gauge_mean_psi,
)

def adam_train_vs(
    *,
    model,
    make_func_model,
    X_inner,              # (Nf,3), requires_grad True
    X_initial,            # (Ni,3), usually requires_grad False
    omega0_initial,       # (Ni,) or (Ni,1)
    nu: float,
    max_steps: int = 20000,
    lr: float = 1e-3,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,   # AdamW-style decay; see note below
    use_gauge: bool = True,
    grad_clip: float | None = None,
    verbose_every: int = 200,
    scheduler=None,              # optional torch scheduler
):
    """
    Train vorticity–streamfunction PINN with Adam.

    Notes:
      - This uses model.parameters() (not func_params).
      - It rebuilds (func_model, func_params) from model each step via make_func_model(model),
        so it can reuse your residuals_vs functions that accept (func_model, params, x).
      - weight_decay here is AdamW-style decay via optimizer. If you want L2 penalty instead,
        set optimizer weight_decay=0 and add ||theta||^2 manually to loss.
    """
    model.train()

    # normalize omega0 shape if needed
    if omega0_initial.ndim == 2 and omega0_initial.shape[1] == 1:
        omega0_initial = omega0_initial.squeeze(1)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    Nf = X_inner.shape[0]
    Ni = X_initial.shape[0]
    sf_f = 1.0 / math.sqrt(Nf)
    sf_i = 1.0 / math.sqrt(Ni)

    losses = []

    for step in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)

        func_model, func_params = make_func_model(model)

        # Residual vectors (scaled)
        L1 = vmap(lambda x: res_vorticity(func_model, func_params, x, nu=nu))(X_inner) * sf_f
        L2 = vmap(lambda x: res_poisson_stream(func_model, func_params, x))(X_inner) * sf_f
        Li = vmap(lambda x, w0: res_initial_omega(func_model, func_params, x, w0))(X_initial, omega0_initial) * sf_i

        loss = L1.square().sum() + L2.square().sum() + Li.square().sum()

        if use_gauge:
            rg = res_gauge_mean_psi(func_model, func_params, X_inner) * sf_f
            loss = loss + rg.square()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = loss.detach().item()
        losses.append(loss_val)

        if (step % verbose_every) == 0 or step == 1:
            print(f"[Adam] step={step:6d}/{max_steps} loss={loss_val:.6e}")

    return losses
