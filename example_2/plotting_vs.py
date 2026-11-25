import numpy as np
import torch
from torch.func import vmap
import matplotlib.pyplot as plt


def cast_params(params, dtype: torch.dtype):
    if isinstance(params, dict):
        return {k: v.to(dtype) for k, v in params.items()}
    return [p.to(dtype) for p in params]


@torch.no_grad()
def plot_omega_psi_slice(
    *,
    func_model,
    params,
    t_val: float,
    x_min: float = 0.0,
    x_max: float = 2 * np.pi,
    y_min: float = 0.0,
    y_max: float = 2 * np.pi,
    n: int = 256,
    device=None,
    dtype: torch.dtype = torch.float64,
    titles=("Predicted ω", "Predicted ψ"),
    cmap="jet",
    use_imshow: bool = True,
):
    """
    Plot ω and ψ predictions on a uniform (x,y) grid at fixed time t=t_val.
    """
    if device is None:
        device = next(iter(params.values())).device if isinstance(params, dict) else params[0].device

    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    x_mesh, y_mesh = np.meshgrid(xs, ys, indexing="xy")

    x_flat = torch.as_tensor(x_mesh.reshape(-1, 1), device=device, dtype=dtype)
    y_flat = torch.as_tensor(y_mesh.reshape(-1, 1), device=device, dtype=dtype)
    t_flat = torch.full_like(x_flat, float(t_val))

    X = torch.cat([x_flat, y_flat, t_flat], dim=1)  # (n*n, 3)

    # predict (N,2)
    params_plot = cast_params(params, dtype)
    pred = vmap(func_model, in_dims=(None, 0))(params_plot, X)
    omega = pred[:, 0].detach().cpu().numpy().reshape(n, n)
    psi   = pred[:, 1].detach().cpu().numpy().reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if use_imshow:
        # imshow is fast; extent makes axes show x/y coordinates
        im0 = axes[0].imshow(
            omega, origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto", cmap=cmap
        )
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(titles[0])
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

        im1 = axes[1].imshow(
            psi, origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto", cmap=cmap
        )
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(titles[1])
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    else:
        # pcolormesh gives correct grid mapping; faster than pcolor
        im0 = axes[0].pcolormesh(x_mesh, y_mesh, omega, shading="auto", cmap=cmap)
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(titles[0])
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

        im1 = axes[1].pcolormesh(x_mesh, y_mesh, psi, shading="auto", cmap=cmap)
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(titles[1])
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    fig.suptitle(f"t = {t_val:g}", y=1.02)
    plt.tight_layout()
    plt.show()

    return omega, psi  # handy if you want to reuse arrays
