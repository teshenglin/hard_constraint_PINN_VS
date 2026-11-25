from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VSConfig:
    # Set to an integer for reproducible runs, or None for fresh randomness.
    seed: int | None = None

    # domain
    x_min: float = 0.0
    x_max: float = 2*np.pi
    y_min: float = 0.0
    y_max: float = 2*np.pi
    t_min: float = 0.0
    t_max: float = 1.0

    # PDE
    nu: float = 0.001

    # Model
    hidden_dim: int = 10
    depth: int = 2

    # sampling
    N_initial: int = 300
    N_boundary: int = 0  # Periodicity is built into the model.
    N_gauge_times: int = 11
    N_gauge_space: int = 256
    N_interior: int = 2500

    # --------------------
    # Optimizers
    # --------------------
    # LM
    LM_iter: int = 3001
    mu_init: float = 1e5
    mu_update_every: int = 2
    div_factor: float = 1.3
    mul_factor: float = 3.0

    # LBFGS
    LBFGS_iter: int = 3000            # LBFGS max_iter (often << LM_iter)
    LBFGS_lr: float = 1.0
    LBFGS_history_size: int = 50
    #LBFGS_line_search_fn: str | None = None
    LBFGS_line_search_fn: str = "strong_wolfe"
    LBFGS_tolerance_grad: float = 1e-12
    LBFGS_tolerance_change: float = 1e-12
    LBFGS_max_eval: int | None = None          # None lets PyTorch decide
    LBFGS_weight_decay: float = 0.0            # usually 0 for PINNs

    # ADAM
    ADAM_steps: int = 5000
    ADAM_lr: float = 1e-3
    ADAM_weight_decay: float = 0.0
    ADAM_grad_clip: float | None = None

    # eval
    N_val: int  = 10000  # Vorticity evaluation points
    N_grad: int = 10000  # Stream-function gradient evaluation points
