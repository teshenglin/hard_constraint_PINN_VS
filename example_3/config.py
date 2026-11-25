from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VSConfig:
    # Use None for fresh randomness.
    seed: int | None = None

    # domain
    x_min: float = 0.0
    x_max: float = 2*np.pi
    y_min: float = 0.0
    y_max: float = 2*np.pi
    t_min: float = 0.0
    t_max: float = 1.0
    nu_min: float = 0.001
    nu_max: float = 1.0

    # Model
    hidden_dim: int = 20
    depth: int = 2

    # sampling
    N_initial: int = 300
    N_interior: int = 2500

    # Gauge loss
    gauge_weight: float = 0.01
    N_gauge_times: int = 11
    N_gauge_viscosities: int = 4
    N_gauge_space: int = 256

    # --------------------
    # Optimizers
    # --------------------
    # LM
    LM_iter: int = 3001
    mu_init: float = 1e5
    mu_update_every: int = 2
    div_factor: float = 1.3
    mul_factor: float = 3.0

    # eval
    N_val: int = 10000
    N_grad: int = 10000
