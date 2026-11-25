import numpy as np
import torch

# ---------- NumPy exact solution (for data / plotting) ----------
def exact_omega_np(x, y, t, eta: float):
    return 2.0 * np.exp(-2.0 * np.exp(eta) * t) * np.sin(x) * np.sin(y)

#def exact_psi_np(x, y, t, nu: float):
#    return np.exp(-2.0 * nu * t) * np.sin(x) * np.sin(y)

#def exact_gradpsi_np(x, y, t, nu: float):
#    e = np.exp(-2.0 * nu * t)
#    psi_x = e * np.cos(x) * np.sin(y)
#    psi_y = e * np.sin(x) * np.cos(y)
#    return psi_x, psi_y

def exact_psix_np(x, y, t, eta: float):
    return np.exp(-2.0 * np.exp(eta) * t) * np.cos(x) * np.sin(y)

def exact_psiy_np(x, y, t, eta: float):
    return np.exp(-2.0 * np.exp(eta) * t) * np.sin(x) * np.cos(y)

# ---------- Torch exact solution (optional; for GPU eval) ----------
#def exact_omega_torch(x, y, t, nu: float):
#    return 2.0 * torch.exp(-2.0 * nu * t) * torch.sin(x) * torch.sin(y)

#def exact_psi_torch(x, y, t, nu: float):
#    return torch.exp(-2.0 * nu * t) * torch.sin(x) * torch.sin(y)

#def exact_gradpsi_torch(x, y, t, nu: float):
#    e = torch.exp(-2.0 * nu * t)
#    psi_x = e * torch.cos(x) * torch.sin(y)
#    psi_y = e * torch.sin(x) * torch.cos(y)
#    return psi_x, psi_y
