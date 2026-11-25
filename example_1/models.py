import torch
import torch.nn as nn
from typing import Callable

class PeriodicFourierMLP(nn.Module):
    r"""
    MLP with fixed sin/cos Fourier embedding for 2π-periodic 2D spatial variables.

    Input:
        x = (x, y, t)

    Embedding:
        (cos x, sin x, cos y, sin y, t)

    Designed for:
        - 2D periodic domains
        - vorticity–stream function PINNs
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        out_dim: int,
        activation: Callable = nn.Tanh,
        bias_last: bool = False,
    ):
        super().__init__()

        assert depth >= 1, "depth must be >= 1"

        self.in_dim = 5  # cos x, sin x, cos y, sin y, t

        layers = []

        # input layer
        layers.append(nn.Linear(self.in_dim, hidden_dim))
        layers.append(activation())

        # hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        self.hidden = nn.Sequential(*layers)

        # output layer
        # output ordering: [omega, psi]
        self.out = nn.Linear(hidden_dim, out_dim, bias=bias_last)

    def fourier_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3) tensor with columns (x, y, t)
        returns: (..., 5) embedded features
        """
        x0, y0, t = x[..., 0], x[..., 1], x[..., 2]

        return torch.stack(
            [
                torch.cos(x0),
                torch.sin(x0),
                torch.cos(y0),
                torch.sin(y0),
                t,
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (..., 3) = (x, y, t)
        """
        feat = self.fourier_embedding(x)
        h = self.hidden(feat)
        return self.out(h)


class PeriodicFourierMLP_nu(nn.Module):
    r"""
    MLP with fixed sin/cos Fourier embedding for 2π-periodic 2D spatial variables.

    Input:
        x = (x, y, t, nu)

    Embedding:
        (cos x, sin x, cos y, sin y, t, nu)

    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        out_dim: int,
        activation: Callable = nn.Tanh,
        bias_last: bool = False,
    ):
        super().__init__()

        assert depth >= 1, "depth must be >= 1"

        self.in_dim = 6  # cos x, sin x, cos y, sin y, t

        layers = []

        # input layer
        layers.append(nn.Linear(self.in_dim, hidden_dim))
        layers.append(activation())

        # hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        self.hidden = nn.Sequential(*layers)

        # output layer
        # output ordering: [omega, psi]
        self.out = nn.Linear(hidden_dim, out_dim, bias=bias_last)

    def fourier_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x0, y0, t, nu = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        return torch.stack(
            [
                torch.cos(x0),
                torch.sin(x0),
                torch.cos(y0),
                torch.sin(y0),
                t,
                nu,
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (..., 3) = (x, y, t, nu)
        """
        feat = self.fourier_embedding(x)
        h = self.hidden(feat)
        return self.out(h)

class BasicMLP(nn.Module):
    """
    Plain MLP for inputs (x,y,t) -> outputs (omega, psi).
    No periodic embedding.
    """
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        out_dim: int,
        activation: Callable = nn.Tanh,
        bias_last: bool = False,
    ):
        super().__init__()
        assert depth >= 1

        self.in_dim = 3

        layers = [nn.Linear(self.in_dim, hidden_dim), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        self.net = nn.Sequential(*layers)
        # output ordering: [omega, psi]
        self.out = nn.Linear(hidden_dim, out_dim, bias=bias_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,3)
        h = self.net(x)
        return self.out(h)


def plot_losses(losses, label=None):
    import matplotlib.pyplot as plt

    it = range(len(losses))

    plt.figure(figsize=(5,4))
    plt.semilogy(it, losses, linewidth=2, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    if label:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def count_parameters(model: nn.Module, requires_grad: bool = True) -> int:
    """Count parameters of a PyTorch module."""
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())