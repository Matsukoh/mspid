import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(
    mlp_dims: list[int],
    activation: str = "leaky_relu",
    last_act: bool = False,
    layer_norm: bool = False,
):
    layers = []
    mlp_dims = mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_act:
            if layer_norm:
                layers.append(nn.LayerNorm(mlp_dims[i + 1]))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "mish":
                layers.append(nn.Mish())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
    net = nn.Sequential(*layers)
    return net


def tensor_clamp(x: torch.Tensor, max_tensor: torch.Tensor, min_tensor: torch.Tensor):
    return torch.max(torch.min(x, max_tensor), min_tensor)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal scalar-time embedding followed by a small MLP."""

    def __init__(self, embed_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if embed_dim < 4:
            raise ValueError("embed_dim must be at least 4.")
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}.")

        half = self.embed_dim // 2
        exponent = (
            -torch.log(torch.tensor(self.max_period, device=t.device, dtype=t.dtype))
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        frequencies = torch.exp(exponent)
        angles = t[:, None] * frequencies[None, :] * (2.0 * torch.pi)
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        if embedding.shape[-1] < self.embed_dim:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.embed_dim - embedding.shape[-1])
            )
        return self.mlp(embedding)
