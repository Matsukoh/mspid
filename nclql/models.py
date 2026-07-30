import math
from dataclasses import dataclass
from math import log
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import SinusoidalTimeEmbedding, make_mlp, tensor_clamp


class Critic(nn.Module):
    def __init__(
        self, D: int, d: int, h_dims: list[int] = [256], activation: str = "silu"
    ):
        super().__init__()
        self.net1 = make_mlp(
            [D] + h_dims, activation=activation, last_act=True, layer_norm=False
        )
        self.final_layer1 = nn.Linear(h_dims[-1], d)

        self.net2 = make_mlp(
            [D] + h_dims, activation=activation, last_act=True, layer_norm=False
        )
        self.final_layer2 = nn.Linear(h_dims[-1], d)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        data = torch.cat([obs, act], -1)

        phi1 = self.net1(data)
        phi1 = phi1 / (phi1.norm(dim=-1, keepdim=True) + 1e-8)
        out1 = self.final_layer1(phi1)

        phi2 = self.net2(data)
        phi2 = phi2 / (phi2.norm(dim=-1, keepdim=True) + 1e-8)
        out2 = self.final_layer1(phi2)
        return out1, out2


class NoiseConditionedCritic(nn.Module):
    def __init__(
        self,
        D: int,
        d: int,
        h_dims: list[int] = [256],
        time_dim: int = 16,
        activation: str = "silu",
    ):
        super().__init__()
        self.net1 = make_mlp(
            [D] + h_dims, activation=activation, last_act=True, layer_norm=True
        )
        self.final_layer1 = nn.Linear(h_dims[-1], d)

        self.net2 = make_mlp(
            [D] + h_dims, activation=activation, last_act=True, layer_norm=True
        )
        self.final_layer2 = nn.Linear(h_dims[-1], d)

        self.time_dim = time_dim

        self.time_mlp = SinusoidalTimeEmbedding(time_dim)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, i: torch.Tensor):
        t = self.time_mlp(i).reshape((-1, self.time_dim))
        data = torch.cat([obs, act, t], -1)

        phi1 = self.net1(data)
        # phi1 = phi1 / (phi1.norm(dim=-1, keepdim=True) + 1e-8)
        out1 = self.final_layer1(phi1)

        phi2 = self.net2(data)
        # phi2 = phi2 / (phi2.norm(dim=-1, keepdim=True) + 1e-8)
        out2 = self.final_layer1(phi2)
        return out1, out2


class AnnealedLangevinDynamics(nn.Module):
    """Annealed Langevin dynamics sampler implemented with PyTorch.

    ``model`` is expected to return two Q-value tensors from ``model(x, level)``.
    Only gradients with respect to the samples are computed; model parameter
    gradients are not accumulated.
    """

    def __init__(
        self,
        model: nn.Module,
        L: int,
        T: int,
        act_dim: int,
        act_max: list[float],
        act_min: list[float],
        q_grad_norm: bool = False,
        w: float = 1.0,
        sigma_max: float = 0.1,
        sigma_min: float = 0.001,
        step_lr: float = 0.001,
    ):
        super().__init__()
        self.model = model
        self.L = L
        self.T = T
        self.q_grad_norm = q_grad_norm
        self.w = w
        self.act_dim = act_dim
        self.act_max = torch.as_tensor(act_max, dtype=torch.float32)
        self.act_min = torch.as_tensor(act_min, dtype=torch.float32)
        self.act_ranges = self.act_max - self.act_min
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.step_lr = step_lr

    def __post_init__(self) -> None:
        if self.L <= 0 or self.T <= 0:
            raise ValueError("L and T must be positive")
        if self.sigma_max <= 0 or self.sigma_min <= 0:
            raise ValueError("sigma_max and sigma_min must be positive")
        if self.step_lr <= 0:
            raise ValueError("step_lr must be positive")

    def sigma_schedule(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the geometrically spaced noise levels."""
        dtype = dtype or torch.get_default_dtype()
        return torch.exp(
            torch.linspace(
                log(self.sigma_max),
                log(self.sigma_min),
                self.L,
                device=device,
                dtype=dtype,
            )
        )

    def sample(
        self,
        obs: torch.Tensor,
        shape: Tuple[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Draw samples, initialized uniformly in ``[-1, 1]``.

        Args:
            model: Callable with the signature ``model(x, level) -> (q1, q2)``.
            shape: Shape of the returned sample tensor.
            generator: Optional random generator for reproducible sampling.
            device: Sampling device. Defaults to the model parameter device.
            dtype: Sampling dtype. Defaults to the model parameter dtype.
        """
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("shape dimensions must be positive")

        try:
            parameter = next(self.model.parameters())
        except (AttributeError, StopIteration):
            parameter = None

        if device is None:
            device = parameter.device if parameter is not None else torch.device("cpu")
        else:
            device = torch.device(device)
        if dtype is None:
            dtype = (
                parameter.dtype if parameter is not None else torch.get_default_dtype()
            )
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")

        sigmas = self.sigma_schedule(device=device, dtype=dtype)
        # x = torch.empty(shape, device=device, dtype=dtype).uniform_(
        #     self.act_min, self.act_max
        # )
        x = torch.rand(shape, device=device) * self.act_ranges.to(
            device
        ) + self.act_min.to(device)

        # Detaching each iteration prevents an ever-growing autograd graph.
        for level, sigma in enumerate(sigmas):
            level_tensor = torch.tensor(level, device=device, dtype=torch.long).expand(
                shape[0]
            )
            step_size = self.step_lr * (sigma / sigmas[-1]).square()
            for _ in range(self.T):
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    q1, q2 = self.model(obs, x, level_tensor)
                    grad_x = torch.autograd.grad(q1.sum() + q2.sum(), x)[0]

                if self.q_grad_norm:
                    grad_x = grad_x / (
                        torch.linalg.vector_norm(grad_x, dim=-1, keepdim=True) + 1e-8
                    )

                noise = torch.randn(shape, device=device, dtype=dtype)
                x = x + 0.5 * step_size * self.w * grad_x + step_size.sqrt() * noise
                # x = x.clamp(self.act_min, self.act_max)
                x = tensor_clamp(x, self.act_max.to(device), self.act_min.to(device))

        return x.detach()
