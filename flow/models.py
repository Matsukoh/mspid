from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.model_utils import SinusoidalTimeEmbedding, tensor_clamp


class ConditionalMeanVelocityNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        h_dim: int = 256,
        time_dim: int = 16,
    ):
        super().__init__()

        self.time_dim = time_dim

        self.act_encoder = nn.Linear(act_dim, h_dim)

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
        )

        self.t_embedding = SinusoidalTimeEmbedding(time_dim)

        self.r_embedding = SinusoidalTimeEmbedding(time_dim)

        self.time_projection = nn.Linear(2 * time_dim, h_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(3 * h_dim, h_dim),
            nn.SiLU(),
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, act_dim),
        )

    def forward(
        self, x_t: torch.Tensor, obs: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ):
        obs_feature = self.obs_encoder(obs)
        act_feature = self.act_encoder(x_t)

        time_feature = self.time_projection(
            torch.cat([self.r_embedding(r), self.t_embedding(t)], dim=-1)
        )

        h = self.input_projection(
            torch.cat([obs_feature, act_feature, time_feature], dim=-1)
        )
        out = self.output_head(h)
        return out


class MeanFlowPolicy(nn.Module):
    def __init__(
        self,
        vnet: nn.Module,
        act_dim: int,
        act_max: list[float],
        act_min: list[float],
    ):
        super().__init__()
        self.vnet = vnet
        self.act_dim = act_dim
        # self.act_max = torch.as_tensor(act_max, dtype=torch.float32)
        # self.act_min = torch.as_tensor(act_min, dtype=torch.float32)
        self.register_buffer(
            "act_max",
            torch.as_tensor(act_max, dtype=torch.float32),
        )
        self.register_buffer(
            "act_min",
            torch.as_tensor(act_min, dtype=torch.float32),
        )

        self.register_buffer(
            "act_ranges",
            self.act_max - self.act_min,
            persistent=False,
        )

        # self.act_ranges = self.act_max - self.act_min

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor | Tuple[torch.Tensor],
        shape: Tuple[int],
        num_steps: int = 1,
        noise: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        batch_size = shape[0]

        if isinstance(obs, tuple):
            device = obs[0].device
        else:
            device = obs.device

        if noise is None:
            if deterministic:
                x = torch.zeros(
                    batch_size,
                    self.act_dim,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                x = torch.randn(
                    batch_size,
                    self.act_dim,
                    device=device,
                    dtype=torch.float32,
                )
        else:
            if noise.shape != (batch_size, self.act_dim):
                raise ValueError(
                    f"noise must have shape [{batch_size}, {self.act_dim}]."
                )
            x = noise.to(device=device, dtype=torch.float32)

        if num_steps == 1:
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)

            u = self.vnet(x, obs, r, t)

            # x_0 = x_1 - u(x_1, 0, 1)
            x0 = tensor_clamp(x - u, self.act_max, self.act_min)

        else:
            z = x

            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)

            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]

                t = torch.full((batch_size,), t_cur, device=device)
                r = torch.full((batch_size,), t_next, device=device)

                u = self.vnet(z, r, t)

                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = z - (t_cur - t_next) * u

            x0 = tensor_clamp(z, self.act_max, self.act_min)

        return x0

    def sample(
        self,
        obs: torch.Tensor | Tuple[torch.Tensor],
        shape: Tuple[int],
        num_steps: int = 1,
        noise: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        return self.forward(obs, shape, num_steps, noise, deterministic)


class RMFlowPolicy(MeanFlowPolicy):
    """One-step RMFlow policy with the paper's noise-injection refinement.

    This codebase uses the reverse-time convention (data at time 0 and the
    Gaussian prior at time 1).  Consequently, ``vnet(x, obs, 0, 1)`` moves a
    prior sample to the intermediate action via ``x - u``.  RMFlow then adds
    Gaussian noise with standard deviation ``sqrt(sigma_min**2 - sigma**2)``.
    """

    def __init__(
        self,
        vnet: nn.Module,
        act_dim: int,
        act_max: list[float],
        act_min: list[float],
        sigma: float = 0.0,
        sigma_min: float = 1e-3,
    ):
        super().__init__(vnet, act_dim, act_max, act_min)

        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if sigma_min <= 0:
            raise ValueError("sigma_min must be positive.")
        if sigma >= sigma_min:
            raise ValueError("sigma must be smaller than sigma_min.")

        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))
        self.register_buffer(
            "sigma_min", torch.as_tensor(sigma_min, dtype=torch.float32)
        )
        self.register_buffer(
            "refinement_std",
            torch.sqrt(self.sigma_min.square() - self.sigma.square()),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor | Tuple[torch.Tensor],
        shape: Tuple[int],
        num_steps: int = 1,
        noise: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        refinement_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_steps != 1:
            raise ValueError("RMFlowPolicy supports only 1-NFE sampling.")

        batch_size = shape[0]
        if isinstance(obs, tuple):
            device = obs[0].device
        else:
            device = obs.device

        expected_shape = (batch_size, self.act_dim)
        if noise is None:
            prior = (
                torch.zeros(expected_shape, device=device)
                if deterministic
                else torch.randn(expected_shape, device=device)
            )
        else:
            if noise.shape != expected_shape:
                raise ValueError(
                    f"noise must have shape [{batch_size}, {self.act_dim}]."
                )
            prior = noise.to(device=device, dtype=torch.float32)

        r = torch.zeros(batch_size, device=device, dtype=prior.dtype)
        t = torch.ones(batch_size, device=device, dtype=prior.dtype)
        intermediate = prior - self.vnet(prior, obs, r, t)

        if refinement_noise is not None:
            if refinement_noise.shape != expected_shape:
                raise ValueError(
                    f"refinement_noise must have shape [{batch_size}, {self.act_dim}]."
                )
            eps = refinement_noise.to(device=device, dtype=prior.dtype)
        elif deterministic:
            eps = torch.zeros_like(intermediate)
        else:
            eps = torch.randn_like(intermediate)

        refined = intermediate + self.refinement_std.to(prior.dtype) * eps
        return tensor_clamp(refined, self.act_max, self.act_min)

    def sample(
        self,
        obs: torch.Tensor | Tuple[torch.Tensor],
        shape: Tuple[int],
        num_steps: int = 1,
        noise: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        refinement_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            obs,
            shape,
            num_steps,
            noise,
            deterministic,
            refinement_noise,
        )
