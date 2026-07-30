import math
from dataclasses import dataclass
from math import log
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import SinusoidalTimeEmbedding, make_mlp, tensor_clamp


class SocialNoiseConditionedCritic(nn.Module):
    def __init__(
        self,
        D: int,
        d: int,
        h_dims: list[int] = [256],
        time_dim: int = 16,
        aggregator: nn.Module = None,
        activation: str = "silu",
    ):
        super().__init__()
        self.aggregator = aggregator

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
        if self.aggregator is not None:
            obs = self.aggregator(*obs)
        t = self.time_mlp(i).reshape((-1, self.time_dim))
        data = torch.cat([obs, act, t], -1)

        phi1 = self.net1(data)
        phi1 = phi1 / (phi1.norm(dim=-1, keepdim=True) + 1e-8)
        out1 = self.final_layer1(phi1)

        phi2 = self.net2(data)
        phi2 = phi2 / (phi2.norm(dim=-1, keepdim=True) + 1e-8)
        out2 = self.final_layer1(phi2)
        return out1, out2


class SocialConditionalMeanVelocityNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        h_dim: int = 256,
        time_dim: int = 16,
        aggregator: nn.Module = None,
    ):
        super().__init__()

        self.aggregator = aggregator

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
        self,
        x_t: torch.Tensor,
        obs: torch.Tensor | Tuple[torch.Tensor],
        r: torch.Tensor,
        t: torch.Tensor,
    ):
        if self.aggregator is not None:
            obs = self.aggregator(*obs)

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
