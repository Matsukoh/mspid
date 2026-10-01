"""QSM networks using the current (robot, humans) observation convention."""

import torch
import torch.nn as nn
from diffusion.helpers import SinusoidalPosEmb
from utils.model_utils import make_mlp


class DMLP(nn.Module):
    """
    MLP for Diffusion model
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        h_dims=[256, 256, 256],
        activation="mish",
        aggregator=None,
        t_dim=16,
    ):
        super(DMLP, self).__init__()
        self.aggregator = aggregator

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = make_mlp(
            [input_dim] + h_dims, activation=activation, last_act=True
        )

        self.final_layer = nn.Linear(h_dims[-1], action_dim)

    def forward(self, x, time, data):
        if self.aggregator is not None:
            data = self.aggregator(*data)

        t = self.time_mlp(time)
        x = torch.cat([x, t, data], dim=-1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class SocialCritic(nn.Module):
    def __init__(
        self, D, d, h_dims=[256], aggregator=None, activation="mish", single=False
    ):
        super().__init__()
        self.aggregator = aggregator
        self.single = single
        self.net1 = make_mlp([D] + h_dims + [d], activation=activation, last_act=False)

        if not single:
            self.net2 = make_mlp(
                [D] + h_dims + [d], activation=activation, last_act=False
            )

    def forward(self, obs, act=None):
        data = self.aggregator(*obs) if self.aggregator is not None else obs

        if not self.single:
            data = torch.cat([data, act], -1)

        out1 = self.net1(data)

        if not self.single:
            out2 = self.net2(data)
            return out1, out2
        else:
            return out1
