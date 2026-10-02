import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.model_utils import make_mlp


class SocialGaussianPolicy(nn.Module):
    """Reparameterized tanh Gaussian with per-dimension action bounds."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        aggregator,
        h_dims=(256, 256),
        act_min=(-1.0, -1.0),
        act_max=(1.0, 1.0),
        log_std_min=-20.0,
        log_std_max=2.0,
        deterministic_eval=True,
        p_norm=False,
    ):
        super().__init__()
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be below log_std_max")
        lower = torch.as_tensor(act_min, dtype=torch.float32)
        upper = torch.as_tensor(act_max, dtype=torch.float32)
        if (
            lower.shape != (act_dim,)
            or upper.shape != (act_dim,)
            or not torch.all(lower < upper)
        ):
            raise ValueError("Action bounds must be ordered vectors of length act_dim")
        self.register_buffer("act_min", lower)
        self.register_buffer("act_max", upper)
        self.register_buffer("action_scale", (upper - lower) / 2)
        self.register_buffer("action_bias", (upper + lower) / 2)
        self.act_dim = act_dim
        self.aggregator = aggregator
        self.net = make_mlp(
            [obs_dim] + list(h_dims), activation="relu", last_act=True, layer_norm=True
        )
        self.mean = nn.Linear(h_dims[-1], act_dim)
        self.log_std = nn.Linear(h_dims[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.deterministic_eval = deterministic_eval
        self.p_norm = p_norm

    def distribution(self, obs):
        features = self.net(self.aggregator(*obs))
        if self.p_norm:
            features_norm = torch.norm(features, dim=1).view((-1, 1))
            features = features / features_norm
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample_with_log_prob(self, obs):
        distribution = self.distribution(obs)
        raw_action = distribution.rsample()
        action = self.action_bias + self.action_scale * raw_action.tanh()
        # Stable log(1 - tanh(x)^2), including the affine action transform.
        log_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        log_prob = (
            distribution.log_prob(raw_action) - log_jacobian - self.action_scale.log()
        ).sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def sample(self, obs, shape=None):
        if shape is not None and tuple(shape) != (obs[0].shape[0], self.act_dim):
            raise ValueError(
                "shape must match the observation batch and action dimension"
            )
        if not self.training and self.deterministic_eval:
            return (
                self.action_bias
                + self.action_scale * self.distribution(obs).mean.tanh()
            )
        return self.sample_with_log_prob(obs)[0]

    def forward(self, obs):
        return self.sample_with_log_prob(obs)


class SocialSACCritic(nn.Module):
    """Twin Q heads with a shared GAT observation encoder."""

    def __init__(self, obs_dim, act_dim, aggregator, h_dims=(256, 256)):
        super().__init__()
        self.aggregator = aggregator
        dims = [obs_dim + act_dim] + list(h_dims) + [1]
        self.q1 = make_mlp(dims, activation="relu")
        self.q2 = make_mlp(dims, activation="relu")

    def forward(self, obs, actions):
        inputs = torch.cat((self.aggregator(*obs), actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)
