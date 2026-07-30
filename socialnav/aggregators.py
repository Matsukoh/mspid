import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.nn.functional import softmax

from socialnav.layers import GraphAttentionLayer
from utils.model_utils import make_mlp


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class GATAggregator(nn.Module):
    def __init__(
        self,
        obs_dim,
        r_obs_dim,
        projection_dim,
        enc_hdims=[64],
        concat=True,
        n_heads=8,
        prediction=False,
        dropout_rate=0.0,
        alpha=0.2,
    ):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        if concat:
            assert projection_dim % n_heads == 0
            self.h_dim = projection_dim // n_heads
        else:
            self.h_dim = projection_dim

        self.projection_dim = projection_dim

        self.enc_r_obs = make_mlp(
            mlp_dims=[r_obs_dim] + enc_hdims + [projection_dim], last_act=True
        )
        self.enc_obs = make_mlp(
            mlp_dims=[obs_dim] + enc_hdims + [projection_dim], last_act=True
        )
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim

        self.gat1 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        self.gat2 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

    def forward(self, r_obs, h_obs):
        n = h_obs.shape[0]
        p_num = h_obs.shape[1]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(h_obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )

        obs_gat1 = self.gat1(obs_stack) + obs_stack
        obs_gat2 = self.gat2(obs_gat1) + obs_gat1
        if self.prediction:
            aggr = obs_gat2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            aggr = obs_gat2[:, 0, :].reshape(-1, self.output_dim)

        return aggr
        # return obs_stack.reshape(n, -1)
