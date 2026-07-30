import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_heads=8,
        dropout_rate=0.0,
        alpha=0.2,
        concat=True,
        share_weights=False,
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        if concat:
            assert out_features % n_heads == 0
            self.h_dim = out_features // n_heads
        else:
            self.h_dim = out_features

        self.linear_l = nn.Linear(in_features, self.h_dim * n_heads, bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.h_dim * n_heads, bias=False)

        self.attn = nn.Linear(self.h_dim, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h):
        n_batches = h.shape[0]
        n_nodes = h.shape[1]

        g_l = self.linear_l(h).view(n_batches, n_nodes, self.n_heads, self.h_dim)
        g_r = self.linear_r(h).view(n_batches, n_nodes, self.n_heads, self.h_dim)

        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)

        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)

        g_sum = g_l_repeat + g_r_repeat_interleave

        g_sum = g_sum.view(n_batches, n_nodes, n_nodes, self.n_heads, self.h_dim)

        e = self.attn(self.activation(g_sum))

        e = e.squeeze(-1)

        a = self.softmax(e)

        a = self.dropout(a)

        attn_res = torch.einsum("bijh,bjhf->bihf", a, g_r)

        if self.concat:
            return attn_res.reshape(n_batches, n_nodes, self.n_heads * self.h_dim)
        else:
            return attn_res.mean(dim=1)
