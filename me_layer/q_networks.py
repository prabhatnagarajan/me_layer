# MIT License
# Copyright (c) 2025 Brett Daley and Prabhat Nagarajan
# Copyright (c) 2026 Prabhat Nagarajan
#
# Portions derived from https://github.com/prabhatnagarajan/reg-duel-q:
# `RDQNetwork.forward`, `RDQNetwork.forward_q`, `RDQNetwork.forward_v`

import torch
import torch.nn as nn

from pfrl.q_functions import DiscreteActionValueHead
from pfrl.nn.mlp import MLP
from pfrl.initializers import init_chainer_default



def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class MeanExpansionLayer(nn.Module):
    def __init__(self, mean_scaling_coefficient):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1 + mean_scaling_coefficient))

    def forward(self, vec):
        mean = vec.mean(dim=-1, keepdim=True)
        residual = vec - mean
        output = self.scale * mean + residual
        return output


class RDQNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        n_input_channels=4,
        activation=torch.nn.functional.relu,
        bias=0.1,
    ):
        super(RDQNetwork, self).__init__()
        self.activation = activation
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    n_input_channels,
                    32,
                    8,
                    stride=4,
                ),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        self.conv_layers.apply(init_chainer_default)  # MLP already applies
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

        self.a_stream = MLP(3136, n_actions, [512], nonlinearity=activation)
        self.v_stream = MLP(3136, 1, [512], nonlinearity=activation)
        self.av_head = DiscreteActionValueHead()

    def forward(self, x, return_adv=False):
        h = x
        for layer in self.conv_layers:
            h = self.activation(layer(h))
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        adv = self.a_stream(h)
        value = self.v_stream(h)
        returns = (self.av_head(adv + value), self.av_head(value))
        if return_adv:
            returns = (*returns, self.av_head(adv))
        return returns

    def forward_q(self, x, **kwargs):
        q, _ = self.forward(x, **kwargs)
        return q

    def forward_v(self, x, **kwargs):
        _, v = self.forward(x, **kwargs)
        return v
