# -*- coding: utf-8 -*-
"""LSTM module for model of algorithms

- Author: whikwon
- Contact: whikwon@gmail.com
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.helper_functions import identity


class LSTM(nn.Module):
    """Baseline of Multilayer perceptron.

    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = identity,
        use_output_layer: bool = True,
        init_w: float = 3e-3,
    ):
        """Initialization.

        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            use_output_layer (bool): whether or not to use the last layer
            init_w (float): weight initialization bound for the last layer

        """
        super(LSTM, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer

        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            lstm = nn.LSTM(in_size, next_size, batch_first=True)
            in_size = next_size
            self.add_module("hidden_lstm{}".format(i), lstm)
            self.hidden_layers.append(lstm)

        # set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, output_size)
            self.output_layer.weight.data.uniform_(-init_w, init_w)
            self.output_layer.bias.data.uniform_(-init_w, init_w)

    def get_last_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the activation of the last hidden layer."""
        for hidden_layer in self.hidden_layers:
            x, _ = hidden_layer(x)
            x = self.hidden_activation(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        assert self.use_output_layer

        x = self.get_last_activation(x)

        output = self.output_layer(x)
        output = self.output_activation(output)

        return output
