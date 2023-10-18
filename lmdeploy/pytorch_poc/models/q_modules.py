# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..kernels.triton_kernels import (linear_dynamic_quant_triton_op_fast,
                                      per_channel_quant,
                                      per_token_quant_int8_tri,
                                      rms_norm_dynamic_quant)


@dataclass
class QTensor:
    tensor: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.tensor, name)


class QRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            hidden_states, self.weight, self.variance_epsilon)
        return QTensor(hidden_states_quant, rms_scale)


class QLinear(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            'weight',
            torch.empty((out_features, in_features),
                        device=device,
                        dtype=torch.int8))
        self.register_buffer('scale', None)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def quant(self, weight):
        weight_quant, scale = per_channel_quant(weight, 8, 1e-5, torch.int8)
        self.weight.data = weight_quant
        self.scale = scale
        self.fp_weight = weight
        return

    def forward(self, input):
        shape = input.shape

        if isinstance(input, torch.Tensor):
            # return F.linear(input, self.fp_weight.to(input.device))
            shape = input.shape
            input_quant, input_scale = per_token_quant_int8_tri(
                input.view(-1, shape[-1]), 1e-7)
        else:
            assert isinstance(input, QTensor)
            shape = input.tensor.shape
            input_quant, input_scale = input.tensor, input.scale
            input_quant = input_quant.view(-1, shape[-1])
        out = linear_dynamic_quant_triton_op_fast(input_quant,
                                                  self.weight,
                                                  input_scale,
                                                  self.scale,
                                                  output_dtype=torch.float16,
                                                  bias=self.bias)
        return out.view(*shape[:-1], -1)


@torch.no_grad()
def group_wise_fake_quant(w: torch.Tensor, n_bits: int):
    """Calculate quantization parameters for each group using min and max
    values."""

    w_min = w.min(dim=-1, keepdim=True)[0]
    w_max = w.max(dim=-1, keepdim=True)[0]

    q_max = 2**n_bits - 1
    scales = (w_max - w_min)

    scales = scales.div_(q_max)
    zero_points = (-w_min / scales).round()

    int_w = ((w - w_min) / scales).round()
    fp_w = (int_w - zero_points) * scales

    return fp_w
