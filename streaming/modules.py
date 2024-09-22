from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import EinMix
from streamvc._utils import auto_batching
from streamvc.modules import CausalConv1d, CausalConvTranspose1d


class StreamingCausalConv1d(nn.Module):
    """CausalConv1d with streaming support"""
    def __init__(self, base: CausalConv1d):
        super().__init__()

        # NOTE: If you want to implement this calss in C++, you should make the same checks
        # that are made in the original class

        self.conv = base.conv
        self.in_channels = base.in_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.dilation = base.dilation
        self.causal_padding = base.causal_padding
        self.padding_mode = base.padding_mode

        self.register_buffer('streaming_buffer',
                             torch.tensor([]), persistent=False)
        self.init_streaming_buffer()

    def init_streaming_buffer(self):
        self.streaming_buffer = torch.zeros(
            self.in_channels, self.causal_padding)

    def remove_streaming_buffer(self):
        self.streaming_buffer = torch.tensor([])

    def forward(self, x):
        if self.streaming_buffer.numel() == 0:
            full_input = x
        else:
            full_input = torch.cat([self.streaming_buffer, x], dim=-1)

        num_samples = full_input.shape[-1]
        kernel_reception_field = self.dilation * (self.kernel_size - 1) + 1
        num_strides = (num_samples - kernel_reception_field) // self.stride + 1
        num_elements_for_forward = kernel_reception_field + \
            (num_strides - 1) * self.stride
        ready_input = full_input[..., :num_elements_for_forward]
        new_buffer_size = num_samples - num_strides * self.stride
        self.streaming_buffer = full_input[..., -new_buffer_size:]
        return self.conv.forward(ready_input)
