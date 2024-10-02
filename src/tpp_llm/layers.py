"""
TPP-LLM Layers
"""
import math
from typing import Union

import torch
from torch import Tensor
from torch import nn


class TimePositionalEncoding(nn.Module):
    """
    Temporal Positional Encoding from the THP
    """

    def __init__(self, embedding_dim: int, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the temporal encoding

        :param embedding_dim: embedding dimension
        :param device: device
        """
        super().__init__()
        i = torch.arange(0, embedding_dim, 1, device=device)
        div_term = (2 * (i // 2).float() * -(math.log(10000.0) / embedding_dim)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, event_times: Tensor) -> Tensor:
        """
        Compute time positional encoding defined in THP model

        :param event_times: event times, (seq_len, 1)
        :return: temporal encoding vector, (seq_len, embedding_dim)
        """
        result = event_times * self.div_term
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result
