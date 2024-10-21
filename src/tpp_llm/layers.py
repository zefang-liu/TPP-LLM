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
    Temporal Positional Encoding from THP
    """

    def __init__(self, embedding_dim: int, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the temporal encoding

        :param embedding_dim: embedding dimension
        :param dtype: data type for the output
        :param device: device
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        i = torch.arange(0, embedding_dim, 1, dtype=self.dtype, device=self.device)
        div_term = (2 * (i // 2).to(self.dtype) * -(math.log(10000.0) / embedding_dim)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, event_times: Tensor) -> Tensor:
        """
        Compute time positional encoding defined in the THP model

        :param event_times: event times, (seq_len, 1)
        :return: temporal encoding vector, (seq_len, embedding_dim)
        """
        result = (event_times * self.div_term).to(self.dtype).to(self.device)
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result


class TimeShiftedPositionalEncoding(nn.Module):
    """
    Time-Shifted Positional Encoding from SAHP
    """

    def __init__(
        self, embedding_dim: int, max_len: int = 5000, dtype=torch.float32, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the temporal encoding

        :param embedding_dim: embedding dimension
        :param max_len: maximum sequence length
        :param dtype: data type for the output
        :param device: device
        """
        super().__init__()
        self.dtype = dtype
        self.device = device

        position = torch.arange(0, max_len, dtype=self.dtype, device=self.device).unsqueeze(1)  # (max_len, 1)
        div_term = (
            torch.arange(0, embedding_dim, 2, dtype=self.dtype, device=self.device)
            * -(math.log(10000.0) / embedding_dim)
        ).exp()  # (model_dim // 2, )
        self.layer_time_delta = nn.Linear(1, embedding_dim // 2, bias=False, dtype=self.dtype, device=self.device)

        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, event_times: Tensor, event_time_deltas: Tensor) -> Tensor:
        """
        Compute time-shifted positional encoding defined in the SAHP model

        :param event_times: event times, (seq_len, 1)
        :param event_time_deltas: event time deltas, (seq_len, 1)
        :return: temporal encoding vector, (seq_len, embedding_dim)
        """
        phi = self.layer_time_delta(event_time_deltas)
        length = event_times.size(0)
        arc = (self.position[:length] * self.div_term).to(self.dtype).to(self.device)
        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe
