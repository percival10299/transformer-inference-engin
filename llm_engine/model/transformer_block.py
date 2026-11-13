from __future__ import annotations

import torch
from torch import nn

from llm_engine.model.attention import CausalSelfAttention
from llm_engine.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.feedforward(self.ln_2(x))
        return x