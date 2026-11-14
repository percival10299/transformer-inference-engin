from __future__ import annotations

import torch
from torch import nn

from llm_engine.model.attention import CausalSelfAttention, KVCache
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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attended, present_kv = self.attention(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attended
        x = x + self.feedforward(self.ln_2(x))
        return x, present_kv