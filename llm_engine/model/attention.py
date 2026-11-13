from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, context_length, context_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape [B, T, C] where:
        # B = batch size, T = sequence length, C = d_model.
        batch_size, sequence_length, channels = x.shape

        # After projection each tensor is still [B, T, C].
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Reshape to [B, T, n_heads, head_dim] then transpose to
        # [B, n_heads, T, head_dim] so attention is computed independently per head.
        query = query.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        attended = attention_weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        output = self.out_proj(attended)
        return self.resid_dropout(output)