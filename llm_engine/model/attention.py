from __future__ import annotations

import math

import torch
from torch import nn

# Cached keys and values for one layer, each shaped [B, n_heads, T_cached, head_dim].
KVCache = tuple[torch.Tensor, torch.Tensor]


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

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        # x has shape [B, T, C] where:
        # B = batch size, T = number of NEW tokens, C = d_model.
        # Without a cache T is the whole sequence. With a cache, T is 1 while decoding.
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

        # Keys and values of earlier tokens never change, so reuse them instead of
        # recomputing. Only the new tokens' K/V were projected above.
        if past_kv is not None:
            past_key, past_value = past_kv
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        present_kv = (key, value) if use_cache else None

        total_length = key.size(2)
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # New queries sit at absolute positions [total_length - T, total_length), so take
        # those mask rows (not rows 0..T) and let them see every key up to their position.
        mask = self.causal_mask[:, :, total_length - sequence_length : total_length, :total_length]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        attended = attention_weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        output = self.out_proj(attended)
        return self.resid_dropout(output), present_kv