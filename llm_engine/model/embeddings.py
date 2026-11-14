from __future__ import annotations

import torch
from torch import nn


class GPTEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, past_length: int = 0) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        # With a KV cache the new tokens continue at position past_length, not 0.
        positions = torch.arange(past_length, past_length + sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(positions)
        embeddings = token_embeddings + position_embeddings
        return self.dropout(embeddings)