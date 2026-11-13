from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from llm_engine.model.embeddings import GPTEmbeddings
from llm_engine.model.transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
        fields = {key: data[key] for key in cls.__dataclass_fields__}
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    context_length=config.context_length,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("input sequence length exceeds context_length")

        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss