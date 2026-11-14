from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from llm_engine.model.attention import KVCache
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

    def _run(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[KVCache] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        past_length = 0 if past_kvs is None else past_kvs[0][0].size(2)
        if past_length + input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence length (cached + new tokens) exceeds context_length")

        x = self.embeddings(input_ids, past_length=past_length)
        presents: list[KVCache] = []
        for layer_index, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[layer_index]
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if present_kv is not None:
                presents.append(present_kv)
        x = self.final_layer_norm(x)
        return self.lm_head(x), (presents if use_cache else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, _ = self._run(input_ids, past_kvs=None, use_cache=False)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Run only `input_ids` and return (logits, updated per-layer KV cache).

        Prefill: pass the whole prompt with past_kvs=None.
        Decode: pass just the newest token together with the previous cache.
        """
        logits, presents = self._run(input_ids, past_kvs=past_kvs, use_cache=True)
        assert presents is not None
        return logits, presents