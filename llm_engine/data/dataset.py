from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from datasets import load_dataset

from llm_engine.data.tokenizer import ByteTokenizer


@dataclass
class TokenDataset:
    token_ids: torch.Tensor
    context_length: int
    seed: int

    def __post_init__(self) -> None:
        if self.token_ids.dtype != torch.long:
            self.token_ids = self.token_ids.to(torch.long)
        if self.token_ids.numel() <= self.context_length:
            raise ValueError("token stream must be longer than context_length")
        self._generator = torch.Generator().manual_seed(self.seed)

    @property
    def num_sequences(self) -> int:
        return self.token_ids.numel() - self.context_length

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.token_ids.numel() - self.context_length - 1
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=self._generator)

        x_batch = []
        y_batch = []
        for start in starts.tolist():
            x_batch.append(self.token_ids[start : start + self.context_length])
            y_batch.append(self.token_ids[start + 1 : start + self.context_length + 1])

        x = torch.stack(x_batch).to(device)
        y = torch.stack(y_batch).to(device)
        return x, y


def split_stories(texts: list[str], seed: int, val_fraction: float = 0.1) -> tuple[list[str], list[str]]:
    if not texts:
        raise ValueError("no texts were loaded from TinyStories")

    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_fraction))
    if val_size >= len(shuffled):
        val_size = max(1, len(shuffled) - 1)

    val_texts = shuffled[:val_size]
    train_texts = shuffled[val_size:]
    return train_texts, val_texts


def concatenate_texts(texts: list[str], tokenizer: ByteTokenizer) -> torch.Tensor:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text, add_eos=True))
    return torch.tensor(token_ids, dtype=torch.long)


def load_tinystories_texts(n_stories: int) -> list[str]:
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{n_stories}]")
    return [row["text"] for row in dataset]


def prepare_tinystories_datasets(
    n_stories: int,
    context_length: int,
    seed: int,
) -> tuple[TokenDataset, TokenDataset, ByteTokenizer]:
    tokenizer = ByteTokenizer()
    texts = load_tinystories_texts(n_stories=n_stories)
    train_texts, val_texts = split_stories(texts, seed=seed)

    train_tokens = concatenate_texts(train_texts, tokenizer)
    val_tokens = concatenate_texts(val_texts, tokenizer)

    train_dataset = TokenDataset(train_tokens, context_length=context_length, seed=seed)
    val_dataset = TokenDataset(val_tokens, context_length=context_length, seed=seed + 1)
    return train_dataset, val_dataset, tokenizer