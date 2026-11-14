from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from llm_engine.data.tokenizer import ByteTokenizer
from llm_engine.model.gpt import GPTConfig, GPTModel
from llm_engine.training.checkpoint import load_checkpoint


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> int:
    """Sample a token id from the last-position logits. temperature=0 means greedy (argmax)."""
    if temperature < 0:
        raise ValueError("temperature must be greater than or equal to 0")
    if temperature == 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k=top_k)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probabilities = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return int(next_token.item())


def stream_token_ids(
    model: GPTModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    use_kv_cache: bool = True,
) -> Iterator[int]:
    """Yield each newly sampled token id as soon as it is produced.

    Stops after max_new_tokens or after yielding the EOS token.

    With use_kv_cache, the prompt is processed once (prefill) and every later step feeds
    only the newest token, reusing the cached keys/values. Once the sequence outgrows
    context_length the window slides and cached positions go stale, so generation
    falls back to recomputing the window each step (same as use_kv_cache=False).
    """
    # An empty prompt is seeded with EOS so the model has something to condition on.
    generated = tokenizer.encode(prompt, add_eos=False) or [tokenizer.eos_token_id]
    context_length = model.config.context_length
    model.eval()
    past_kvs = None
    for _ in range(max_new_tokens):
        # Scope inference_mode to a single step, never across a yield: the caller may
        # resume this generator from a different thread (e.g. a web server threadpool).
        with torch.inference_mode():
            if use_kv_cache and len(generated) <= context_length:
                new_ids = generated if past_kvs is None else generated[-1:]
                input_ids = torch.tensor([new_ids], dtype=torch.long, device=device)
                logits, past_kvs = model.forward_with_cache(input_ids, past_kvs)
            else:
                past_kvs = None
                input_ids = torch.tensor([generated[-context_length:]], dtype=torch.long, device=device)
                logits, _ = model(input_ids)
            next_token_id = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        generated.append(next_token_id)
        yield next_token_id
        if next_token_id == tokenizer.eos_token_id:
            break


def generate_token_ids(
    model: GPTModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    use_kv_cache: bool = True,
) -> list[int]:
    prompt_token_ids = tokenizer.encode(prompt, add_eos=False)
    new_token_ids = stream_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_kv_cache=use_kv_cache,
    )
    return prompt_token_ids + list(new_token_ids)


def decode_generated_text(tokenizer: ByteTokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids)


def load_model_for_generation(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GPTModel, ByteTokenizer, dict[str, Any]]:
    tokenizer = ByteTokenizer()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device, trusted=False, load_optimizer=False)
    model_config = GPTConfig.from_dict(checkpoint["model_config"])
    model = GPTModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_metadata = checkpoint.get("tokenizer_metadata", {})
    if tokenizer_metadata.get("vocab_size") not in (None, tokenizer.vocab_size):
        raise ValueError("checkpoint tokenizer metadata does not match ByteTokenizer vocabulary")

    return model, tokenizer, checkpoint