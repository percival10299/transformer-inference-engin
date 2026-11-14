import pytest
import torch

from llm_engine.data.tokenizer import ByteTokenizer
from llm_engine.generation.generator import generate_token_ids, sample_next_token
from llm_engine.model.attention import CausalSelfAttention
from llm_engine.model.gpt import GPTConfig, GPTModel

CONTEXT_LENGTH = 16


class NoEosTokenizer(ByteTokenizer):
    """Never matches a sampled id, so generation always runs for max_new_tokens."""

    eos_token_id = -1


def build_model() -> GPTModel:
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=257, context_length=CONTEXT_LENGTH, d_model=32, n_heads=4, n_layers=2, dropout=0.0
    )
    model = GPTModel(config)
    # Default init gives near-uniform logits, where float noise could flip an argmax tie.
    torch.nn.init.normal_(model.lm_head.weight, std=0.5)
    return model.eval()


def test_attention_incremental_matches_full_pass() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=16, n_heads=4, context_length=8, dropout=0.0).eval()
    x = torch.randn(2, 8, 16)

    full, _ = attention(x)

    prefill_out, cache = attention(x[:, :5], use_cache=True)
    pieces = [prefill_out]
    for t in range(5, 8):
        out, cache = attention(x[:, t : t + 1], past_kv=cache, use_cache=True)
        pieces.append(out)

    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5)


def test_attention_cache_grows_by_new_tokens() -> None:
    attention = CausalSelfAttention(d_model=16, n_heads=4, context_length=8, dropout=0.0).eval()

    _, cache = attention(torch.randn(2, 5, 16), use_cache=True)
    assert cache is not None
    assert cache[0].shape == cache[1].shape == (2, 4, 5, 4)  # [B, heads, T, head_dim]

    _, cache = attention(torch.randn(2, 1, 16), past_kv=cache, use_cache=True)
    assert cache is not None
    assert cache[0].shape == cache[1].shape == (2, 4, 6, 4)


def test_attention_without_use_cache_returns_no_cache() -> None:
    attention = CausalSelfAttention(d_model=16, n_heads=4, context_length=8, dropout=0.0)

    _, cache = attention(torch.randn(1, 4, 16))

    assert cache is None


def test_model_cached_logits_match_uncached_logits() -> None:
    model = build_model()
    input_ids = torch.randint(0, 257, (2, CONTEXT_LENGTH))

    with torch.inference_mode():
        full_logits, _ = model(input_ids)

        prompt_length = 6
        logits, cache = model.forward_with_cache(input_ids[:, :prompt_length])
        pieces = [logits]
        for t in range(prompt_length, CONTEXT_LENGTH):
            logits, cache = model.forward_with_cache(input_ids[:, t : t + 1], cache)
            pieces.append(logits)

    assert torch.allclose(full_logits, torch.cat(pieces, dim=1), atol=1e-5)
    assert len(cache) == model.config.n_layers
    assert cache[0][0].size(2) == CONTEXT_LENGTH


def test_cache_beyond_context_length_raises() -> None:
    model = build_model()
    _, cache = model.forward_with_cache(torch.randint(0, 257, (1, CONTEXT_LENGTH)))

    with pytest.raises(ValueError, match="exceeds context_length"):
        model.forward_with_cache(torch.randint(0, 257, (1, 1)), cache)


@pytest.mark.parametrize(
    ("prompt", "max_new_tokens"),
    [
        ("hello", 8),  # stays inside the context window
        ("hello", 11),  # ends exactly at context_length
        ("hello world", 20),  # outgrows the window, so the sliding fallback is used
    ],
)
def test_greedy_generation_is_identical_with_and_without_cache(prompt: str, max_new_tokens: int) -> None:
    model = build_model()
    tokenizer = NoEosTokenizer()
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=torch.device("cpu"),
        max_new_tokens=max_new_tokens,
        temperature=0,
    )

    uncached = generate_token_ids(**kwargs, use_kv_cache=False)
    cached = generate_token_ids(**kwargs, use_kv_cache=True)

    assert len(cached) == len(prompt) + max_new_tokens
    assert cached == uncached


def test_greedy_sampling_is_argmax_and_negative_temperature_is_rejected() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5]])

    assert sample_next_token(logits, temperature=0) == 1
    with pytest.raises(ValueError):
        sample_next_token(logits, temperature=-1.0)
