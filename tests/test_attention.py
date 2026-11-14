import torch

from llm_engine.model.attention import CausalSelfAttention


def test_attention_output_shape() -> None:
    attention = CausalSelfAttention(d_model=16, n_heads=4, context_length=8, dropout=0.0)
    x = torch.randn(2, 8, 16)

    output, _ = attention(x)

    assert output.shape == (2, 8, 16)


def test_attention_causal_mask_prevents_future_leakage() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=12, n_heads=3, context_length=6, dropout=0.0)
    attention.eval()

    base = torch.randn(1, 6, 12)
    changed = base.clone()
    changed[:, 4:, :] = torch.randn(1, 2, 12)

    base_output, _ = attention(base)
    changed_output, _ = attention(changed)

    assert torch.allclose(base_output[:, :4, :], changed_output[:, :4, :], atol=1e-6)