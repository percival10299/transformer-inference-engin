import torch

from llm_engine.generation.generator import generate_token_ids
from llm_engine.model.embeddings import GPTEmbeddings
from llm_engine.model.gpt import GPTConfig, GPTModel


def build_test_model() -> GPTModel:
    config = GPTConfig(
        vocab_size=257,
        context_length=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    return GPTModel(config)


def test_embeddings_shape() -> None:
    embeddings = GPTEmbeddings(vocab_size=257, context_length=16, d_model=32, dropout=0.0)
    input_ids = torch.randint(0, 257, (4, 16))

    output = embeddings(input_ids)

    assert output.shape == (4, 16, 32)


def test_model_forward_shapes_and_finite_loss() -> None:
    model = build_test_model()
    input_ids = torch.randint(0, 257, (2, 16))
    targets = torch.randint(0, 257, (2, 16))

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (2, 16, 257)
    assert loss is not None
    assert torch.isfinite(loss)


def test_gradients_flow_through_model() -> None:
    model = build_test_model()
    input_ids = torch.randint(0, 257, (2, 16))
    targets = torch.randint(0, 257, (2, 16))

    _, loss = model(input_ids, targets=targets)
    assert loss is not None
    loss.backward()

    grad_norm = model.embeddings.token_embeddings.weight.grad.norm().item()
    assert grad_norm > 0


def test_model_causal_mask_blocks_future_token_changes() -> None:
    torch.manual_seed(7)
    model = build_test_model()
    model.eval()

    prefix = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    seq_a = torch.cat([prefix, torch.tensor([[9, 10, 11, 12, 13, 14, 15, 16]])], dim=1)
    seq_b = torch.cat([prefix, torch.tensor([[100, 101, 102, 103, 104, 105, 106, 107]])], dim=1)

    logits_a, _ = model(seq_a)
    logits_b, _ = model(seq_b)

    assert torch.allclose(logits_a[:, :8, :], logits_b[:, :8, :], atol=1e-6)


def test_generation_returns_valid_token_ids() -> None:
    model = build_test_model()
    token_ids = generate_token_ids(
        model=model,
        tokenizer=type("Tokenizer", (), {"encode": lambda self, text, add_eos=False: [1, 2], "eos_token_id": 256})(),
        prompt="hi",
        device=torch.device("cpu"),
        max_new_tokens=5,
        temperature=1.0,
        top_k=10,
    )

    assert token_ids
    assert all(0 <= token_id <= 256 for token_id in token_ids)