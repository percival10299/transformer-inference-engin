from pathlib import Path

import torch

from llm_engine.data.dataset import TokenDataset
from llm_engine.data.tokenizer import ByteTokenizer
from llm_engine.model.gpt import GPTConfig, GPTModel
from llm_engine.training.checkpoint import load_checkpoint, save_checkpoint


def build_test_model() -> GPTModel:
    config = GPTConfig(
        vocab_size=257,
        context_length=8,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    return GPTModel(config)


def test_single_optimizer_step_executes() -> None:
    model = build_test_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 257, (4, 8))
    targets = torch.randint(0, 257, (4, 8))

    _, loss = model(input_ids, targets=targets)
    assert loss is not None

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


def test_checkpoint_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    model = build_test_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokenizer = ByteTokenizer()
    checkpoint_path = tmp_path / "transformer.pt"
    input_ids = torch.randint(0, 257, (2, 8))

    model.eval()
    with torch.inference_mode():
        original_logits, _ = model(input_ids)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=3,
        model_config=model.config.to_dict(),
        tokenizer_metadata=tokenizer.metadata(),
    )

    restored_model = build_test_model()
    load_checkpoint(checkpoint_path, model=restored_model, trusted=False, load_optimizer=False)
    restored_model.eval()
    with torch.inference_mode():
        restored_logits, _ = restored_model(input_ids)

    assert torch.allclose(original_logits, restored_logits, atol=1e-6)


def test_token_dataset_batch_shapes() -> None:
    token_ids = torch.arange(0, 64, dtype=torch.long) % 257
    dataset = TokenDataset(token_ids=token_ids, context_length=8, seed=42)

    x, y = dataset.get_batch(batch_size=4, device=torch.device("cpu"))

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])