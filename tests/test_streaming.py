import json

import pytest
import torch
from fastapi.testclient import TestClient

from llm_engine.data.tokenizer import ByteStreamDecoder, ByteTokenizer
from llm_engine.generation.generator import generate_token_ids, stream_token_ids
from llm_engine.model.gpt import GPTConfig, GPTModel
from llm_engine.server.app import LoadedModel, create_app


def build_loaded_model() -> LoadedModel:
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=257, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    return LoadedModel(model=GPTModel(config), tokenizer=ByteTokenizer(), device=torch.device("cpu"))


@pytest.fixture
def client():
    app = create_app(model_loader=build_loaded_model)
    with TestClient(app) as test_client:
        yield test_client


def parse_events(body: str) -> list[str]:
    events = [block for block in body.split("\n\n") if block]
    assert all(block.startswith("data: ") for block in events)
    return [block.removeprefix("data: ") for block in events]


def test_stream_token_ids_yields_incrementally_and_respects_limit() -> None:
    loaded = build_loaded_model()
    stream = stream_token_ids(
        loaded.model, loaded.tokenizer, "hi", loaded.device, max_new_tokens=5, temperature=1.0, top_k=10
    )

    assert hasattr(stream, "__next__")  # a lazy generator, not a finished list
    first = next(stream)
    assert 0 <= first <= 256
    assert len(list(stream)) <= 4


def test_generate_token_ids_is_prompt_plus_streamed_tokens() -> None:
    loaded = build_loaded_model()
    kwargs = dict(device=loaded.device, max_new_tokens=6, temperature=1.0, top_k=5)

    torch.manual_seed(1)
    streamed = list(stream_token_ids(loaded.model, loaded.tokenizer, "hi", **kwargs))
    torch.manual_seed(1)
    full = generate_token_ids(loaded.model, loaded.tokenizer, "hi", **kwargs)

    assert full == list(b"hi") + streamed


def test_byte_stream_decoder_buffers_partial_utf8_characters() -> None:
    decoder = ByteStreamDecoder()
    emoji_bytes = "é".encode("utf-8")  # two bytes

    assert decoder.push(emoji_bytes[0]) == ""
    assert decoder.push(emoji_bytes[1]) == "é"
    assert decoder.push(256) == ""  # EOS is not text
    assert decoder.flush() == ""


def test_stream_endpoint_returns_incremental_sse_events(client: TestClient) -> None:
    with client.stream("POST", "/generate/stream", json={"prompt": "hi", "max_new_tokens": 8}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(e) if e != "[DONE]" else e for e in parse_events("".join(response.iter_text()))]

    assert events[-1] == "[DONE]"
    assert all("token" in event for event in events[:-1])


def test_stream_endpoint_accepts_empty_prompt(client: TestClient) -> None:
    response = client.post("/generate/stream", json={"max_new_tokens": 3})

    assert response.status_code == 200
    assert parse_events(response.text)[-1] == "[DONE]"


@pytest.mark.parametrize(
    "payload",
    [{"max_new_tokens": 0}, {"max_new_tokens": 100000}, {"temperature": 0}, {"top_k": 0}],
)
def test_stream_endpoint_validates_request(client: TestClient, payload: dict) -> None:
    response = client.post("/generate/stream", json=payload)

    assert response.status_code == 422


def test_health_endpoint(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
