from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_engine.data.tokenizer import ByteStreamDecoder, ByteTokenizer
from llm_engine.device import resolve_device
from llm_engine.generation.generator import load_model_for_generation, stream_token_ids
from llm_engine.model.gpt import GPTModel


@dataclass
class LoadedModel:
    model: GPTModel
    tokenizer: ByteTokenizer
    device: torch.device


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=1.0, gt=0, le=5.0)
    top_k: int | None = Field(default=None, ge=1)


def load_model_from_env() -> LoadedModel:
    checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", "checkpoints/tiny_gpt.pt"))
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found at {checkpoint_path}. Train a model first or set CHECKPOINT_PATH.")
    device = resolve_device(os.environ.get("DEVICE", "auto"))
    model, tokenizer, _ = load_model_for_generation(checkpoint_path, device=device)
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def sse_event(payload: dict[str, str] | str) -> str:
    # JSON-encode so newlines inside generated text cannot break SSE framing.
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def stream_events(loaded: LoadedModel, request: GenerateRequest) -> Iterator[str]:
    decoder = ByteStreamDecoder()
    token_ids = stream_token_ids(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt=request.prompt,
        device=loaded.device,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    )
    for token_id in token_ids:
        text = decoder.push(token_id)
        if text:
            yield sse_event({"token": text})
    tail = decoder.flush()
    if tail:
        yield sse_event({"token": tail})
    yield sse_event("[DONE]")


def create_app(model_loader: Callable[[], LoadedModel] = load_model_from_env) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loaded = model_loader()
        yield

    app = FastAPI(title="Transformer Inference Engine", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate/stream")
    def generate_stream(body: GenerateRequest, request: Request) -> StreamingResponse:
        # A plain (sync) generator is iterated in a threadpool, so the blocking model
        # forward pass does not freeze the event loop.
        return StreamingResponse(
            stream_events(request.app.state.loaded, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
