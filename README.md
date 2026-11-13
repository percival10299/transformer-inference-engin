# Transformer Inference & LLM Serving Engine

This repository starts with Phase 1 of a larger portfolio project: a decoder-only GPT-style Transformer implemented from scratch in PyTorch, trained on TinyStories, saved to checkpoints, and used for basic autoregressive generation.

Later phases will add KV caching, an inference engine with continuous batching, a FastAPI server, and performance benchmarking.

## Phase 1 Scope

- GPT-style decoder-only Transformer written with PyTorch building blocks
- TinyStories data loading and byte-level tokenization
- next-token prediction training loop with validation
- checkpoint save and resume support
- autoregressive text generation without KV caching
- CPU-friendly unit tests that do not download TinyStories

## Project Layout

```text
configs/                 YAML hyperparameters
checkpoints/             saved model/optimizer state (gitignored)
tests/                   CPU-friendly unit tests
llm_engine/
  device.py              device selection (cuda / mps / cpu)
  model/                 GPT architecture (embeddings, attention, feedforward, block, gpt)
  data/                  ByteTokenizer and TinyStories dataset
  training/              Trainer and checkpoint save/load
  generation/            autoregressive sampling (incl. streaming) and checkpoint loading
  server/                FastAPI app with the /generate/stream endpoint
  cli/                   command-line entry points (train, generate)
```

## Architecture Overview

The model is implemented from first principles rather than using a pretrained Hugging Face Transformer class.

1. `llm_engine/model/embeddings.py`
   Combines token embeddings and learned positional embeddings for inputs of shape `[B, T]` into hidden states of shape `[B, T, C]`.
2. `llm_engine/model/attention.py`
   Implements causal multi-head self-attention explicitly: Q/K/V projections, head reshaping, scaled dot-product attention, causal masking, softmax, value aggregation, and output projection.
3. `llm_engine/model/feedforward.py`
   Implements the Transformer MLP block: `Linear -> GELU -> Linear -> Dropout`.
4. `llm_engine/model/transformer_block.py`
   Uses Pre-LayerNorm residual blocks:
   `x = x + attention(layernorm1(x))`
   `x = x + feedforward(layernorm2(x))`
5. `llm_engine/model/gpt.py`
   Stacks embeddings, decoder blocks, a final layer norm, and a vocabulary projection head. The forward pass optionally computes cross-entropy loss when targets are provided.

## Data and Training Modules

1. `llm_engine/data/tokenizer.py`
   Byte-level UTF-8 tokenizer with an EOS token.
2. `llm_engine/data/dataset.py`
   Loads `roneneldan/TinyStories`, splits stories into train and validation sets before tokenization, concatenates stories with EOS separators, and serves fixed-length next-token batches.
3. `llm_engine/training/trainer.py`
   Builds the model, optimizer, device selection, training loop, validation, metric logging, gradient clipping, and checkpointing.
4. `llm_engine/training/checkpoint.py`
   Saves model state, optimizer state, config, training step, and tokenizer metadata. Supports trusted resume loading versus inference-oriented loading.

## Generation Modules

1. `llm_engine/generation/generator.py`
   Loads a saved checkpoint, performs autoregressive sampling with temperature and optional top-k filtering, handles empty prompts, and decodes output text.
2. `llm_engine/cli/generate.py`
   CLI entrypoint for text generation from a saved checkpoint.

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run Unit Tests

```bash
python -m pytest tests/ -v
```

## Train The Model

```bash
python -m llm_engine.cli.train --config configs/tiny_gpt.yaml
```

To disable automatic checkpoint resume:

```bash
python -m llm_engine.cli.train --config configs/tiny_gpt.yaml --no-resume
```

## Resume Training

If `checkpoints/tiny_gpt.pt` exists, training resumes from it by default.

```bash
python -m llm_engine.cli.train --config configs/tiny_gpt.yaml --checkpoint checkpoints/tiny_gpt.pt
```

## Generate Text

```bash
python -m llm_engine.cli.generate --checkpoint checkpoints/tiny_gpt.pt --prompt "Once upon a time"
```

You can adjust sampling:

```bash
python -m llm_engine.cli.generate --checkpoint checkpoints/tiny_gpt.pt --prompt "Once upon a time" --temperature 0.8 --top-k 40 --max-new-tokens 120
```

## Stream Tokens Over HTTP

Start the server (needs a trained checkpoint; override with `CHECKPOINT_PATH` / `DEVICE`):

```bash
python -m uvicorn llm_engine.server.app:app --port 8000
```

Stream a completion as Server-Sent Events (`-N` disables curl buffering):

```bash
curl -N -X POST localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 100, "temperature": 0.8, "top_k": 40}'
```

Each event is `data: {"token": "..."}`, and the stream ends with `data: [DONE]`.

## Check CUDA Availability

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Configuration

The training and model hyperparameters live in `configs/tiny_gpt.yaml`, including:

- vocabulary size
- context length
- model width and depth
- batch size
- learning rate
- evaluation cadence
- number of TinyStories examples to load
- device selection

## Future Phases

Planned as new sub-packages of `llm_engine/`:

- `llm_engine/engine/`: KV-cache support, asynchronous request scheduling, continuous batching, latency and throughput instrumentation
- `benchmarks/`: generation throughput benchmarks, latency profiling, and CUDA profiling scripts