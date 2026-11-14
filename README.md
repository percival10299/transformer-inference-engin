# Transformer Inference & LLM Serving Engine

This repository starts with Phase 1 of a larger portfolio project: a decoder-only GPT-style Transformer implemented from scratch in PyTorch, trained on TinyStories, saved to checkpoints, and used for basic autoregressive generation.

Later phases will add an inference engine with continuous batching and further performance work.

## Phase 1 Scope

- GPT-style decoder-only Transformer written with PyTorch building blocks
- TinyStories data loading and byte-level tokenization
- next-token prediction training loop with validation
- checkpoint save and resume support
- autoregressive text generation with an optional KV cache (on by default)
- CPU-friendly unit tests that do not download TinyStories

## Project Layout

```text
configs/                 YAML hyperparameters
checkpoints/             saved model/optimizer state (gitignored)
tests/                   CPU-friendly unit tests
benchmarks/              KV-cache decoding benchmark
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

## KV Cache

Without a cache, every decode step re-runs the whole sequence through the model, so the work per token grows with context length. With a cache, each layer stores the keys and values of tokens it has already seen:

1. **Prefill:** the full prompt goes through the model once, producing the first token and filling the cache (`GPTModel.forward_with_cache(prompt)`).
2. **Decode:** each later step feeds only the newest token plus the cache (`forward_with_cache(token, cache)`). Its query attends to all cached keys, and its own K/V is appended.

`CausalSelfAttention.forward(x, past_kv, use_cache)` returns `(output, present_kv)`. The positional embedding is offset by the cache length, and the causal mask rows are taken at the tokens' absolute positions.

Generation uses the cache by default. Pass `use_kv_cache=False` to `generate_token_ids` / `stream_token_ids` for the uncached path. `temperature=0` selects greedy decoding.

Limits:

- Position embeddings are learned and absolute, so a sequence cannot exceed `context_length`. Past that point the window slides, cached positions become invalid, and generation falls back to recomputing the window each step.
- The cache grows with `torch.cat` each step, which copies it. A preallocated buffer would avoid that.

## Benchmark: KV Cache vs No Cache

```bash
python -m benchmarks.kv_cache --device cpu
```

Greedy decoding of 64 tokens at prompt lengths 128, 512 and 1,024; the median of 5 runs is reported. The trained checkpoint only supports 128 positions, so the benchmark uses a randomly initialised model with the same architecture (d_model 128, 4 heads, 4 layers, 1.0M parameters) and a longer context window. Speed does not depend on the weight values. Every run also checks that the cached and uncached tokens are identical (`match`) and reports the largest logit difference.

Apple M3 Max, CPU, torch 2.14.1, 10 threads. Decode columns time only the 63 steps after the prefill:

| Context | Uncached ms/token | Cached ms/token | Uncached tok/s | Cached tok/s | Decode speedup | Prefill ms | KV cache MiB | Tokens match | Max logit diff |
|--------:|------------------:|----------------:|---------------:|-------------:|---------------:|-----------:|-------------:|:------------:|---------------:|
| 128     | 2.96              | 0.69            | 338            | 1,440        | 4.3x           | 2.5        | 0.7          | yes          | 1.7e-05        |
| 512     | 13.15             | 1.10            | 76             | 912          | 12.0x          | 12.3       | 2.2          | yes          | 1.8e-05        |
| 1,024   | 24.43             | 1.15            | 41             | 870          | 21.2x          | 18.6       | 4.2          | yes          | 1.9e-05        |

Uncached cost per token grows with context, roughly 8x from 128 to 1,024 tokens. Cached cost grows only 1.7x, so the speedup widens with context. A second CPU run was within about 5% for the 128 and 512 rows. At 1,024 tokens the uncached time stayed within 2% (24.95 ms), but the cached time moved 9% (1.05 ms vs 1.15 ms). That shifts the speedup to 23.7x, so read the 1,024 row as roughly 21-24x.

On the Apple GPU (`--device mps`) the speedup is only 1.0x-1.2x. This is not a cache problem. A cached step takes about 0.9 ms when the cache length is fixed, but about 5 ms when it grows each step, because each new tensor shape adds per-step overhead on MPS that outweighs the saved compute. Use CPU or CUDA numbers when judging the cache.

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
python -m llm_engine.cli.train --config configs/transformer.yaml
```

To disable automatic checkpoint resume:

```bash
python -m llm_engine.cli.train --config configs/transformer.yaml --no-resume
```

## Resume Training

If `checkpoints/transformer.pt` exists, training resumes from it by default.

```bash
python -m llm_engine.cli.train --config configs/transformer.yaml --checkpoint checkpoints/transformer.pt
```

## Generate Text

```bash
python -m llm_engine.cli.generate --checkpoint checkpoints/transformer.pt --prompt "Once upon a time"
```

You can adjust sampling:

```bash
python -m llm_engine.cli.generate --checkpoint checkpoints/transformer.pt --prompt "Once upon a time" --temperature 0.8 --top-k 40 --max-new-tokens 120
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

The training and model hyperparameters live in `configs/transformer.yaml`, including:

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

- `llm_engine/engine/`: asynchronous request scheduling, continuous batching, latency and throughput instrumentation
- `benchmarks/`: more benchmarks for throughput and latency, and CUDA profiling scripts