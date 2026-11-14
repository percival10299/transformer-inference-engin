"""Benchmark KV-cached vs uncached greedy decoding at several context lengths.

Usage (from the repo root):
    python -m benchmarks.kv_cache
    python -m benchmarks.kv_cache --device cpu --contexts 128 512 1024 --new-tokens 64

The trained checkpoint only supports 128 positions (learned absolute position
embeddings), so this uses a randomly initialised model with the same architecture
and a larger context window. Speed does not depend on the weight values.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import torch

from llm_engine.device import resolve_device
from llm_engine.model.gpt import GPTConfig, GPTModel


@dataclass
class RunResult:
    prefill_s: float
    decode_s: float
    tokens: torch.Tensor  # [new_tokens]
    logits: torch.Tensor  # [new_tokens, vocab]
    cache_bytes: int


def synchronize(device: torch.device) -> None:
    # GPU kernels run asynchronously; without this the timer would stop too early.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run(model: GPTModel, prompt: torch.Tensor, new_tokens: int, use_cache: bool, device: torch.device) -> RunResult:
    """Greedy-decode `new_tokens` tokens.

    Step 1 is a full forward pass over the prompt in both modes (prefill). The remaining
    new_tokens - 1 steps are the timed decode phase: cached feeds only the newest token,
    uncached feeds the whole sequence again.
    """
    tokens: list[torch.Tensor] = []
    logits_per_step: list[torch.Tensor] = []
    cache = None

    synchronize(device)
    start = time.perf_counter()
    if use_cache:
        logits, cache = model.forward_with_cache(prompt)
    else:
        logits, _ = model(prompt)
    last_logits = logits[:, -1]
    next_token = last_logits.argmax(dim=-1, keepdim=True)
    tokens.append(next_token)
    logits_per_step.append(last_logits)
    synchronize(device)
    prefill_s = time.perf_counter() - start

    sequence = torch.cat([prompt, next_token], dim=1)
    start = time.perf_counter()
    for _ in range(new_tokens - 1):
        if use_cache:
            logits, cache = model.forward_with_cache(next_token, cache)
        else:
            logits, _ = model(sequence)
        last_logits = logits[:, -1]
        next_token = last_logits.argmax(dim=-1, keepdim=True)
        tokens.append(next_token)
        logits_per_step.append(last_logits)
        if not use_cache:
            sequence = torch.cat([sequence, next_token], dim=1)
    synchronize(device)
    decode_s = time.perf_counter() - start

    cache_bytes = 0
    if cache is not None:
        cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)
    return RunResult(
        prefill_s=prefill_s,
        decode_s=decode_s,
        tokens=torch.cat(tokens, dim=1)[0],
        logits=torch.cat(logits_per_step, dim=0),
        cache_bytes=cache_bytes,
    )


def benchmark_context(
    model: GPTModel, context: int, new_tokens: int, repeats: int, warmup: int, device: torch.device
) -> dict[str, float | bool]:
    generator = torch.Generator().manual_seed(context)
    prompt = torch.randint(0, model.config.vocab_size, (1, context), generator=generator).to(device)

    for _ in range(warmup):
        run(model, prompt, new_tokens, use_cache=False, device=device)
        run(model, prompt, new_tokens, use_cache=True, device=device)

    # Alternate modes inside each repeat so drift (thermal throttling, other load) hits both.
    uncached_runs: list[RunResult] = []
    cached_runs: list[RunResult] = []
    for _ in range(repeats):
        uncached_runs.append(run(model, prompt, new_tokens, use_cache=False, device=device))
        cached_runs.append(run(model, prompt, new_tokens, use_cache=True, device=device))

    decode_steps = new_tokens - 1
    uncached_decode = statistics.median(r.decode_s for r in uncached_runs) / decode_steps * 1000
    cached_decode = statistics.median(r.decode_s for r in cached_runs) / decode_steps * 1000
    uncached_total = statistics.median(r.prefill_s + r.decode_s for r in uncached_runs) * 1000
    cached_total = statistics.median(r.prefill_s + r.decode_s for r in cached_runs) * 1000

    uncached, cached = uncached_runs[-1], cached_runs[-1]
    return {
        "context": context,
        "uncached_ms_per_token": uncached_decode,
        "cached_ms_per_token": cached_decode,
        "uncached_tok_s": 1000 / uncached_decode,
        "cached_tok_s": 1000 / cached_decode,
        "decode_speedup": uncached_decode / cached_decode,
        "prefill_ms": statistics.median(r.prefill_s for r in cached_runs) * 1000,
        "end_to_end_speedup": uncached_total / cached_total,
        "cache_mib": cached.cache_bytes / 2**20,
        "tokens_match": bool(torch.equal(uncached.tokens, cached.tokens)),
        "max_logit_diff": float((uncached.logits - cached.logits).abs().max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KV-cached vs uncached decoding")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024], help="Prompt lengths to test")
    parser.add_argument("--new-tokens", type=int, default=64, help="Tokens to generate per run (>= 2)")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per mode (median reported)")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs per mode")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.new_tokens < 2:
        raise SystemExit("--new-tokens must be at least 2 (one prefill step plus one decode step)")

    device = resolve_device(args.device)
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=257,
        context_length=max(args.contexts) + args.new_tokens,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.0,
    )
    model = GPTModel(config).to(device).eval()
    # Default init gives near-uniform logits; widen them so greedy argmax is not a float-noise tie.
    torch.nn.init.normal_(model.lm_head.weight, std=0.5)
    params = sum(p.numel() for p in model.parameters())

    print(f"device={device} torch={torch.__version__} threads={torch.get_num_threads()}")
    print(
        f"model: d_model={config.d_model} heads={config.n_heads} layers={config.n_layers} "
        f"params={params / 1e6:.2f}M context_length={config.context_length} (random weights)"
    )
    print(f"greedy decoding, {args.new_tokens} new tokens ({args.new_tokens - 1} timed decode steps), "
          f"median of {args.repeats} runs\n")

    rows = [benchmark_context(model, c, args.new_tokens, args.repeats, args.warmup, device) for c in args.contexts]

    header = (
        f"{'context':>7} | {'uncached ms/tok':>15} | {'cached ms/tok':>13} | {'uncached tok/s':>14} | "
        f"{'cached tok/s':>12} | {'decode speedup':>14} | {'prefill ms':>10} | {'end-to-end':>10} | "
        f"{'KV MiB':>6} | {'match':>5} | {'max |dlogit|':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['context']:>7} | {r['uncached_ms_per_token']:>15.2f} | {r['cached_ms_per_token']:>13.2f} | "
            f"{r['uncached_tok_s']:>14.1f} | {r['cached_tok_s']:>12.1f} | {r['decode_speedup']:>13.1f}x | "
            f"{r['prefill_ms']:>10.1f} | {r['end_to_end_speedup']:>9.1f}x | {r['cache_mib']:>6.1f} | "
            f"{'yes' if r['tokens_match'] else 'NO':>5} | {r['max_logit_diff']:>12.2e}"
        )

    if not all(r["tokens_match"] for r in rows):
        print("\nFAIL: cached and uncached greedy tokens differ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
