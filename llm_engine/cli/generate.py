from __future__ import annotations

import argparse

from llm_engine.device import resolve_device
from llm_engine.generation.generator import decode_generated_text, generate_token_ids, load_model_for_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of tokens to sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling limit")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, tokenizer, _ = load_model_for_generation(args.checkpoint, device=device)
    token_ids = generate_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = decode_generated_text(tokenizer, token_ids)
    print(text)


if __name__ == "__main__":
    main()