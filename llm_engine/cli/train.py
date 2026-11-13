from __future__ import annotations

import argparse

from llm_engine.data.dataset import prepare_tinystories_datasets
from llm_engine.training.trainer import Trainer, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-style Transformer on TinyStories")
    parser.add_argument("--config", type=str, default="configs/tiny_gpt.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tiny_gpt.pt",
        help="Checkpoint path for save/resume",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume if the checkpoint already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    train_dataset, val_dataset, tokenizer = prepare_tinystories_datasets(
        n_stories=config["n_stories"],
        context_length=config["context_length"],
        seed=config["seed"],
    )

    trainer = Trainer(config=config, checkpoint_path=args.checkpoint)
    if not args.no_resume:
        trainer.maybe_resume()
    trainer.train(train_dataset, val_dataset, tokenizer)


if __name__ == "__main__":
    main()