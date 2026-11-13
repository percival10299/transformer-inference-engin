from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from llm_engine.data.dataset import TokenDataset
from llm_engine.data.tokenizer import ByteTokenizer
from llm_engine.device import resolve_device
from llm_engine.model.gpt import GPTConfig, GPTModel
from llm_engine.training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        checkpoint_path: str | Path,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.get("device", "auto"))
        self.model_config = GPTConfig.from_dict(config)
        self.model = GPTModel(self.model_config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.checkpoint_path = Path(checkpoint_path)
        self.step = 0

    def maybe_resume(self) -> bool:
        latest_checkpoint = find_latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint is None:
            return False

        checkpoint = load_checkpoint(
            latest_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            map_location=self.device,
            trusted=True,
            load_optimizer=True,
        )
        self.step = int(checkpoint.get("step", 0))
        return True

    def evaluate(self, dataset: TokenDataset) -> float:
        self.model.eval()
        losses = []
        with torch.inference_mode():
            for _ in range(self.config["eval_batches"]):
                x, y = dataset.get_batch(self.config["batch_size"], self.device)
                _, loss = self.model(x, targets=y)
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def save(self, tokenizer: ByteTokenizer) -> None:
        save_checkpoint(
            path=self.checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            model_config=self.model_config.to_dict(),
            tokenizer_metadata=tokenizer.metadata(),
        )

    def train(self, train_dataset: TokenDataset, val_dataset: TokenDataset, tokenizer: ByteTokenizer) -> None:
        progress = tqdm(total=self.config["max_steps"], initial=self.step, desc="training")

        while self.step < self.config["max_steps"]:
            x, y = train_dataset.get_batch(self.config["batch_size"], self.device)
            logits, loss = self.model(x, targets=y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()

            self.step += 1
            progress.update(1)

            if self.step == 1 or self.step % self.config["eval_interval"] == 0 or self.step == self.config["max_steps"]:
                val_loss = self.evaluate(val_dataset)
                lr = self.optimizer.param_groups[0]["lr"]
                gpu_mem_mb = 0.0
                if self.device.type == "cuda":
                    gpu_mem_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

                progress.set_postfix(
                    train_loss=f"{loss.item():.4f}",
                    val_loss=f"{val_loss:.4f}",
                    lr=f"{lr:.6f}",
                    gpu_mem_mb=f"{gpu_mem_mb:.1f}",
                )
                tqdm.write(
                    f"step={self.step} train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
                    f"lr={lr:.6f} gpu_mem_mb={gpu_mem_mb:.1f}"
                )
                self.save(tokenizer)

        progress.close()