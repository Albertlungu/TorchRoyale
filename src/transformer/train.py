"""
Training loop for the Decision Transformer.

Usage:
  python -m src.transformer.train --episodes output/pkl/all_combined.pkl

Public API:
  Trainer -- orchestrates data loading, training, validation, and checkpointing
  main()  -- CLI entry point
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.constants.game import GRID_COLS, GRID_ROWS
from src.data.dataset import load_dataset
from src.transformer.model import DTConfig, DecisionTransformer


class Trainer:
    """
    Manages the full training lifecycle for the Decision Transformer.

    Loads data, constructs the model and optimiser, and exposes train() to
    run the full epoch loop with checkpointing.
    """

    def __init__(
        self,
        cfg: DTConfig,
        episodes_path: str,
        output_dir: str,
        device: str = "auto",
    ) -> None:
        """
        Args:
            cfg:           DTConfig with all hyperparameters.
            episodes_path: path to a .pkl file of Episode objects.
            output_dir:    directory where checkpoints are saved.
            device:        PyTorch device string ("auto", "cpu", "cuda", "mps").
        """
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = DecisionTransformer(cfg).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.card_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.pos_loss_fn  = nn.CrossEntropyLoss(reduction="none")

        self.train_ds, self.val_ds = load_dataset(episodes_path, context_len=cfg.context_len)

    def _batch_to(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move all tensors in a batch dict to self.device."""
        return {key: val.to(self.device) for key, val in batch.items()}

    def _forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a full forward pass and compute masked losses and accuracy metrics.

        Args:
            batch: raw data loader batch dict.

        Returns:
            (loss, card_loss, pos_loss, card_acc, pos_acc, pos_top5)
        """
        batch_dev = self._batch_to(batch)
        card_logits, pos_logits = self.model(
            batch_dev["states"],
            batch_dev["actions_card"],
            batch_dev["actions_pos"],
            batch_dev["returns_to_go"],
            batch_dev["timesteps"],
            batch_dev["attention_mask"],
        )
        mask = batch_dev["attention_mask"]
        batch_size, seq_len = mask.shape

        card_loss = self.card_loss_fn(
            card_logits.reshape(-1, self.cfg.n_cards),
            batch_dev["actions_card"].reshape(-1),
        ).reshape(batch_size, seq_len)
        pos_loss = self.pos_loss_fn(
            pos_logits.reshape(-1, self.n_positions_tile),
            batch_dev["actions_pos"].reshape(-1),
        ).reshape(batch_size, seq_len)

        normaliser = mask.sum().clamp(min=1)
        card_loss_mean = (card_loss * mask).sum() / normaliser
        pos_loss_mean  = (pos_loss  * mask).sum() / normaliser
        total_loss = (
            self.cfg.card_loss_weight * card_loss_mean
            + self.cfg.pos_loss_weight * pos_loss_mean
        )

        with torch.no_grad():
            card_pred = card_logits.argmax(-1)
            pos_pred  = pos_logits.argmax(-1)
            card_acc  = ((card_pred == batch_dev["actions_card"]) * mask).sum() / normaliser
            pos_acc   = ((pos_pred  == batch_dev["actions_pos"])  * mask).sum() / normaliser
            pos_top5  = (
                (
                    pos_logits.topk(5, -1).indices == batch_dev["actions_pos"].unsqueeze(-1)
                ).any(-1) * mask
            ).sum() / normaliser

        return total_loss, card_loss_mean, pos_loss_mean, card_acc, pos_acc, pos_top5

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run one full training epoch.

        Args:
            loader: DataLoader over the training dataset.

        Returns:
            Dict with keys: loss, card_loss, pos_loss, card_acc.
        """
        self.model.train()
        total_loss = total_card = total_pos = total_card_acc = n_batches = 0.0
        for batch in loader:
            loss, card_loss, pos_loss, card_acc, _, _ = self._forward(batch)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_card += card_loss.item()
            total_pos  += pos_loss.item()
            total_card_acc += card_acc.item()
            n_batches += 1
        n_batches = max(1.0, n_batches)
        return {
            "loss":      total_loss / n_batches,
            "card_loss": total_card / n_batches,
            "pos_loss":  total_pos  / n_batches,
            "card_acc":  total_card_acc / n_batches,
        }

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run one full validation epoch.

        Args:
            loader: DataLoader over the validation dataset.

        Returns:
            Dict with keys: card_acc, pos_acc, pos_top5.
        """
        self.model.eval()
        total_card_acc = total_pos_acc = total_pos_top5 = n_batches = 0.0
        for batch in loader:
            _, _, _, card_acc, pos_acc, pos_top5 = self._forward(batch)
            total_card_acc += card_acc.item()
            total_pos_acc  += pos_acc.item()
            total_pos_top5 += pos_top5.item()
            n_batches += 1
        n_batches = max(1.0, n_batches)
        return {
            "card_acc": total_card_acc / n_batches,
            "pos_acc":  total_pos_acc  / n_batches,
            "pos_top5": total_pos_top5 / n_batches,
        }

    def save(self, name: str) -> None:
        """
        Save a model checkpoint to output_dir.

        Args:
            name: filename (e.g. "best.pt" or "epoch_10.pt").
        """
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.cfg.__dict__,
        }, self.output_dir / name)

    def train(self, save_every: int = 25) -> None:
        """Run the full training loop for cfg.max_epochs, saving checkpoints.

        Args:
            save_every: save a numbered checkpoint every this many epochs.
        """
        train_loader = DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0
        )
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.max_epochs)

        n_train = len(self.train_ds)
        n_val = len(self.val_ds)
        total_params = sum(param.numel() for param in self.model.parameters())
        print(f"Training on {self.device} | {n_train} train, {n_val} val samples")
        print(f"Model params: {total_params:,}")
        print()

        best_val_loss = float("inf")
        for epoch in range(self.cfg.max_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.val_epoch(val_loader)
            scheduler.step()
            elapsed = time.time() - t0

            val_loss = 1.0 - val_metrics["card_acc"]
            star = " *" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("best.pt")

            if (epoch + 1) % save_every == 0:
                self.save(f"epoch_{epoch+1}.pt")

            print(
                f"Epoch {epoch+1:3d}/{self.cfg.max_epochs}"
                f" | train card={train_metrics['card_loss']:.4f}"
                f" pos={train_metrics['pos_loss']:.4f}"
                f" card_acc={train_metrics['card_acc']:.3f}"
                f" | val card_acc={val_metrics['card_acc']:.3f}"
                f" pos_acc={val_metrics['pos_acc']:.3f}"
                f" pos_top5={val_metrics['pos_top5']:.3f}"
                f" | {elapsed:.1f}s{star}"
            )

        print(f"\nBest val loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {self.output_dir}")

    @property
    def n_positions_tile(self) -> int:
        """Total tile count used as the position action space size."""
        return GRID_COLS * GRID_ROWS


def main() -> None:
    """CLI entry point for training the Decision Transformer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", default="data/models/dt")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    cfg = DTConfig()
    cfg.max_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.context_len:
        cfg.context_len = args.context_len

    trainer = Trainer(cfg, args.episodes, args.output, args.device)
    trainer.train(save_every=args.save_every)


if __name__ == "__main__":
    main()
