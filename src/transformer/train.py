"""
Training loop for the Decision Transformer.

Usage:
  python -m src.transformer.train --episodes output/pkl/all_combined.pkl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import load_dataset
from src.transformer.model import DTConfig, DecisionTransformer


class Trainer:
    def __init__(self, cfg: DTConfig, episodes_path: str, output_dir: str, device: str = "auto"):
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
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.card_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.pos_loss_fn  = nn.CrossEntropyLoss(reduction="none")

        self.train_ds, self.val_ds = load_dataset(episodes_path, context_len=cfg.context_len)

    def _batch_to(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forward(self, batch: dict):
        b = self._batch_to(batch)
        card_logits, pos_logits = self.model(
            b["states"], b["actions_card"], b["actions_pos"],
            b["returns_to_go"], b["timesteps"], b["attention_mask"],
        )
        mask = b["attention_mask"]
        B, T = mask.shape

        card_loss = self.card_loss_fn(
            card_logits.reshape(-1, self.cfg.n_cards),
            b["actions_card"].reshape(-1),
        ).reshape(B, T)
        pos_loss = self.pos_loss_fn(
            pos_logits.reshape(-1, self.cfg.n_positions_tile),
            b["actions_pos"].reshape(-1),
        ).reshape(B, T)

        n = mask.sum().clamp(min=1)
        card_loss = (card_loss * mask).sum() / n
        pos_loss  = (pos_loss  * mask).sum() / n
        loss = self.cfg.card_loss_weight * card_loss + self.cfg.pos_loss_weight * pos_loss

        # Accuracy
        with torch.no_grad():
            card_pred = card_logits.argmax(-1)
            pos_pred  = pos_logits.argmax(-1)
            card_acc  = ((card_pred == b["actions_card"]) * mask).sum() / n
            pos_acc   = ((pos_pred  == b["actions_pos"])  * mask).sum() / n
            pos_top5  = (
                (pos_logits.topk(5, -1).indices == b["actions_pos"].unsqueeze(-1)).any(-1) * mask
            ).sum() / n

        return loss, card_loss, pos_loss, card_acc, pos_acc, pos_top5

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = total_card = total_pos = total_card_acc = n_batches = 0
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
        return {
            "loss": total_loss / n_batches,
            "card_loss": total_card / n_batches,
            "pos_loss": total_pos / n_batches,
            "card_acc": total_card_acc / n_batches,
        }

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_card_acc = total_pos_acc = total_pos_top5 = n_batches = 0
        for batch in loader:
            _, _, _, card_acc, pos_acc, pos_top5 = self._forward(batch)
            total_card_acc += card_acc.item()
            total_pos_acc  += pos_acc.item()
            total_pos_top5 += pos_top5.item()
            n_batches += 1
        return {
            "card_acc": total_card_acc / n_batches,
            "pos_acc":  total_pos_acc  / n_batches,
            "pos_top5": total_pos_top5 / n_batches,
        }

    def save(self, name: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.cfg.__dict__,
        }, self.output_dir / name)

    def train(self) -> None:
        train_loader = DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0
        )
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.max_epochs)

        n_train, n_val = len(self.train_ds), len(self.val_ds)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Training on {self.device} | {n_train} train, {n_val} val samples")
        print(f"Model params: {total_params:,}")
        print()

        best_val_loss = float("inf")
        for epoch in range(self.cfg.max_epochs):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m   = self.val_epoch(val_loader)
            scheduler.step()
            dt = time.time() - t0

            val_loss = 1.0 - val_m["card_acc"]
            star = " *" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("best.pt")

            if (epoch + 1) % 10 == 0:
                self.save(f"epoch_{epoch+1}.pt")

            print(
                f"Epoch {epoch+1:3d}/{self.cfg.max_epochs}"
                f" | train card={train_m['card_loss']:.4f} pos={train_m['pos_loss']:.4f}"
                f" card_acc={train_m['card_acc']:.3f}"
                f" | val card_acc={val_m['card_acc']:.3f}"
                f" pos_acc={val_m['pos_acc']:.3f} pos_top5={val_m['pos_top5']:.3f}"
                f" | {dt:.1f}s{star}"
            )

        print(f"\nBest val loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {self.output_dir}")

    @property
    def n_positions_tile(self):
        from src.constants.game import GRID_COLS, GRID_ROWS
        return GRID_COLS * GRID_ROWS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", default="data/models/dt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = DTConfig()
    if args.epochs:       cfg.max_epochs = args.epochs
    if args.batch_size:   cfg.batch_size = args.batch_size
    if args.lr:           cfg.lr = args.lr
    if args.context_len:  cfg.context_len = args.context_len

    trainer = Trainer(cfg, args.episodes, args.output, args.device)
    trainer.train()


if __name__ == "__main__":
    main()
