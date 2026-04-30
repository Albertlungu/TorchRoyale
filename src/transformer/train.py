"""
Training script for the Decision Transformer.

Usage:
    python -m src.transformer.train --episodes data/episodes.pkl --output data/models/dt/
    python -m src.transformer.train --episodes data/episodes.pkl --output data/models/dt/ --epochs 200
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import DTConfig
from .model import DecisionTransformer
from src.data.dt_dataset import load_dataset, DecisionTransformerDataset


class DTTrainer:
    """Handles the Decision Transformer training loop.

    Attributes:
        config (DTConfig): Training configuration.
        model (DecisionTransformer): The DT model being trained.
        dataset (DecisionTransformerDataset): Training dataset.
        output_dir (Path): Directory for saving checkpoints.
        device (torch.device): Training device (CPU/CUDA/MPS).
        optimizer: AdamW optimizer.
        card_loss_fn: CrossEntropyLoss for card prediction.
        pos_loss_fn: CrossEntropyLoss for position prediction.
    """

    def __init__(
        self,
        config: DTConfig,
        model: DecisionTransformer,
        dataset: DecisionTransformerDataset,
        output_dir: str = "data/models/dt",
        device: str = "auto",
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.card_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.pos_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_card_loss = 0.0
        total_pos_loss = 0.0
        total_card_correct = 0
        total_pos_correct = 0
        total_samples = 0

        for batch in dataloader:
            states = batch["states"].to(self.device)
            actions_card = batch["actions_card"].to(self.device)
            actions_pos = batch["actions_pos"].to(self.device)
            rtg = batch["returns_to_go"].to(self.device)
            timesteps = batch["timesteps"].to(self.device)
            mask = batch["attention_mask"].to(self.device)

            card_logits, pos_logits = self.model(
                states, actions_card, actions_pos, rtg, timesteps, mask
            )

            B, K = states.shape[0], states.shape[1]

            # Per-token losses
            card_loss = self.card_loss_fn(
                card_logits.reshape(-1, self.config.card_action_dim),
                actions_card.reshape(-1),
            ).reshape(B, K)

            pos_loss = self.pos_loss_fn(
                pos_logits.reshape(-1, self.config.pos_action_dim),
                actions_pos.reshape(-1),
            ).reshape(B, K)

            # Mask padded positions
            n_real = mask.sum()
            card_loss = (card_loss * mask).sum() / n_real
            pos_loss = (pos_loss * mask).sum() / n_real

            loss = (
                self.config.card_loss_weight * card_loss
                + self.config.pos_loss_weight * pos_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Track metrics
            with torch.no_grad():
                card_preds = card_logits.argmax(dim=-1)
                pos_preds = pos_logits.argmax(dim=-1)
                total_card_correct += ((card_preds == actions_card) * mask).sum().item()
                total_pos_correct += ((pos_preds == actions_pos) * mask).sum().item()

            total_card_loss += card_loss.item() * n_real.item()
            total_pos_loss += pos_loss.item() * n_real.item()
            total_samples += n_real.item()

        return {
            "card_loss": total_card_loss / max(total_samples, 1),
            "pos_loss": total_pos_loss / max(total_samples, 1),
            "card_acc": total_card_correct / max(total_samples, 1),
            "pos_acc": total_pos_correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        total_card_loss = 0.0
        total_pos_loss = 0.0
        total_card_correct = 0
        total_pos_correct = 0
        total_pos_top5 = 0
        total_samples = 0

        for batch in dataloader:
            states = batch["states"].to(self.device)
            actions_card = batch["actions_card"].to(self.device)
            actions_pos = batch["actions_pos"].to(self.device)
            rtg = batch["returns_to_go"].to(self.device)
            timesteps = batch["timesteps"].to(self.device)
            mask = batch["attention_mask"].to(self.device)

            card_logits, pos_logits = self.model(
                states, actions_card, actions_pos, rtg, timesteps, mask
            )

            B, K = states.shape[0], states.shape[1]

            card_loss = self.card_loss_fn(
                card_logits.reshape(-1, self.config.card_action_dim),
                actions_card.reshape(-1),
            ).reshape(B, K)

            pos_loss = self.pos_loss_fn(
                pos_logits.reshape(-1, self.config.pos_action_dim),
                actions_pos.reshape(-1),
            ).reshape(B, K)

            n_real = mask.sum()
            card_loss = (card_loss * mask).sum() / n_real
            pos_loss = (pos_loss * mask).sum() / n_real

            card_preds = card_logits.argmax(dim=-1)
            pos_preds = pos_logits.argmax(dim=-1)
            total_card_correct += ((card_preds == actions_card) * mask).sum().item()
            total_pos_correct += ((pos_preds == actions_pos) * mask).sum().item()

            # Top-5 position accuracy
            pos_top5 = pos_logits.topk(5, dim=-1).indices
            pos_target_expanded = actions_pos.unsqueeze(-1).expand_as(pos_top5)
            top5_match = (pos_top5 == pos_target_expanded).any(dim=-1).float()
            total_pos_top5 += (top5_match * mask).sum().item()

            total_card_loss += card_loss.item() * n_real.item()
            total_pos_loss += pos_loss.item() * n_real.item()
            total_samples += n_real.item()

        return {
            "card_loss": total_card_loss / max(total_samples, 1),
            "pos_loss": total_pos_loss / max(total_samples, 1),
            "card_acc": total_card_correct / max(total_samples, 1),
            "pos_acc": total_pos_correct / max(total_samples, 1),
            "pos_top5_acc": total_pos_top5 / max(total_samples, 1),
        }

    def save_checkpoint(self, filename: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "rtg_mean": self.dataset.rtg_mean,
            "rtg_std": self.dataset.rtg_std,
        }, self.output_dir / filename)

    def train(self, test_size: float = 0.2):
        """Run the full training loop with train/val split."""
        # Split dataset
        n_val = int(len(self.dataset) * test_size)
        n_train = len(self.dataset) - n_val

        if n_train == 0:
            print("Not enough data to train.")
            return

        train_ds, val_ds = random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )

        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_epochs)
        best_val_loss = float("inf")

        print(f"Training on {self.device} | {n_train} train, {n_val} val samples")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        for epoch in range(self.config.max_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            scheduler.step()
            elapsed = time.time() - t0

            val_total = val_metrics["card_loss"] + val_metrics["pos_loss"]

            # Save best
            if val_total < best_val_loss:
                best_val_loss = val_total
                self.save_checkpoint("best.pt")
                marker = " *"
            else:
                marker = ""

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            print(
                f"Epoch {epoch + 1:3d}/{self.config.max_epochs} "
                f"| train card_loss={train_metrics['card_loss']:.4f} "
                f"pos_loss={train_metrics['pos_loss']:.4f} "
                f"card_acc={train_metrics['card_acc']:.3f} "
                f"| val card_acc={val_metrics['card_acc']:.3f} "
                f"pos_acc={val_metrics['pos_acc']:.3f} "
                f"pos_top5={val_metrics['pos_top5_acc']:.3f} "
                f"| {elapsed:.1f}s{marker}"
            )

        # Save final
        self.save_checkpoint("final.pt")
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--episodes", required=True, help="Path to episodes.pkl")
    parser.add_argument("--output", default="data/models/dt", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--context-length", type=int, default=None, help="Override context length K")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    config = DTConfig()
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.context_length is not None:
        config.context_length = args.context_length

    # Load dataset
    dataset = load_dataset(
        args.episodes,
        context_length=config.context_length,
        state_noise_std=config.state_noise_std,
        rtg_noise_std=config.rtg_noise_std,
    )
    print(f"Loaded {len(dataset.episodes)} episodes, {len(dataset)} samples")

    # Build model
    model = DecisionTransformer(config)

    # Train
    trainer = DTTrainer(
        config=config,
        model=model,
        dataset=dataset,
        output_dir=args.output,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
