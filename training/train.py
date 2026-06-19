"""
training/train.py — Fine-tuning loop for the embedding model.

Key techniques used here:
  - SimCSE contrastive loss (see models/embedding_model.py)
  - Hard negative mining: instead of random negatives, we pick documents that
    are *almost* relevant — these are harder to distinguish and force the model
    to learn finer-grained semantics
  - Warmup + cosine LR schedule: standard for transformer fine-tuning
  - Weights & Biases (wandb) for experiment tracking
  - Gradient checkpointing to fit larger batches in GPU memory
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from pathlib import Path
from typing import List, Tuple
import random
import json

# wandb is optional — gracefully skip if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Train] wandb not installed. Install with `pip install wandb` for experiment tracking.")

from models.embedding_model import EmbeddingModel, SimCSELoss
from config import cfg, TrainingConfig


# ── Dataset ───────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Each sample is a (query, positive_doc, hard_negative_doc) triplet.

    Format of input JSON:
        [{"query": "...", "positive": "...", "hard_negative": "..."}, ...]

    Hard negatives come from BM25 or a previous embedding model's top-k results
    that are *not* the true answer (mined offline, see data/mine_negatives.py).
    """

    def __init__(self, data_path: str):
        with open(data_path) as f:
            self.samples: List[dict] = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        s = self.samples[idx]
        return s["query"], s["positive"], s["hard_negative"]


def collate_fn(batch):
    """Unzip list of (q, p, n) tuples into three separate lists."""
    queries, positives, negatives = zip(*batch)
    return list(queries), list(positives), list(negatives)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    data_path: str,
    output_dir: Path = None,
    train_cfg: TrainingConfig = None,
    device: str = None,
):
    """
    Fine-tune the embedding model using contrastive learning.

    Args:
        data_path:  Path to triplet training data JSON.
        output_dir: Where to save checkpoints.
        train_cfg:  Hyperparameters (defaults to cfg.training).
        device:     "cuda", "mps", or "cpu".
    """
    train_cfg = train_cfg or cfg.training
    output_dir = output_dir or cfg.embedding.checkpoint_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect the best available device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            # Apple Silicon GPU (great for Mac development)
            device = "mps"
        else:
            device = "cpu"
    print(f"[Train] Using device: {device}")

    # ── Model & loss ──────────────────────────────────────────────────────────
    model = EmbeddingModel().to(device)
    # Gradient checkpointing trades compute for memory — lets you use 2-4x larger batches
    model.backbone.gradient_checkpointing_enable()

    loss_fn = SimCSELoss(temperature=train_cfg.temperature if hasattr(train_cfg, 'temperature') else cfg.embedding.temperature)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = TripletDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device == "cuda"),  # pin_memory speeds up CPU→GPU transfers
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    # AdamW decouples weight decay from the gradient update (fixes Adam's L2 bug)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    total_steps = len(loader) * train_cfg.epochs
    # Cosine decay with linear warmup: the de-facto standard for transformer fine-tuning
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── W&B init ──────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE:
        wandb.init(
            project=train_cfg.wandb_project,
            config={
                "base_model": cfg.embedding.base_model,
                "embedding_dim": cfg.embedding.embedding_dim,
                "temperature": cfg.embedding.temperature,
                **train_cfg.__dict__,
            },
        )
        wandb.watch(model, log="gradients", log_freq=50)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")

    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (queries, positives, hard_negatives) in enumerate(loader):
            # Tokenize all three text groups
            q_enc = model.tokenizer(queries,   padding=True, truncation=True, max_length=cfg.embedding.max_seq_length, return_tensors="pt").to(device)
            p_enc = model.tokenizer(positives, padding=True, truncation=True, max_length=cfg.embedding.max_seq_length, return_tensors="pt").to(device)
            n_enc = model.tokenizer(hard_negatives, padding=True, truncation=True, max_length=cfg.embedding.max_seq_length, return_tensors="pt").to(device)

            # Forward pass
            q_emb = model(q_enc["input_ids"], q_enc["attention_mask"])
            p_emb = model(p_enc["input_ids"], p_enc["attention_mask"])
            n_emb = model(n_enc["input_ids"], n_enc["attention_mask"])

            loss = loss_fn(q_emb, p_emb, n_emb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients — prevents exploding gradients in transformer layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                })

            # Periodic console logging
            if batch_idx % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{train_cfg.epochs} | Step {global_step} | Loss {loss.item():.4f} | LR {lr:.2e}")

            # Checkpoint if this is the best loss seen so far
            if global_step % train_cfg.eval_every_n_steps == 0:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    _save_checkpoint(model, output_dir / "best_model.pt", global_step, best_loss)

        avg_loss = epoch_loss / len(loader)
        print(f"[Train] Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    # Save final model weights
    _save_checkpoint(model, output_dir / "final_model.pt", global_step, avg_loss)
    if WANDB_AVAILABLE:
        wandb.finish()

    print(f"[Train] Training complete. Checkpoints saved to {output_dir}")
    return model


def _save_checkpoint(model: EmbeddingModel, path: Path, step: int, loss: float) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
        "config": {
            "base_model": cfg.embedding.base_model,
            "embedding_dim": cfg.embedding.embedding_dim,
        },
    }, path)
    print(f"  [Checkpoint] Saved → {path} (step={step}, loss={loss:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune NeuralRAG embedding model")
    parser.add_argument("--data", required=True, help="Path to triplet training data JSON")
    parser.add_argument("--output", default=None, help="Output directory for checkpoints")
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    train(data_path=args.data, output_dir=args.output, device=args.device)
