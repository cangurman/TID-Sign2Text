"""Train the TID-Sign2Text phrase classifier.

Usage (from repo root):
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\train.py
    ... train.py --model transformer --epochs 120 --seq-len 60

Outputs:
    backend/models/checkpoints/best.pt        (weights + config + labels)
    backend/models/checkpoints/labels.json
    backend/models/checkpoints/confusion.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import LandmarkDataset, discover_samples, stratified_split  # noqa: E402
from model import build_model  # noqa: E402


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    preds_arr = np.concatenate(preds) if preds else np.array([])
    targets_arr = np.concatenate(targets) if targets else np.array([])
    acc = float((preds_arr == targets_arr).mean()) if len(preds_arr) else 0.0
    return acc, preds_arr, targets_arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BACKEND / "data" / "landmarks")
    parser.add_argument("--model", choices=["bilstm", "transformer"], default="bilstm")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streams", action="store_true",
                        help="add bone-vector + motion streams (input dim 258 -> 660)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, labels = discover_samples(args.data_dir)
    train_refs, val_refs = stratified_split(samples, args.val_ratio, args.seed)
    print(f"Classes: {len(labels)} | train: {len(train_refs)} | val: {len(val_refs)}")
    for i, name in enumerate(labels):
        n = sum(1 for s in samples if s.label_idx == i)
        print(f"  [{i:2d}] {name}: {n} samples")

    train_ds = LandmarkDataset(train_refs, seq_len=args.seq_len, augment=True, streams=args.streams)
    val_ds = LandmarkDataset(val_refs, seq_len=args.seq_len, augment=False, streams=args.streams)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    from features import FRAME_DIM, STREAMS_DIM
    input_dim = STREAMS_DIM if args.streams else FRAME_DIM
    model = build_model(args.model, num_classes=len(labels), input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_dir = BACKEND / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": args.model,
                    "seq_len": args.seq_len,
                    "labels": labels,
                    "streams": args.streams,
                },
                ckpt_dir / "best.pt",
            )
            marker = "  <- saved"
        print(f"epoch {epoch:3d}/{args.epochs} | loss {total_loss / max(n_batches, 1):.4f} "
              f"| val_acc {val_acc:.3f} (best {best_acc:.3f}){marker}")

    (ckpt_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final report with the best checkpoint
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    val_acc, preds, targets = evaluate(model, val_loader, device)
    print(f"\nBest val accuracy: {val_acc:.3f}")
    if len(preds):
        from sklearn.metrics import classification_report, confusion_matrix
        present = sorted(set(targets.tolist()) | set(preds.tolist()))
        print(classification_report(
            targets, preds, labels=present,
            target_names=[labels[i] for i in present], zero_division=0))
        cm = confusion_matrix(targets, preds, labels=list(range(len(labels))))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5),) * 2)
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
            fig.tight_layout()
            fig.savefig(ckpt_dir / "confusion.png", dpi=150)
            print(f"Confusion matrix -> {ckpt_dir / 'confusion.png'}")
        except Exception as exc:  # plotting must never fail the run
            print(f"[warn] confusion matrix plot skipped: {exc}")


if __name__ == "__main__":
    main()
