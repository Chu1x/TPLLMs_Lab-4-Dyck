from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from lab4_detection import DyckDataset, build_split, plot_history
from lab4_length_aug import evaluate
from lab4_ood import load_detection_model, predict


def eval_ood(model: nn.Module, examples_per_depth: int, max_len: int, batch_size: int, device: torch.device) -> dict[str, float]:
    results = {}
    for depth in [5, 6, 7]:
        examples = build_split(examples_per_depth, (40, max_len - 2), [depth], True, max_len)
        loader = DataLoader(DyckDataset(examples), batch_size=batch_size)
        gold, pred, _ = predict(model, loader, device)
        results[str(depth)] = accuracy_score(gold, pred)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/length_aug_finetune"))
    parser.add_argument("--finetune-size", type=int, default=10_000)
    parser.add_argument("--dev-size", type=int, default=1_000)
    parser.add_argument("--ood-size-per-depth", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    criterion = nn.CrossEntropyLoss()

    before = eval_ood(model, args.ood_size_per_depth, max_len, args.batch_size, device)
    train_examples = build_split(args.finetune_size, (4, max_len - 2), [1, 2, 3, 4], False, max_len)
    dev_examples = build_split(args.dev_size, (4, max_len - 2), [1, 2, 3, 4], False, max_len)
    train_loader = DataLoader(DyckDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(DyckDataset(dev_examples), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        gold = []
        pred = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
            gold.extend(labels.cpu().tolist())
            pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        dev_metrics = evaluate(model, dev_loader, device, criterion)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "train_accuracy": float(accuracy_score(gold, pred)),
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"dev_acc={row['dev_accuracy']:.4f}"
        )

    after = eval_ood(model, args.ood_size_per_depth, max_len, args.batch_size, device)
    metrics = {
        "config": {
            "metrics_path": str(args.metrics_path),
            "model_path": str(args.model_path),
            "output_dir": str(args.output_dir),
            "finetune_size": args.finetune_size,
            "dev_size": args.dev_size,
            "ood_size_per_depth": args.ood_size_per_depth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
        },
        "before_ood_by_depth": before,
        "after_ood_by_depth": after,
        "before_ood_overall": float(np.mean(list(before.values()))),
        "after_ood_overall": float(np.mean(list(after.values()))),
        "delta_by_depth": {depth: after[depth] - before[depth] for depth in before},
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(history, args.output_dir / "training_curves.png")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
