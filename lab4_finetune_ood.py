from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from lab4_detection import DyckDataset, build_split
from lab4_ood import load_detection_model, predict


def evaluate_depths(
    model: nn.Module,
    depths: list[int],
    examples_per_depth: int,
    length_range: tuple[int, int],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    results = {}
    for depth in depths:
        examples = build_split(examples_per_depth, length_range, [depth], True, max_len)
        loader = DataLoader(DyckDataset(examples), batch_size=batch_size)
        gold, pred, _ = predict(model, loader, device)
        results[str(depth)] = accuracy_score(gold, pred)
    return results


def plot_before_after(before: dict[str, float], after: dict[str, float], output_path: Path) -> None:
    depths = list(before.keys())
    x = np.arange(len(depths))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, [before[depth] for depth in depths], width, label="before")
    ax.bar(x + width / 2, [after[depth] for depth in depths], width, label="after")
    ax.set_xticks(x, labels=depths)
    ax.set_xlabel("Maximum nesting depth")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finetune_ood"))
    parser.add_argument("--finetune-size", type=int, default=500)
    parser.add_argument("--eval-size-per-depth", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    length_range = (40, max_len - 2)

    before = evaluate_depths(
        model,
        [5, 6, 7],
        args.eval_size_per_depth,
        length_range,
        max_len,
        args.batch_size,
        device,
    )

    finetune_examples = build_split(args.finetune_size, length_range, [5], True, max_len)
    finetune_loader = DataLoader(DyckDataset(finetune_examples), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        gold = []
        pred = []
        for batch in finetune_loader:
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
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "accuracy": accuracy_score(gold, pred),
            }
        )
        print(f"epoch={epoch} loss={history[-1]['loss']:.4f} acc={history[-1]['accuracy']:.4f}")

    model.eval()
    after = evaluate_depths(
        model,
        [5, 6, 7],
        args.eval_size_per_depth,
        length_range,
        max_len,
        args.batch_size,
        device,
    )

    metrics = {
        "config": {
            "finetune_size": args.finetune_size,
            "eval_size_per_depth": args.eval_size_per_depth,
            "epochs": args.epochs,
            "lr": args.lr,
            "length_range": list(length_range),
            "device": str(device),
        },
        "before": before,
        "after": after,
        "delta": {depth: after[depth] - before[depth] for depth in before},
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_before_after(before, after, args.output_dir / "before_after_accuracy.png")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
