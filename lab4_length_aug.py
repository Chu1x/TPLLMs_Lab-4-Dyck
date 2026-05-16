from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from lab4_detection import DyckDataset, DyckTransformerClassifier, TOKENS, build_split, plot_history
from lab4_ood import predict


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    losses = []
    gold = []
    pred = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            losses.append(criterion(logits, labels).item())
            gold.extend(labels.cpu().tolist())
            pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro"),
    }


def evaluate_ood_by_depth(
    model: nn.Module,
    examples_per_depth: int,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    results = {}
    for depth in [5, 6, 7]:
        examples = build_split(examples_per_depth, (40, max_len - 2), [depth], True, max_len)
        loader = DataLoader(DyckDataset(examples), batch_size=batch_size)
        gold, pred, _ = predict(model, loader, device)
        results[str(depth)] = accuracy_score(gold, pred)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--dev-size", type=int, default=1_000)
    parser.add_argument("--id-test-size", type=int, default=1_000)
    parser.add_argument("--ood-size-per-depth", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/length_aug"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modification: keep the training depth bound n <= 4, but expose the model to the
    # long lengths used in OOD evaluation.
    train_examples = build_split(args.train_size, (4, args.max_len - 2), [1, 2, 3, 4], False, args.max_len)
    dev_examples = build_split(args.dev_size, (4, args.max_len - 2), [1, 2, 3, 4], False, args.max_len)
    id_test_examples = build_split(args.id_test_size, (4, 40), [1, 2, 3, 4], False, args.max_len)

    train_loader = DataLoader(DyckDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(DyckDataset(dev_examples), batch_size=args.batch_size)
    id_test_loader = DataLoader(DyckDataset(id_test_examples), batch_size=args.batch_size)

    model = DyckTransformerClassifier(
        vocab_size=len(TOKENS),
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_gold = []
        train_pred = []
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
            train_losses.append(loss.item())
            train_gold.extend(labels.cpu().tolist())
            train_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())

        dev_metrics = evaluate(model, dev_loader, device, criterion)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(accuracy_score(train_gold, train_pred)),
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"dev_loss={row['dev_loss']:.4f} dev_acc={row['dev_accuracy']:.4f}"
        )

    id_metrics = evaluate(model, id_test_loader, device, criterion)
    ood_by_depth = evaluate_ood_by_depth(model, args.ood_size_per_depth, args.max_len, args.batch_size, device)
    metrics = {
        "config": vars(args) | {"output_dir": str(args.output_dir), "device": str(device)},
        "history": history,
        "id_test": id_metrics,
        "ood_by_depth": ood_by_depth,
        "ood_overall_accuracy": float(np.mean(list(ood_by_depth.values()))),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(history, args.output_dir / "training_curves.png")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
