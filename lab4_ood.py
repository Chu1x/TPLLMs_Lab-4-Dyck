from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from lab4_detection import (
    DyckDataset,
    DyckTransformerClassifier,
    build_split,
)


def load_detection_model(metrics_path: Path, model_path: Path, device: torch.device) -> tuple[DyckTransformerClassifier, int]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics["config"]
    max_len = int(config["max_len"])
    model = DyckTransformerClassifier(
        vocab_size=7,
        max_len=max_len,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["layers"]),
        num_heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, max_len


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[int]]:
    gold: list[int] = []
    pred: list[int] = []
    lengths: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            gold.extend(batch["label"].tolist())
            pred.extend(logits.argmax(dim=-1).cpu().tolist())
            lengths.extend((attention_mask.sum(dim=1).cpu().numpy() - 2).tolist())
    return gold, pred, lengths


def plot_depth_accuracy(depth_results: dict[str, float], output_path: Path) -> None:
    depths = [int(depth) for depth in depth_results]
    accuracies = [depth_results[str(depth)] for depth in depths]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(depths, accuracies, marker="o")
    ax.set_xlabel("Maximum nesting depth")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(depths)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_length_accuracy(lengths: list[int], gold: list[int], pred: list[int], output_path: Path) -> dict[str, float]:
    bins = [(40, 49), (50, 59), (60, 69), (70, 78)]
    results = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = []
    ys = []
    for low, high in bins:
        indices = [idx for idx, length in enumerate(lengths) if low <= length <= high]
        if not indices:
            continue
        bin_gold = [gold[idx] for idx in indices]
        bin_pred = [pred[idx] for idx in indices]
        acc = accuracy_score(bin_gold, bin_pred)
        label = f"{low}-{high}"
        results[label] = acc
        xs.append(label)
        ys.append(acc)
    ax.bar(xs, ys)
    ax.set_xlabel("Sequence length bin")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ood"))
    parser.add_argument("--examples-per-depth", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)

    # The trained detector uses max_len=80 including [CLS] and [SEP], so raw strings longer
    # than 78 would be truncated. We cap OOD raw length at 78 to evaluate depth rather than
    # truncation artefacts.
    max_raw_len = max_len - 2
    length_range = (40, min(78, max_raw_len))

    depth_results: dict[str, float] = {}
    all_gold: list[int] = []
    all_pred: list[int] = []
    all_lengths: list[int] = []

    for depth in [5, 6, 7]:
        examples = build_split(args.examples_per_depth, length_range, [depth], True, max_len)
        loader = DataLoader(DyckDataset(examples), batch_size=args.batch_size)
        gold, pred, lengths = predict(model, loader, device)
        depth_results[str(depth)] = accuracy_score(gold, pred)
        all_gold.extend(gold)
        all_pred.extend(pred)
        all_lengths.extend(lengths)

    length_results = plot_length_accuracy(all_lengths, all_gold, all_pred, args.output_dir / "accuracy_by_length.png")
    plot_depth_accuracy(depth_results, args.output_dir / "accuracy_by_depth.png")
    metrics = {
        "config": {
            "examples_per_depth": args.examples_per_depth,
            "length_range": list(length_range),
            "device": str(device),
            "model_path": str(args.model_path),
        },
        "overall_accuracy": accuracy_score(all_gold, all_pred),
        "accuracy_by_depth": depth_results,
        "accuracy_by_length": length_results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
