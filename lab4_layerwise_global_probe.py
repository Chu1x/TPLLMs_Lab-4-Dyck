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
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from lab4_detection import encode, sample_dyck
from lab4_local_probe import plot_r2
from lab4_ood import load_detection_model


def cls_representations_by_layer(
    model: nn.Module,
    sequences: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    chunks_by_layer: list[list[np.ndarray]] | None = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            encoded = [encode(sequence, max_len) for sequence in batch_sequences]
            input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool, device=device)
            x = model.embedding(input_ids)
            x = model.position(x)
            outputs = [x[:, 0].detach().cpu().numpy()]
            for layer in model.encoder.layers:
                x = layer(x, src_key_padding_mask=~attention_mask)
                outputs.append(model.norm(x[:, 0]).detach().cpu().numpy())
            if chunks_by_layer is None:
                chunks_by_layer = [[] for _ in outputs]
            for layer_idx, output in enumerate(outputs):
                chunks_by_layer[layer_idx].append(output)
    assert chunks_by_layer is not None
    return [np.concatenate(chunks, axis=0) for chunks in chunks_by_layer]


def build_dataset(examples_per_depth: int, length_range: tuple[int, int]) -> tuple[list[str], np.ndarray]:
    sequences = []
    depths = []
    for depth in range(1, 8):
        for _ in range(examples_per_depth):
            sequences.append(sample_dyck(length_range, [depth], True))
            depths.append(depth)
    return sequences, np.array(depths)


def plot_accuracy(results: dict[str, float], output_path: Path) -> None:
    labels = list(results.keys())
    values = [results[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(labels)), values, marker="o")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Representation layer")
    ax.set_ylabel("Depth classification accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/layerwise_global_probe"))
    parser.add_argument("--examples-per-depth", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    sequences, depths = build_dataset(args.examples_per_depth, (4, max_len - 2))
    reps_by_layer = cls_representations_by_layer(model, sequences, max_len, args.batch_size, device)

    indices = np.arange(len(depths))
    np.random.shuffle(indices)
    test_size = int(len(indices) * args.test_fraction)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    r2_by_layer = {}
    accuracy_by_layer = {}
    for layer_idx, reps in enumerate(reps_by_layer):
        label = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        regressor = make_pipeline(StandardScaler(), LinearRegression())
        regressor.fit(reps[train_idx], depths[train_idx])
        r2_by_layer[label] = float(r2_score(depths[test_idx], regressor.predict(reps[test_idx])))

        classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        classifier.fit(reps[train_idx], depths[train_idx])
        accuracy_by_layer[label] = float(accuracy_score(depths[test_idx], classifier.predict(reps[test_idx])))

    metrics = {
        "config": {
            "examples_per_depth": args.examples_per_depth,
            "test_fraction": args.test_fraction,
            "device": str(device),
        },
        "global_depth_r2_by_layer": r2_by_layer,
        "global_depth_accuracy_by_layer": accuracy_by_layer,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_r2(r2_by_layer, args.output_dir / "global_r2_by_layer.png")
    plot_accuracy(accuracy_by_layer, args.output_dir / "global_accuracy_by_layer.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
