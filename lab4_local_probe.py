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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from lab4_detection import OPEN_TO_CLOSE, encode, sample_dyck
from lab4_ood import load_detection_model


def local_depths(sequence: str) -> list[int]:
    depth = 0
    depths = []
    for token in sequence:
        if token in OPEN_TO_CLOSE:
            depth += 1
        else:
            depth -= 1
        depths.append(depth)
    return depths


def collect_layer_representations(
    model: nn.Module,
    sequences: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], np.ndarray]:
    all_layer_reps: list[list[np.ndarray]] | None = None
    all_depths = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            encoded = [encode(sequence, max_len) for sequence in batch_sequences]
            input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool, device=device)
            padding_mask = ~attention_mask

            x = model.embedding(input_ids)
            x = model.position(x)
            layer_outputs = [x.detach().cpu()]
            for layer in model.encoder.layers:
                x = layer(x, src_key_padding_mask=padding_mask)
                layer_outputs.append(x.detach().cpu())

            if all_layer_reps is None:
                all_layer_reps = [[] for _ in layer_outputs]

            for layer_idx, output in enumerate(layer_outputs):
                token_reps = []
                for row_idx, sequence in enumerate(batch_sequences):
                    seq_len = len(sequence)
                    token_reps.append(output[row_idx, 1 : 1 + seq_len].numpy())
                all_layer_reps[layer_idx].append(np.concatenate(token_reps, axis=0))

            for sequence in batch_sequences:
                all_depths.extend(local_depths(sequence))

    assert all_layer_reps is not None
    return [np.concatenate(layer_chunks, axis=0) for layer_chunks in all_layer_reps], np.array(all_depths)


def build_dataset(examples_per_depth: int, length_range: tuple[int, int]) -> list[str]:
    sequences = []
    for depth in range(1, 8):
        for _ in range(examples_per_depth):
            sequences.append(sample_dyck(length_range, [depth], True))
    random.shuffle(sequences)
    return sequences


def plot_r2(layer_results: dict[str, float], output_path: Path) -> None:
    labels = list(layer_results.keys())
    values = [layer_results[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(labels)), values, marker="o")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Representation layer")
    ax.set_ylabel("Probe R2")
    ax.set_ylim(min(0, min(values) - 0.05), 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local_probe"))
    parser.add_argument("--examples-per-depth", type=int, default=300)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    length_range = (4, max_len - 2)
    sequences = build_dataset(args.examples_per_depth, length_range)
    layer_reps, depths = collect_layer_representations(model, sequences, max_len, args.batch_size, device)

    indices = np.arange(len(depths))
    np.random.shuffle(indices)
    test_size = int(len(indices) * args.test_fraction)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    results = {}
    for layer_idx, reps in enumerate(layer_reps):
        label = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(reps[train_idx], depths[train_idx])
        predictions = probe.predict(reps[test_idx])
        results[label] = float(r2_score(depths[test_idx], predictions))

    best_layer = max(results, key=results.get)
    metrics = {
        "config": {
            "examples_per_depth": args.examples_per_depth,
            "test_fraction": args.test_fraction,
            "length_range": list(length_range),
            "num_token_examples": int(len(depths)),
            "device": str(device),
        },
        "r2_by_layer": results,
        "best_layer": best_layer,
        "best_r2": results[best_layer],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_r2(results, args.output_dir / "r2_by_layer.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
