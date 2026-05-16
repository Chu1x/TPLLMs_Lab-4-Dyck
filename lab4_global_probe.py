from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from lab4_detection import encode, sample_dyck
from lab4_ood import load_detection_model


def cls_representations(
    model: nn.Module,
    sequences: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    reps = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            encoded = [encode(sequence, max_len) for sequence in batch_sequences]
            input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool, device=device)
            x = model.embedding(input_ids)
            x = model.position(x)
            x = model.encoder(x, src_key_padding_mask=~attention_mask)
            cls = model.norm(x[:, 0])
            reps.append(cls.cpu().numpy())
    return np.concatenate(reps, axis=0)


def build_depth_dataset(examples_per_depth: int, length_range: tuple[int, int]) -> tuple[list[str], np.ndarray]:
    sequences = []
    labels = []
    for depth in range(1, 8):
        for _ in range(examples_per_depth):
            sequence = sample_dyck(length_range, [depth], True)
            sequences.append(sequence)
            labels.append(depth)
    return sequences, np.array(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/global_probe"))
    parser.add_argument("--examples-per-depth", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    length_range = (4, max_len - 2)

    sequences, depths = build_depth_dataset(args.examples_per_depth, length_range)
    reps = cls_representations(model, sequences, max_len, args.batch_size, device)

    indices = np.arange(len(depths))
    np.random.shuffle(indices)
    test_size = int(len(indices) * args.test_fraction)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    x_train, x_test = reps[train_idx], reps[test_idx]
    y_train, y_test = depths[train_idx], depths[test_idx]

    regressor = make_pipeline(StandardScaler(), LinearRegression())
    regressor.fit(x_train, y_train)
    y_pred_reg = regressor.predict(x_test)
    r2 = r2_score(y_test, y_pred_reg)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    classifier.fit(x_train, y_train)
    y_pred_cls = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_cls)
    cm = confusion_matrix(y_test, y_pred_cls, labels=list(range(1, 8)))
    per_depth_accuracy = {
        str(depth): float(accuracy_score(y_test[y_test == depth], y_pred_cls[y_test == depth]))
        for depth in range(1, 8)
    }

    metrics = {
        "config": {
            "examples_per_depth": args.examples_per_depth,
            "test_fraction": args.test_fraction,
            "length_range": list(length_range),
            "device": str(device),
        },
        "regression_r2": float(r2),
        "classification_accuracy": float(accuracy),
        "classification_confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(range(1, 8)),
        "per_depth_accuracy": per_depth_accuracy,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
