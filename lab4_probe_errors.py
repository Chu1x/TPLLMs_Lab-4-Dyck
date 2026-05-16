from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from lab4_attention_errors import corrupt_with_position
from lab4_detection import CLOSE_TO_OPEN, ERROR_TYPES, OPEN_TO_CLOSE, encode, sample_dyck
from lab4_local_probe import local_depths
from lab4_ood import load_detection_model


def prefix_balance(sequence: str) -> list[int]:
    depth = 0
    depths = []
    for token in sequence:
        if token in OPEN_TO_CLOSE:
            depth += 1
        elif token in CLOSE_TO_OPEN:
            depth -= 1
        depths.append(depth)
    return depths


def token_representations(
    model: nn.Module,
    sequences: list[str],
    max_len: int,
    layer_index: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            encoded = [encode(sequence, max_len) for sequence in batch_sequences]
            input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool, device=device)
            x = model.embedding(input_ids)
            x = model.position(x)
            if layer_index > 0:
                for layer in model.encoder.layers[:layer_index]:
                    x = layer(x, src_key_padding_mask=~attention_mask)
            x = x.detach().cpu().numpy()
            for row_idx, sequence in enumerate(batch_sequences):
                outputs.append(x[row_idx, 1 : 1 + len(sequence)])
    return outputs


def build_correct_sequences(examples_per_depth: int, length_range: tuple[int, int]) -> list[str]:
    sequences = []
    for depth in range(1, 8):
        for _ in range(examples_per_depth):
            sequences.append(sample_dyck(length_range, [depth], True))
    random.shuffle(sequences)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/probe_errors"))
    parser.add_argument("--train-examples-per-depth", type=int, default=300)
    parser.add_argument("--test-examples-per-depth", type=int, default=100)
    parser.add_argument("--error-examples-per-type", type=int, default=100)
    parser.add_argument("--layer-index", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    length_range = (4, max_len - 2)

    train_sequences = build_correct_sequences(args.train_examples_per_depth, length_range)
    train_reps_by_seq = token_representations(model, train_sequences, max_len, args.layer_index, args.batch_size, device)
    x_train = np.concatenate(train_reps_by_seq, axis=0)
    y_train = np.array([depth for seq in train_sequences for depth in local_depths(seq)], dtype=np.float64)
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    probe.fit(x_train, y_train)

    correct_test = build_correct_sequences(args.test_examples_per_depth, length_range)
    correct_reps_by_seq = token_representations(model, correct_test, max_len, args.layer_index, args.batch_size, device)
    x_correct = np.concatenate(correct_reps_by_seq, axis=0)
    y_correct = np.array([depth for seq in correct_test for depth in local_depths(seq)], dtype=np.float64)
    correct_pred = probe.predict(x_correct)
    correct_r2 = r2_score(y_correct, correct_pred)
    correct_mae = float(np.mean(np.abs(correct_pred - y_correct)))

    error_examples = []
    for error_type in ERROR_TYPES[1:]:
        for _ in range(args.error_examples_per_type):
            clean = sample_dyck((40, min(78, max_len - 2)), [1, 2, 3, 4], False)
            corrupted, error_pos = corrupt_with_position(clean, error_type)
            error_examples.append({"sequence": corrupted, "error_type": error_type, "error_pos": error_pos})

    error_sequences = [str(example["sequence"]) for example in error_examples]
    error_reps_by_seq = token_representations(model, error_sequences, max_len, args.layer_index, args.batch_size, device)

    segment_rows = {error_type: {"before": [], "at": [], "after": []} for error_type in ERROR_TYPES[1:]}
    localization_ranks = {error_type: [] for error_type in ERROR_TYPES[1:]}
    localization_top1 = {error_type: [] for error_type in ERROR_TYPES[1:]}
    all_error_targets = []
    all_error_predictions = []

    for example, reps in zip(error_examples, error_reps_by_seq, strict=True):
        sequence = str(example["sequence"])
        error_type = str(example["error_type"])
        error_pos = int(example["error_pos"])
        targets = np.array(prefix_balance(sequence), dtype=np.float64)
        predictions = probe.predict(reps)
        residuals = np.abs(predictions - targets)
        all_error_targets.extend(targets.tolist())
        all_error_predictions.extend(predictions.tolist())

        before = residuals[:error_pos]
        at = residuals[error_pos : error_pos + 1]
        after = residuals[error_pos + 1 :]
        if len(before):
            segment_rows[error_type]["before"].extend(before.tolist())
        segment_rows[error_type]["at"].extend(at.tolist())
        if len(after):
            segment_rows[error_type]["after"].extend(after.tolist())

        ranked_positions = np.argsort(-residuals)
        rank = int(np.where(ranked_positions == error_pos)[0][0]) + 1
        localization_ranks[error_type].append(rank)
        localization_top1[error_type].append(int(rank == 1))

    segment_summary = {
        error_type: {
            segment: float(np.mean(values)) if values else None
            for segment, values in segments.items()
        }
        for error_type, segments in segment_rows.items()
    }
    localization_summary = {
        error_type: {
            "mean_rank": float(np.mean(localization_ranks[error_type])),
            "top1_accuracy": float(np.mean(localization_top1[error_type])),
        }
        for error_type in ERROR_TYPES[1:]
    }

    metrics = {
        "config": {
            "train_examples_per_depth": args.train_examples_per_depth,
            "test_examples_per_depth": args.test_examples_per_depth,
            "error_examples_per_type": args.error_examples_per_type,
            "layer_index": args.layer_index,
            "device": str(device),
        },
        "correct_strings": {
            "r2": float(correct_r2),
            "mae": correct_mae,
        },
        "erroneous_strings": {
            "r2_against_corrupted_prefix_balance": float(r2_score(all_error_targets, all_error_predictions)),
            "mae_by_segment": segment_summary,
            "localization_from_probe_residual": localization_summary,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
