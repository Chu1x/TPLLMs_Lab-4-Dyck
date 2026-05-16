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

from lab4_attention import encoder_forward_with_attention
from lab4_attention_errors import WINDOW_RADIUS, corrupt_with_position
from lab4_detection import ERROR_TYPES, encode, sample_dyck
from lab4_ood import load_detection_model


def compute_rollout(attentions: list[torch.Tensor], seq_lens: list[int]) -> list[np.ndarray]:
    batch_size = attentions[0].shape[0]
    results = []
    for batch_idx in range(batch_size):
        seq_len = seq_lens[batch_idx]
        rollout = np.eye(seq_len, dtype=np.float64)
        for layer_attention in attentions:
            # Average over heads, add residual connection, then row-normalise.
            attn = layer_attention[batch_idx, :, :seq_len, :seq_len].mean(dim=0).numpy()
            attn = attn + np.eye(seq_len, dtype=np.float64)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            rollout = attn @ rollout
        results.append(rollout)
    return results


def plot_rollout(tokens: list[str], weights: np.ndarray, error_pos: int, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["tab:red" if idx == error_pos else "tab:blue" for idx in range(len(tokens))]
    ax.bar(range(len(tokens)), weights, color=colors)
    ax.set_xticks(range(len(tokens)), labels=tokens, rotation=90, fontsize=6)
    ax.set_ylabel("Rollout weight to [CLS]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/attention_rollout"))
    parser.add_argument("--examples-per-type", type=int, default=50)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    model.eval()

    examples: list[dict[str, object]] = []
    for error_type in ERROR_TYPES[1:]:
        for _ in range(args.examples_per_type):
            clean = sample_dyck((args.length, args.length), [1, 2, 3, 4], False)
            corrupted, error_pos = corrupt_with_position(clean, error_type)
            examples.append({"corrupted": corrupted, "error_type": error_type, "error_pos": error_pos})

    input_ids = []
    attention_masks = []
    seq_lens = []
    for example in examples:
        ids, mask = encode(str(example["corrupted"]), max_len)
        input_ids.append(ids)
        attention_masks.append(mask)
        seq_lens.append(sum(mask))
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_masks, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, attentions = encoder_forward_with_attention(model, input_tensor, mask_tensor)

    rollouts = compute_rollout(attentions, seq_lens)
    by_type: dict[str, list[dict[str, float]]] = {error_type: [] for error_type in ERROR_TYPES[1:]}

    for idx, example in enumerate(examples):
        error_type = str(example["error_type"])
        seq_len = seq_lens[idx]
        error_pos = int(example["error_pos"]) + 1
        weights_to_cls = rollouts[idx][0]
        non_special_positions = list(range(1, seq_len - 1))
        control_positions = [pos for pos in non_special_positions if abs(pos - error_pos) > WINDOW_RADIUS]
        control_mean = float(weights_to_cls[control_positions].mean())
        control_std = float(weights_to_cls[control_positions].std())
        error_weight = float(weights_to_cls[error_pos])
        row = {
            "error_weight": error_weight,
            "control_mean": control_mean,
            "control_std": control_std,
            "error_over_control": error_weight / control_mean if control_mean > 0 else 0.0,
            "error_percentile": float(np.mean(weights_to_cls[non_special_positions] <= error_weight)),
        }
        by_type[error_type].append(row)

    summary = {
        error_type: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for error_type, rows in by_type.items()
    }

    # Save one representative rollout plot per error type.
    for error_type in ERROR_TYPES[1:]:
        ex_idx = next(idx for idx, example in enumerate(examples) if example["error_type"] == error_type)
        sequence = str(examples[ex_idx]["corrupted"])
        tokens = ["[CLS]"] + list(sequence) + ["[SEP]"]
        plot_rollout(
            tokens,
            rollouts[ex_idx][0],
            int(examples[ex_idx]["error_pos"]) + 1,
            args.output_dir / f"{error_type}_rollout.png",
            error_type,
        )

    metrics = {
        "config": {
            "examples_per_type": args.examples_per_type,
            "length": args.length,
            "device": str(device),
        },
        "summary": summary,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
