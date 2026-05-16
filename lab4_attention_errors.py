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
from lab4_detection import CLOSE_TO_OPEN, ERROR_TYPES, OPEN_TO_CLOSE, encode, sample_dyck
from lab4_ood import load_detection_model


WINDOW_RADIUS = 3


def corrupt_with_position(clean: str, error_type: str) -> tuple[str, int]:
    chars = list(clean)

    if error_type == "E1_missing_closer":
        positions = [idx for idx, token in enumerate(chars) if token in CLOSE_TO_OPEN]
        removed_pos = random.choice(positions)
        del chars[removed_pos]
        # The error is an absent token. We anchor it to the previous observed token, or to
        # the first token if the first closer was removed.
        return "".join(chars), max(0, removed_pos - 1)

    if error_type == "E2_spurious_opener":
        pos = random.randrange(len(chars) + 1)
        chars.insert(pos, random.choice(["(", "["]))
        return "".join(chars), pos

    if error_type == "E3_type_mismatch":
        positions = [idx for idx, token in enumerate(chars) if token in CLOSE_TO_OPEN]
        pos = random.choice(positions)
        chars[pos] = ")" if chars[pos] == "]" else "]"
        return "".join(chars), pos

    if error_type == "E4_premature_close":
        stack: list[str] = []
        candidates: list[tuple[int, str]] = []
        for pos, token in enumerate(chars):
            active_openers = set(stack)
            for closer, opener in CLOSE_TO_OPEN.items():
                if opener not in active_openers:
                    candidates.append((pos, closer))
            if token in OPEN_TO_CLOSE:
                stack.append(token)
            else:
                expected = CLOSE_TO_OPEN[token]
                if stack and stack[-1] == expected:
                    stack.pop()
        pos, closer = random.choice(candidates) if candidates else (0, random.choice([")", "]"]))
        chars.insert(pos, closer)
        return "".join(chars), pos

    raise ValueError(f"Unknown error type: {error_type}")


def attention_entropy(row: np.ndarray) -> float:
    row = row[row > 0]
    return float(-(row * np.log(row)).sum())


def local_window(attention: np.ndarray, center: int, radius: int = WINDOW_RADIUS) -> np.ndarray:
    size = 2 * radius + 1
    result = np.full((size, size), np.nan, dtype=np.float64)
    positions = list(range(center - radius, center + radius + 1))
    for i, query_pos in enumerate(positions):
        for j, key_pos in enumerate(positions):
            if 0 <= query_pos < attention.shape[0] and 0 <= key_pos < attention.shape[1]:
                result[i, j] = attention[query_pos, key_pos]
    return result


def plot_local_window(matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=max(0.15, float(np.nanmax(matrix))))
    labels = [str(offset) for offset in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)]
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Key offset from error")
    ax.set_ylabel("Query offset from error")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/attention_errors"))
    parser.add_argument("--examples-per-type", type=int, default=25)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
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
            examples.append(
                {
                    "clean": clean,
                    "corrupted": corrupted,
                    "error_type": error_type,
                    "error_pos": error_pos,
                }
            )

    input_ids = []
    attention_masks = []
    for example in examples:
        ids, mask = encode(str(example["corrupted"]), max_len)
        input_ids.append(ids)
        attention_masks.append(mask)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_masks, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, attentions = encoder_forward_with_attention(model, input_tensor, mask_tensor)

    layer_attention = attentions[args.layer][:, args.head].numpy()
    by_type: dict[str, list[dict[str, float]]] = {error_type: [] for error_type in ERROR_TYPES[1:]}
    windows_by_type: dict[str, list[np.ndarray]] = {error_type: [] for error_type in ERROR_TYPES[1:]}

    for idx, example in enumerate(examples):
        seq_len = len(str(example["corrupted"])) + 2
        error_pos = int(example["error_pos"]) + 1
        attention = layer_attention[idx, :seq_len, :seq_len]
        control_candidates = [pos for pos in range(1, seq_len - 1) if abs(pos - error_pos) > WINDOW_RADIUS]
        control_pos = random.choice(control_candidates) if control_candidates else max(1, min(seq_len - 2, error_pos + 1))

        def stats_for(pos: int) -> dict[str, float]:
            return {
                "incoming_mean": float(np.delete(attention[:, pos], pos).mean()),
                "outgoing_entropy": attention_entropy(attention[pos]),
                "cls_to_position": float(attention[0, pos]),
                "position_to_cls": float(attention[pos, 0]),
                "local_mass": float(attention[pos, max(0, pos - WINDOW_RADIUS) : min(seq_len, pos + WINDOW_RADIUS + 1)].sum()),
            }

        error_stats = stats_for(error_pos)
        control_stats = stats_for(control_pos)
        row = {
            f"error_{key}": value for key, value in error_stats.items()
        } | {
            f"control_{key}": value for key, value in control_stats.items()
        }
        error_type = str(example["error_type"])
        by_type[error_type].append(row)
        windows_by_type[error_type].append(local_window(attention, error_pos))

    summary = {}
    for error_type, rows in by_type.items():
        summary[error_type] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        mean_window = np.nanmean(np.stack(windows_by_type[error_type]), axis=0)
        plot_local_window(
            mean_window,
            f"{error_type}, layer {args.layer}, head {args.head}",
            args.output_dir / f"{error_type}_local_window.png",
        )

    metrics = {
        "config": {
            "examples_per_type": args.examples_per_type,
            "length": args.length,
            "layer": args.layer,
            "head": args.head,
            "device": str(device),
        },
        "summary": summary,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
