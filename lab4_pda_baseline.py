from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from lab4_detection import CLOSE_TO_OPEN, ERROR_TYPES, OPEN_TO_CLOSE, build_split


def is_dyck(sequence: str) -> bool:
    stack: list[str] = []
    for token in sequence:
        if token in OPEN_TO_CLOSE:
            stack.append(token)
        elif token in CLOSE_TO_OPEN:
            if not stack:
                return False
            opener = stack.pop()
            if OPEN_TO_CLOSE[opener] != token:
                return False
        else:
            raise ValueError(f"Unknown token: {token}")
    return not stack


def parity_predict_error(sequence: str) -> int:
    return int(len(sequence) % 2 == 1)


def evaluate_pda(size: int, length_range: tuple[int, int], depths: list[int], exact_depth: bool, max_len: int) -> dict[str, object]:
    examples = build_split(size, length_range, depths, exact_depth, max_len)
    gold = [example.label for example in examples]
    pda_pred = [0 if is_dyck(example.sequence) else 1 for example in examples]
    parity_pred = [parity_predict_error(example.sequence) for example in examples]

    by_type: dict[str, list[int]] = {error_type: [] for error_type in ERROR_TYPES}
    parity_by_type: dict[str, list[int]] = {error_type: [] for error_type in ERROR_TYPES}
    for example, prediction, parity_prediction in zip(examples, pda_pred, parity_pred, strict=True):
        by_type[example.error_type].append(int(prediction == example.label))
        parity_by_type[example.error_type].append(int(parity_prediction == example.label))

    return {
        "accuracy": accuracy_score(gold, pda_pred),
        "confusion_matrix": confusion_matrix(gold, pda_pred).tolist(),
        "accuracy_by_type": {
            error_type: float(np.mean(values)) if values else 0.0
            for error_type, values in by_type.items()
        },
        "parity_only": {
            "accuracy": accuracy_score(gold, parity_pred),
            "confusion_matrix": confusion_matrix(gold, parity_pred).tolist(),
            "accuracy_by_type": {
                error_type: float(np.mean(values)) if values else 0.0
                for error_type, values in parity_by_type.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pda"))
    parser.add_argument("--id-size", type=int, default=1000)
    parser.add_argument("--ood-size-per-depth", type=int, default=1000)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    id_results = evaluate_pda(args.id_size, (4, 40), [1, 2, 3, 4], False, args.max_len)
    ood_by_depth = {}
    for depth in [5, 6, 7]:
        # Match the Transformer OOD script: raw strings are capped at max_len - 2 to avoid
        # truncation when [CLS] and [SEP] are added.
        ood_by_depth[str(depth)] = evaluate_pda(
            args.ood_size_per_depth,
            (40, args.max_len - 2),
            [depth],
            True,
            args.max_len,
        )

    metrics = {
        "config": {
            "id_size": args.id_size,
            "ood_size_per_depth": args.ood_size_per_depth,
            "max_len": args.max_len,
        },
        "in_distribution": id_results,
        "ood_by_depth": ood_by_depth,
        "ood_overall_accuracy": float(np.mean([ood_by_depth[str(depth)]["accuracy"] for depth in [5, 6, 7]])),
        "parity_only": {
            "in_distribution_accuracy": id_results["parity_only"]["accuracy"],
            "ood_by_depth_accuracy": {
                str(depth): ood_by_depth[str(depth)]["parity_only"]["accuracy"]
                for depth in [5, 6, 7]
            },
            "ood_overall_accuracy": float(
                np.mean([ood_by_depth[str(depth)]["parity_only"]["accuracy"] for depth in [5, 6, 7]])
            ),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
