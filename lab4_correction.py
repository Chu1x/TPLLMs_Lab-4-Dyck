from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, Dataset

from lab4_detection import (
    CLOSE_TO_OPEN,
    ERROR_TYPES,
    OPEN_TO_CLOSE,
    PAD_ID,
    PAIRS,
    TOKENS,
    TOKEN_TO_ID,
    DyckTransformerClassifier,
    encode,
    sample_dyck,
)
from lab4_pda_baseline import is_dyck


IGNORE_INDEX = -100
LABELS = [
    "OK",
    "DELETE",
    "INSERT_(",
    "INSERT_)",
    "INSERT_[",
    "INSERT_]",
    "REPLACE_(",
    "REPLACE_)",
    "REPLACE_[",
    "REPLACE_]",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class CorrectionExample:
    clean: str
    corrupted: str
    error_type: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def make_ok_labels(sequence: str, max_len: int) -> list[int]:
    labels = [IGNORE_INDEX] + [LABEL_TO_ID["OK"]] * len(sequence) + [IGNORE_INDEX]
    return labels[:max_len] + [IGNORE_INDEX] * max(0, max_len - len(labels))


def corrupt_with_labels(clean: str, error_type: str, max_len: int) -> tuple[str, list[int]]:
    chars = list(clean)

    if error_type == "no_error":
        return clean, make_ok_labels(clean, max_len)

    if error_type == "E1_missing_closer":
        positions = [idx for idx, token in enumerate(chars) if token in CLOSE_TO_OPEN]
        removed_pos = random.choice(positions)
        removed_token = chars.pop(removed_pos)
        corrupted = "".join(chars)
        labels = make_ok_labels(corrupted, max_len)
        # INSERT(x) is interpreted as "insert x after this token".
        anchor_pos = max(0, removed_pos - 1)
        labels[1 + anchor_pos] = LABEL_TO_ID[f"INSERT_{removed_token}"]
        return corrupted, labels

    if error_type == "E2_spurious_opener":
        insert_pos = random.randrange(len(chars) + 1)
        chars.insert(insert_pos, random.choice(["(", "["]))
        corrupted = "".join(chars)
        labels = make_ok_labels(corrupted, max_len)
        labels[1 + insert_pos] = LABEL_TO_ID["DELETE"]
        return corrupted, labels

    if error_type == "E3_type_mismatch":
        positions = [idx for idx, token in enumerate(chars) if token in CLOSE_TO_OPEN]
        replace_pos = random.choice(positions)
        correct_token = chars[replace_pos]
        chars[replace_pos] = ")" if correct_token == "]" else "]"
        corrupted = "".join(chars)
        labels = make_ok_labels(corrupted, max_len)
        labels[1 + replace_pos] = LABEL_TO_ID[f"REPLACE_{correct_token}"]
        return corrupted, labels

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
        insert_pos, closer = random.choice(candidates) if candidates else (0, random.choice([")", "]"]))
        chars.insert(insert_pos, closer)
        corrupted = "".join(chars)
        labels = make_ok_labels(corrupted, max_len)
        labels[1 + insert_pos] = LABEL_TO_ID["DELETE"]
        return corrupted, labels

    raise ValueError(f"Unknown error type: {error_type}")


def apply_correction(sequence: str, label_ids: list[int]) -> str:
    output: list[str] = []
    for token, label_id in zip(sequence, label_ids, strict=False):
        label = ID_TO_LABEL[int(label_id)]
        if label == "OK":
            output.append(token)
        elif label == "DELETE":
            continue
        elif label.startswith("REPLACE_"):
            output.append(label.removeprefix("REPLACE_"))
        elif label.startswith("INSERT_"):
            output.append(token)
            output.append(label.removeprefix("INSERT_"))
        else:
            raise ValueError(f"Unknown label: {label}")
    return "".join(output)


def single_edit_repair(sequence: str) -> str:
    """Return a valid Dyck string reachable by one edit, or the original string if none is found."""
    if is_dyck(sequence):
        return sequence

    candidates: list[tuple[int, str]] = []
    brackets = ["(", ")", "[", "]"]

    for pos in range(len(sequence)):
        candidates.append((0, sequence[:pos] + sequence[pos + 1 :]))

    for pos in range(len(sequence) + 1):
        for bracket in brackets:
            candidates.append((1, sequence[:pos] + bracket + sequence[pos:]))

    for pos, token in enumerate(sequence):
        for bracket in brackets:
            if bracket != token:
                candidates.append((2, sequence[:pos] + bracket + sequence[pos + 1 :]))

    valid_candidates = [(priority, candidate) for priority, candidate in candidates if is_dyck(candidate)]
    if not valid_candidates:
        return sequence
    valid_candidates.sort(key=lambda item: (item[0], abs(len(item[1]) - len(sequence)), item[1]))
    return valid_candidates[0][1]


def build_split(size: int, length_range: tuple[int, int], depths: list[int], exact_depth: bool, max_len: int) -> list[CorrectionExample]:
    examples: list[CorrectionExample] = []
    num_clean = size // 2
    num_corrupt = size - num_clean
    corrupt_types = ERROR_TYPES[1:]
    per_corrupt_type = num_corrupt // len(corrupt_types)
    remainder = num_corrupt % len(corrupt_types)
    schedule = ["no_error"] * num_clean
    for idx, error_type in enumerate(corrupt_types):
        schedule.extend([error_type] * (per_corrupt_type + int(idx < remainder)))
    random.shuffle(schedule)

    for error_type in schedule:
        clean = sample_dyck(length_range, depths, exact_depth)
        corrupted, labels = corrupt_with_labels(clean, error_type, max_len)
        input_ids, attention_mask = encode(corrupted, max_len)
        examples.append(
            CorrectionExample(
                clean=clean,
                corrupted=corrupted,
                error_type=error_type,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        )
    return examples


def compute_class_weights(examples: list[CorrectionExample], power: float) -> torch.Tensor:
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for example in examples:
        for label in example.labels:
            if label != IGNORE_INDEX:
                counts[label] += 1
    weights = np.zeros(len(LABELS), dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = (counts[nonzero].sum() / counts[nonzero]) ** power
    weights[nonzero] /= weights[nonzero].mean()
    return torch.tensor(weights, dtype=torch.float32)


class CorrectionDataset(Dataset):
    def __init__(self, examples: list[CorrectionExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.bool),
            "labels": torch.tensor(example.labels, dtype=torch.long),
            "clean": example.clean,
            "corrupted": example.corrupted,
            "error_type": example.error_type,
        }


class DyckCorrectionModel(nn.Module):
    def __init__(self, max_len: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        base = DyckTransformerClassifier(
            vocab_size=len(TOKENS),
            max_len=max_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.embedding = base.embedding
        self.position = base.position
        self.encoder = base.encoder
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.position(x)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = self.norm(x)
        return self.classifier(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, object]:
    model.eval()
    losses = []
    gold_labels: list[int] = []
    pred_labels: list[int] = []
    exact = []
    by_type: dict[str, list[int]] = {error_type: [] for error_type in ERROR_TYPES}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            losses.append(criterion(logits.transpose(1, 2), labels).item())
            predictions = logits.argmax(dim=-1).cpu().numpy()
            gold = labels.cpu().numpy()

            for row_idx, corrupted in enumerate(batch["corrupted"]):
                seq_len = len(corrupted)
                gold_row = gold[row_idx, 1 : 1 + seq_len]
                pred_row = predictions[row_idx, 1 : 1 + seq_len]
                mask = gold_row != IGNORE_INDEX
                gold_labels.extend(gold_row[mask].tolist())
                pred_labels.extend(pred_row[mask].tolist())
                predicted_clean = apply_correction(corrupted, pred_row.tolist())
                is_exact = int(predicted_clean == batch["clean"][row_idx])
                exact.append(is_exact)
                by_type[batch["error_type"][row_idx]].append(is_exact)

    return {
        "loss": float(np.mean(losses)),
        "token_accuracy": accuracy_score(gold_labels, pred_labels),
        "exact_match": float(np.mean(exact)),
        "exact_match_by_type": {key: float(np.mean(values)) if values else 0.0 for key, values in by_type.items()},
    }


def evaluate_baselines(examples: list[CorrectionExample]) -> dict[str, object]:
    rows = {
        "no_edit": {
            "exact": [],
            "valid": [],
            "by_type": {error_type: [] for error_type in ERROR_TYPES},
            "valid_by_type": {error_type: [] for error_type in ERROR_TYPES},
        },
        "pda_single_edit": {
            "exact": [],
            "valid": [],
            "by_type": {error_type: [] for error_type in ERROR_TYPES},
            "valid_by_type": {error_type: [] for error_type in ERROR_TYPES},
        },
    }

    for example in examples:
        predictions = {
            "no_edit": example.corrupted,
            "pda_single_edit": single_edit_repair(example.corrupted),
        }
        for name, prediction in predictions.items():
            exact = int(prediction == example.clean)
            valid = int(is_dyck(prediction))
            rows[name]["exact"].append(exact)
            rows[name]["valid"].append(valid)
            rows[name]["by_type"][example.error_type].append(exact)
            rows[name]["valid_by_type"][example.error_type].append(valid)

    output = {}
    for name, row in rows.items():
        output[name] = {
            "exact_match": float(np.mean(row["exact"])),
            "valid_dyck_rate": float(np.mean(row["valid"])),
            "exact_match_by_type": {
                error_type: float(np.mean(values)) if values else 0.0
                for error_type, values in row["by_type"].items()
            },
            "valid_dyck_rate_by_type": {
                error_type: float(np.mean(values)) if values else 0.0
                for error_type, values in row["valid_by_type"].items()
            },
        }
        corrupt_values = [
            value
            for error_type, values in row["by_type"].items()
            if error_type != "no_error"
            for value in values
        ]
        output[name]["corrupted_only_exact_match"] = float(np.mean(corrupt_values)) if corrupt_values else 0.0
    return output


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["dev_loss"] for entry in history], label="dev")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [entry["dev_token_accuracy"] for entry in history], label="token accuracy")
    axes[1].plot(epochs, [entry["dev_exact_match"] for entry in history], label="exact match")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Development score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--dev-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/correction"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_examples = build_split(args.train_size, (4, 40), [1, 2, 3, 4], False, args.max_len)
    dev_examples = build_split(args.dev_size, (4, 40), [1, 2, 3, 4], False, args.max_len)
    test_examples = build_split(args.test_size, (4, 40), [1, 2, 3, 4], False, args.max_len)
    train_loader = DataLoader(CorrectionDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(CorrectionDataset(dev_examples), batch_size=args.batch_size)
    test_loader = DataLoader(CorrectionDataset(test_examples), batch_size=args.batch_size)

    model = DyckCorrectionModel(args.max_len, args.hidden_dim, args.layers, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    class_weights = None if args.no_class_weights else compute_class_weights(train_examples, args.weight_power).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.transpose(1, 2), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        dev_metrics = evaluate(model, dev_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_loss": dev_metrics["loss"],
            "dev_token_accuracy": dev_metrics["token_accuracy"],
            "dev_exact_match": dev_metrics["exact_match"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"dev_loss={row['dev_loss']:.4f} token_acc={row['dev_token_accuracy']:.4f} "
            f"exact={row['dev_exact_match']:.4f}"
        )

    test_metrics = evaluate(model, test_loader, device, criterion)
    baseline_metrics = evaluate_baselines(test_examples)
    metrics = {
        "config": vars(args) | {"output_dir": str(args.output_dir), "device": str(device)},
        "labels": LABELS,
        "history": history,
        "test": test_metrics,
        "baselines": baseline_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(history, args.output_dir / "training_curves.png")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
