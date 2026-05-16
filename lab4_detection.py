from __future__ import annotations

import argparse
import json
import math
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAIRS = [("(", ")"), ("[", "]")]
OPEN_TO_CLOSE = dict(PAIRS)
CLOSE_TO_OPEN = {close: open_ for open_, close in PAIRS}
TOKENS = ["(", ")", "[", "]", "[PAD]", "[CLS]", "[SEP]"]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
PAD_ID = TOKEN_TO_ID["[PAD]"]
CLS_ID = TOKEN_TO_ID["[CLS]"]
SEP_ID = TOKEN_TO_ID["[SEP]"]
ERROR_TYPES = ["no_error", "E1_missing_closer", "E2_spurious_opener", "E3_type_mismatch", "E4_premature_close"]


def max_nesting_depth(sequence: str) -> int:
    depth = 0
    max_depth = 0
    for token in sequence:
        if token in OPEN_TO_CLOSE:
            depth += 1
            max_depth = max(max_depth, depth)
        else:
            depth -= 1
    return max_depth


def generate_dyck(length: int, max_depth: int, k: int = 2) -> str | None:
    """Return a random Dyck string of exactly length tokens with depth <= max_depth."""
    assert length % 2 == 0
    stack: list[tuple[str, str]] = []
    result: list[str] = []

    for step in range(length):
        remaining = length - step
        must_close = len(stack) == remaining
        can_open = len(stack) < max_depth and remaining > len(stack) + 1

        choices = []
        if can_open and not must_close:
            choices.append("open")
        if stack:
            choices.append("close")
        if not choices:
            return None

        if random.choice(choices) == "open":
            pair = random.choice(PAIRS[:k])
            stack.append(pair)
            result.append(pair[0])
        else:
            pair = stack.pop()
            result.append(pair[1])

    return "".join(result) if not stack else None


def sample_dyck(length_range: tuple[int, int], depths: list[int], exact_depth: bool) -> str:
    while True:
        length = random.randrange(length_range[0] // 2, length_range[1] // 2 + 1) * 2
        depth = random.choice(depths)
        sequence = generate_dyck(length, depth)
        if sequence is None:
            continue
        actual_depth = max_nesting_depth(sequence)
        if exact_depth and actual_depth == depth:
            return sequence
        if not exact_depth and actual_depth <= max(depths):
            return sequence


def corrupt(sequence: str, error_type: str) -> str:
    chars = list(sequence)

    if error_type == "E1_missing_closer":
        positions = [idx for idx, token in enumerate(chars) if token in CLOSE_TO_OPEN]
        del chars[random.choice(positions)]
        return "".join(chars)

    if error_type == "E2_spurious_opener":
        pos = random.randrange(len(chars) + 1)
        chars.insert(pos, random.choice(["(", "["]))
        return "".join(chars)

    if error_type == "E3_type_mismatch":
        positions = [idx for idx, token in enumerate(chars) if token in CLOSE_TO_OPEN]
        pos = random.choice(positions)
        chars[pos] = ")" if chars[pos] == "]" else "]"
        return "".join(chars)

    if error_type == "E4_premature_close":
        stack: list[str] = []
        candidates: list[tuple[int, str]] = []
        for pos, token in enumerate(chars):
            for closer, opener in CLOSE_TO_OPEN.items():
                if opener not in stack:
                    candidates.append((pos, closer))
            if token in OPEN_TO_CLOSE:
                stack.append(token)
            else:
                expected = CLOSE_TO_OPEN[token]
                if expected in stack:
                    stack.remove(expected)
        if candidates:
            pos, closer = random.choice(candidates)
        else:
            pos, closer = 0, random.choice([")", "]"])
        chars.insert(pos, closer)
        return "".join(chars)

    raise ValueError(f"Unknown error type: {error_type}")


def encode(sequence: str, max_len: int) -> tuple[list[int], list[int]]:
    ids = [CLS_ID] + [TOKEN_TO_ID[token] for token in sequence] + [SEP_ID]
    ids = ids[:max_len]
    attention_mask = [1] * len(ids)
    pad_len = max_len - len(ids)
    ids.extend([PAD_ID] * pad_len)
    attention_mask.extend([0] * pad_len)
    return ids, attention_mask


@dataclass
class Example:
    sequence: str
    label: int
    error_type: str
    input_ids: list[int]
    attention_mask: list[int]


def build_split(size: int, length_range: tuple[int, int], depths: list[int], exact_depth: bool, max_len: int) -> list[Example]:
    examples: list[Example] = []
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
        sequence = clean if error_type == "no_error" else corrupt(clean, error_type)
        input_ids, attention_mask = encode(sequence, max_len)
        examples.append(
            Example(
                sequence=sequence,
                label=int(error_type != "no_error"),
                error_type=error_type,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
    return examples


class DyckDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.bool),
            "label": torch.tensor(example.label, dtype=torch.long),
            "error_type": example.error_type,
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DyckTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
        self.position = SinusoidalPositionalEncoding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.position(x)
        padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        cls = self.norm(x[:, 0])
        return self.classifier(cls)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, object]:
    model.eval()
    losses = []
    labels = []
    predictions = []
    error_types = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
            losses.append(loss.item())
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            error_types.extend(batch["error_type"])

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "labels": labels,
        "predictions": predictions,
        "error_types": error_types,
    }


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["dev_loss"] for entry in history], label="dev")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [entry["train_accuracy"] for entry in history], label="train")
    axes[1].plot(epochs, [entry["dev_accuracy"] for entry in history], label="dev")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(2), labels=["pred no error", "pred error"])
    ax.set_yticks(range(len(ERROR_TYPES)), labels=ERROR_TYPES)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_title("Detection predictions by error type")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--dev-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/detection"))
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

    train_loader = DataLoader(DyckDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(DyckDataset(dev_examples), batch_size=args.batch_size)
    test_loader = DataLoader(DyckDataset(test_examples), batch_size=args.batch_size)

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

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_labels = []
        train_predictions = []
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
            train_labels.extend(labels.cpu().tolist())
            train_predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())

        train_accuracy = accuracy_score(train_labels, train_predictions)
        dev_metrics = evaluate(model, dev_loader, device, criterion)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(train_accuracy),
            "dev_loss": float(dev_metrics["loss"]),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} dev_loss={row['dev_loss']:.4f} "
            f"dev_acc={row['dev_accuracy']:.4f} dev_f1={row['dev_macro_f1']:.4f}"
        )

    test_metrics = evaluate(model, test_loader, device, criterion)
    error_type_to_index = {name: idx for idx, name in enumerate(ERROR_TYPES)}
    per_type_matrix = np.zeros((len(ERROR_TYPES), 2), dtype=int)
    for error_type, prediction in zip(test_metrics["error_types"], test_metrics["predictions"], strict=True):
        per_type_matrix[error_type_to_index[error_type], prediction] += 1

    binary_cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"]).tolist()
    metrics = {
        "config": vars(args) | {"output_dir": str(args.output_dir), "device": str(device)},
        "history": history,
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "binary_confusion_matrix": binary_cm,
            "error_type_prediction_matrix": per_type_matrix.tolist(),
            "error_type_rows": ERROR_TYPES,
            "prediction_columns": ["pred_no_error", "pred_error"],
        },
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(history, args.output_dir / "training_curves.png")
    plot_confusion(per_type_matrix, args.output_dir / "confusion_by_error_type.png")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
