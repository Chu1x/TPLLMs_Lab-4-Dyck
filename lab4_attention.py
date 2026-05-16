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
from torch import nn

from lab4_detection import CLOSE_TO_OPEN, OPEN_TO_CLOSE, TOKEN_TO_ID, encode, sample_dyck
from lab4_ood import load_detection_model


def matched_pairs(sequence: str) -> list[tuple[int, int]]:
    stack: list[tuple[str, int]] = []
    pairs: list[tuple[int, int]] = []
    for idx, token in enumerate(sequence):
        if token in OPEN_TO_CLOSE:
            stack.append((token, idx))
        elif token in CLOSE_TO_OPEN:
            opener, opener_idx = stack.pop()
            if OPEN_TO_CLOSE[opener] != token:
                raise ValueError(f"Invalid Dyck sequence: {sequence}")
            pairs.append((opener_idx, idx))
    if stack:
        raise ValueError(f"Invalid Dyck sequence: {sequence}")
    return pairs


def activation(layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> torch.Tensor:
    if layer.activation_relu_or_gelu == 1:
        return torch.relu(x)
    if layer.activation_relu_or_gelu == 2:
        return torch.nn.functional.gelu(x)
    return layer.activation(x)


def encoder_forward_with_attention(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = model.embedding(input_ids)
    x = model.position(x)
    padding_mask = ~attention_mask
    attentions: list[torch.Tensor] = []

    for layer in model.encoder.layers:
        if layer.norm_first:
            y = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                y,
                y,
                y,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=False,
            )
            x = x + layer.dropout1(attn_out)
            y = layer.norm2(x)
            ff = layer.linear2(layer.dropout(activation(layer, layer.linear1(y))))
            x = x + layer.dropout2(ff)
        else:
            attn_out, attn_weights = layer.self_attn(
                x,
                x,
                x,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=False,
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            ff = layer.linear2(layer.dropout(activation(layer, layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff))
        attentions.append(attn_weights.detach().cpu())

    return x, attentions


def plot_attention(matrix: np.ndarray, tokens: list[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=max(0.15, float(matrix.max())), cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_xticks(range(len(tokens)), labels=tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tokens)), labels=tokens, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/detection/metrics.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/detection/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/attention"))
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, max_len = load_detection_model(args.metrics_path, args.model_path, device)
    model.eval()

    sequences = [sample_dyck((args.length, args.length), [1, 2, 3, 4], False) for _ in range(args.num_examples)]
    input_ids = []
    attention_masks = []
    for sequence in sequences:
        ids, mask = encode(sequence, max_len)
        input_ids.append(ids)
        attention_masks.append(mask)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_masks, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, attentions = encoder_forward_with_attention(model, input_tensor, mask_tensor)

    seq_len = args.length + 2
    tokens = ["[CLS]"] + list(sequences[0]) + ["[SEP]"]
    summary = {}
    best_heads = []

    for layer_idx, layer_attention in enumerate(attentions):
        # layer_attention shape: batch, heads, query, key
        num_heads = layer_attention.shape[1]
        summary[str(layer_idx)] = {}
        for head_idx in range(num_heads):
            head_attention = layer_attention[:, head_idx, :seq_len, :seq_len].numpy()
            mean_matrix = head_attention.mean(axis=0)
            plot_attention(
                mean_matrix,
                tokens,
                f"Layer {layer_idx}, head {head_idx}",
                args.output_dir / f"layer_{layer_idx}_head_{head_idx}.png",
            )

            forward_scores = []
            backward_scores = []
            for ex_idx, sequence in enumerate(sequences):
                for opener_idx, closer_idx in matched_pairs(sequence):
                    opener_pos = opener_idx + 1
                    closer_pos = closer_idx + 1
                    forward_scores.append(float(head_attention[ex_idx, opener_pos, closer_pos]))
                    backward_scores.append(float(head_attention[ex_idx, closer_pos, opener_pos]))

            pair_mean = float(np.mean(forward_scores + backward_scores))
            pair_std = float(np.std(forward_scores + backward_scores))
            row = {
                "alpha_opener_to_closer_mean": float(np.mean(forward_scores)),
                "alpha_opener_to_closer_std": float(np.std(forward_scores)),
                "alpha_closer_to_opener_mean": float(np.mean(backward_scores)),
                "alpha_closer_to_opener_std": float(np.std(backward_scores)),
                "bidirectional_pair_mean": pair_mean,
                "bidirectional_pair_std": pair_std,
            }
            summary[str(layer_idx)][str(head_idx)] = row
            best_heads.append((pair_mean, layer_idx, head_idx))

    best_heads.sort(reverse=True)
    metrics = {
        "config": {
            "num_examples": args.num_examples,
            "length": args.length,
            "device": str(device),
        },
        "summary": summary,
        "top_heads_by_bidirectional_pair_attention": [
            {
                "layer": layer,
                "head": head,
                "bidirectional_pair_mean": score,
            }
            for score, layer, head in best_heads[:5]
        ],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
