from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class PieceGroup:
    piece_ids: list[int]
    weights: list[torch.Tensor]
    biases: list[torch.Tensor]


@dataclass
class LoadedPieces:
    linear_96x48: PieceGroup
    linear_48x96: PieceGroup
    last_piece_id: int
    last_weight: torch.Tensor
    last_bias: torch.Tensor


def load_pieces(pieces_dir: Path) -> LoadedPieces:
    a_ids: list[int] = []
    a_w: list[torch.Tensor] = []
    a_b: list[torch.Tensor] = []
    b_ids: list[int] = []
    b_w: list[torch.Tensor] = []
    b_b: list[torch.Tensor] = []
    last: tuple[int, torch.Tensor, torch.Tensor] | None = None

    for path in sorted(pieces_dir.glob("piece_*.pth"), key=lambda p: int(p.stem.split("_")[1])):
        piece_id = int(path.stem.split("_")[1])
        state = torch.load(path, map_location="cpu")
        weight = state["weight"].float()
        bias = state["bias"].float()
        shape = tuple(weight.shape)

        if shape == (96, 48):
            a_ids.append(piece_id)
            a_w.append(weight)
            a_b.append(bias)
        elif shape == (48, 96):
            b_ids.append(piece_id)
            b_w.append(weight)
            b_b.append(bias)
        elif shape == (1, 48):
            if last is not None:
                raise ValueError("Found multiple last-layer candidates.")
            last = (piece_id, weight, bias)
        else:
            raise ValueError(f"Unexpected shape {shape} in {path}")

    if len(a_ids) != 48 or len(b_ids) != 48 or last is None:
        raise ValueError(
            f"Expected 48 pieces of each block linear layer and 1 last layer, got "
            f"{len(a_ids)}, {len(b_ids)}, {0 if last is None else 1}."
        )

    return LoadedPieces(
        linear_96x48=PieceGroup(a_ids, a_w, a_b),
        linear_48x96=PieceGroup(b_ids, b_w, b_b),
        last_piece_id=last[0],
        last_weight=last[1],
        last_bias=last[2],
    )


def pair_block_linears(pieces: LoadedPieces) -> np.ndarray:
    """Pair each 96x48 piece to a 48x96 piece using Hungarian assignment."""
    a = pieces.linear_96x48.weights
    b = pieces.linear_48x96.weights
    n = len(a)
    cost = np.zeros((n, n), dtype=np.float64)

    # For a true Block, hidden-space geometry between inp/out linears is coherent.
    # We match by minimizing ||W_in W_in^T - W_out^T W_out||_F.
    for i in range(n):
        gram_in = a[i] @ a[i].t()
        for j in range(n):
            gram_out = b[j].t() @ b[j]
            cost[i, j] = torch.norm(gram_in - gram_out).item()

    row_idx, col_idx = linear_sum_assignment(cost)
    pair_for_a = np.zeros(n, dtype=np.int64)
    for r, c in zip(row_idx, col_idx):
        pair_for_a[r] = c
    return pair_for_a


@torch.no_grad()
def forward_mse_to_target(
    x: torch.Tensor,
    target: torch.Tensor,
    pieces: LoadedPieces,
    pair_for_a: np.ndarray,
    order_a: list[int],
) -> float:
    a_w = pieces.linear_96x48.weights
    a_b = pieces.linear_96x48.biases
    b_w = pieces.linear_48x96.weights
    b_b = pieces.linear_48x96.biases
    last_w = pieces.last_weight
    last_b = pieces.last_bias

    h = x
    for ai in order_a:
        bi = int(pair_for_a[ai])
        h = h + (torch.relu(h @ a_w[ai].t() + a_b[ai]) @ b_w[bi].t() + b_b[bi])
    pred = h @ last_w.t() + last_b
    return torch.mean((pred - target) ** 2).item()


@torch.no_grad()
def initial_order_by_delta_scale(
    x: torch.Tensor,
    pieces: LoadedPieces,
    pair_for_a: np.ndarray,
) -> list[int]:
    scores: list[tuple[int, float]] = []
    a_w = pieces.linear_96x48.weights
    a_b = pieces.linear_96x48.biases
    b_w = pieces.linear_48x96.weights
    b_b = pieces.linear_48x96.biases

    # Empirically this network behaves like progressively larger residual updates.
    # We sort blocks by update scale on historical inputs.
    for ai in range(len(a_w)):
        bi = int(pair_for_a[ai])
        delta = torch.relu(x @ a_w[ai].t() + a_b[ai]) @ b_w[bi].t() + b_b[bi]
        score = delta.std(dim=0).mean().item()
        scores.append((ai, score))

    return [ai for ai, _ in sorted(scores, key=lambda t: t[1])]


def refine_by_adjacent_swaps(
    x: torch.Tensor,
    target: torch.Tensor,
    pieces: LoadedPieces,
    pair_for_a: np.ndarray,
    order_a: list[int],
) -> tuple[list[int], float]:
    """Monotonic local search with adjacent swaps."""
    current = order_a[:]
    current_loss = forward_mse_to_target(x, target, pieces, pair_for_a, current)

    while True:
        improved = False
        for i in range(len(current) - 1):
            candidate = current[:]
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
            cand_loss = forward_mse_to_target(x, target, pieces, pair_for_a, candidate)
            if cand_loss + 1e-12 < current_loss:
                current = candidate
                current_loss = cand_loss
                improved = True
        if not improved:
            break
    return current, current_loss


def build_permutation(
    pieces: LoadedPieces,
    pair_for_a: np.ndarray,
    order_a: list[int],
) -> list[int]:
    perm: list[int] = []
    a_ids = pieces.linear_96x48.piece_ids
    b_ids = pieces.linear_48x96.piece_ids
    for ai in order_a:
        bi = int(pair_for_a[ai])
        perm.append(a_ids[ai])
        perm.append(b_ids[bi])
    perm.append(pieces.last_piece_id)
    return perm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct dropped model permutation.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("historical_data_and_pieces"),
        help="Directory containing historical_data.csv and pieces/",
    )
    parser.add_argument(
        "--target-column",
        choices=["pred", "true"],
        default="pred",
        help="Column to match during reconstruction.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output file for permutation and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    csv_path = data_dir / "historical_data.csv"
    pieces_dir = data_dir / "pieces"

    df = pd.read_csv(csv_path)
    x_np = np.array(df.iloc[:, :48].to_numpy(dtype=np.float32), copy=True)
    target_np = np.array(df[[args.target_column]].to_numpy(dtype=np.float32), copy=True)
    x = torch.from_numpy(x_np)
    target = torch.from_numpy(target_np)

    pieces = load_pieces(pieces_dir)
    pair_for_a = pair_block_linears(pieces)
    order_a = initial_order_by_delta_scale(x, pieces, pair_for_a)
    order_a, final_mse = refine_by_adjacent_swaps(x, target, pieces, pair_for_a, order_a)
    permutation = build_permutation(pieces, pair_for_a, order_a)

    if len(permutation) != 97 or set(permutation) != set(range(97)):
        raise ValueError("Output is not a valid permutation of 0..96.")

    result = {
        "target_column": args.target_column,
        "mse": final_mse,
        "permutation": permutation,
    }

    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
