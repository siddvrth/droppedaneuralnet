# droppedaneuralnet

## Problem Statement
We are given 97 disassembled linear-layer state dicts and historical input/output data.  
Architecture is fixed:

- `Block(48, 96)` repeated 48 times, where each block is
  - `Linear(48 -> 96)`
  - `ReLU`
  - `Linear(96 -> 48)`
  - residual add
- one `LastLayer(48 -> 1)` at the end

Goal: recover a permutation `P` of `0..96` where `P[i]` is the piece index at model position `i`.

Puzzle: https://huggingface.co/spaces/jane-street/droppedaneuralnet

## Context
I came across this puzzle from Dwarkesh Patel's interview with Elon Musk (timestamped): https://youtu.be/BYXbuik3dgA?si=WqPITMxrP-l75Y86&t=4394

Brute force over `48! × 48!` (`≈ 1.54 × 10^122`) is impossible, and direct greedy/simulated-annealing search over full permutations can get trapped in local minima. 

This solution was developed with assistance from Codex 5.3.

## Reconstruction Strategy
Implementation: `solve.py`

### 1. Piece typing by shape
Each shard contains a `weight` and `bias`.

- 48 pieces with weight shape `(96, 48)` are candidate `Block.inp`.
- 48 pieces with weight shape `(48, 96)` are candidate `Block.out`.
- 1 piece with shape `(1, 48)` is the unique final `LastLayer`.

This enforces the architecture count exactly (48 blocks + 1 last layer).

### 2. Pairing `Block.inp` with `Block.out`
For each `(96,48)` piece `W_in` and `(48,96)` piece `W_out`, define cost:

`C(i,j) = || W_in W_in^T - W_out^T W_out ||_F`

We solve one-to-one matching with Hungarian assignment (minimum total cost).

Reasoning: in a true block, input/output linears share hidden-space geometry, so Gram matrices are structurally compatible.

### 3. Initial ordering heuristic
With pairs fixed, each block induces residual update:

`delta(x) = Linear_out(ReLU(Linear_in(x)))`

On historical inputs, compute per-block scale score:

`score = mean(std(delta(x), dim=0))`

Sort blocks ascending by this score to initialize order.

### 4. Monotonic local refinement
Objective:

`L(order) = MSE(model(order, x_hist), pred_hist)`

Apply repeated adjacent swaps; keep a swap only if it strictly decreases `L`.  
Stop when one full pass has no improving swap.

This is a deterministic local descent from a strong initialization.

### 5. Assemble full permutation
For each ordered block, append:

1. its matched `(96,48)` piece index
2. its matched `(48,96)` piece index

Then append the final `(1,48)` piece.

### Pairing diagnostics
- Hungarian vs per-row nearest neighbor agreement: `41 / 48`.
- Assignment margin ratio (`second_best / best`) stats:
  - mean `1.01897`
  - min `1.00025`
  - max `1.10225`

### Ordering diagnostics
MSE to `pred` column (`lower is better`):

- Random order baseline: `0.81279`
- Index order baseline: `0.76597`
- Delta-scale init: `0.01826`
- Final refined order: `7.77e-15`

Max absolute prediction error at final order: `4.77e-07`

### Consistency with `true`
Using reconstructed model:

- `MSE(model_output, true) = 0.10648094`
- `MSE(pred, true)` from CSV = `0.10648096`

These match up to floating-point tolerance, indicating the reconstructed model reproduces the provided `pred`.

## Final Recovered Permutation
`[43, 34, 65, 22, 69, 89, 28, 12, 27, 76, 81, 8, 5, 21, 62, 79, 64, 70, 94, 96, 4, 17, 48, 9, 23, 46, 14, 33, 95, 26, 50, 66, 1, 40, 15, 67, 41, 92, 16, 83, 77, 32, 10, 20, 3, 53, 45, 19, 87, 71, 88, 54, 39, 38, 18, 25, 56, 30, 91, 29, 44, 82, 35, 24, 61, 80, 86, 57, 31, 36, 13, 7, 59, 52, 68, 47, 84, 63, 74, 90, 0, 75, 73, 11, 37, 6, 58, 78, 42, 55, 49, 72, 2, 51, 60, 93, 85]`

## Run 
```bash
.venv/bin/python solve.py --output-json solution.json
```
