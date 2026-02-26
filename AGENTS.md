# Project: Neural Network Reconstruction Puzzle

## Objective

We are given disassembled neural network layers (97 total pieces).
The goal is to reconstruct the original model by determining the correct
permutation order of these pieces.

The final answer must be:
A permutation P such that for each index 0..96 (inclusive),
P[i] gives the index of the piece applied at position i.

This is NOT a task to redesign the architecture.
It is a reconstruction task.

---

## Architecture Constraints

The only valid components are:

- Block(in_dim, hidden_dim)
    - Linear(in_dim → hidden_dim)
    - ReLU
    - Linear(hidden_dim → in_dim)
    - Residual connection

- LastLayer(in_dim, out_dim)
    - Single Linear(in_dim → out_dim)

These class definitions MUST NOT be modified.

---

## Important Properties

- Blocks preserve dimensionality due to residual connections.
- Therefore Blocks can be composed sequentially if dimensions match.
- The LastLayer must appear at the end of the network.
- Only one LastLayer should exist.

---

## Strategy Requirements

The agent should:

1. Load all pieces.
2. Treat reconstruction as a permutation search problem.
3. Use historical data to score candidate orderings.
4. Use validation loss to determine correctness.
5. Avoid brute-force over 97! possibilities.
6. Implement heuristics or optimization strategies:
   - greedy search
   - beam search
   - simulated annealing
   - gradient-based ordering relaxation (if feasible)

---

## Prohibited Actions

- Do NOT rewrite the provided layer classes.
- Do NOT change dimensions.
- Do NOT fabricate new layers.
- Do NOT hardcode the permutation.

---

## Deliverable

Output a permutation list of length 97:

Example format:
[12, 4, 88, 0, ..., 31]

Where index i is the position in the network,
and value is the index of the piece used there.
