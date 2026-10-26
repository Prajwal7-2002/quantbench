# DEVLOG: quantbench

## 2026-10-26: Project Inception & Phase 1 Mathematical Primitives

- Initialized Project #5 `quantbench` (Unified Quantization Engine & Quality-vs-Speed Pareto Benchmark).
- Architecture decision: Unified from-scratch quantization algorithms (`quantbench.quant`), drop-in PyTorch modules (`quantbench.nn`), and sliding-window Pareto evaluation (`quantbench.eval`, `quantbench.bench`) in a single zero-dependency-friction repository.
- Initialized Apache 2.0 license, `.gitignore`, and `pyproject.toml` with `hatchling` build backend and `uv` package manager.
- Implemented Phase 1 Mathematical Primitives:
  - Affine asymmetric quantization ($s, z$) and symmetric AbsMax quantization ($s, z=0$) with dynamic per-tensor and per-channel scaling along arbitrary dimensions.
  - Provable clamp invariants verifying $|x - \hat{x}| \le \frac{s}{2}$.
  - Group-wise / block-wise quantization for group sizes $G \in \{32, 64, 128\}$ with arbitrary tensor padding and dequantization reconstruction.
  - Bit-level simulation of OCP FP8 E4M3FN (range $\pm 448$, 3 mantissa bits) and E5M2 (range $\pm 57344$, 2 mantissa bits).
  - 100% test coverage across mathematical invariants and error bounds.

## 2026-10-26: Phase 2 — Integer Bit-Packing Runtimes (Sub-Byte Storage)

- Implemented vectorized sub-byte bit-packing runtimes in `quantbench.quant.packing`:
  - `int4_pack.py`: Vectorized bit-shift packing of two INT4 values (unsigned $[0, 15]$ or signed two's complement $[-8, 7]$) into a single `uint8` byte container: `byte = (v0 & 0x0F) | ((v1 & 0x0F) << 4)`. Full support for odd dimension length padding and exact shape restoration.
  - `ternary_pack.py`: BitNet 1.58b packing supporting both hardware-friendly 2-bit SIMD packing (4 weights/byte using biased offset $u = v + 1 \in \{0, 1, 2\}$) and theoretical maximal density Radix-3 base-3 packing (5 weights/byte via $3^5 = 243 < 256$).
  - `sparse_pack.py`: NVIDIA Ampere/Ada/Hopper 2:4 structured sparsity layout. Automated row pruning, value condensation to $50\%$ storage ($M, N/2$), and 2-bit metadata column index packing into 4-bit nibbles ($M, N/8$). Decompression scatter preserves exact zeros and non-zero positions.
- Added CLI command `pack-inspect` to visualize sub-byte bit layouts and theoretical memory compression directly from the terminal.
- 22/22 unit tests passing with 93% test coverage in 2.18s. All Ruff checks passing.

