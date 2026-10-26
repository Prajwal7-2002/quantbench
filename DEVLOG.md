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
