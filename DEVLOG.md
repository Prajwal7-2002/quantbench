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

## 2026-10-26: Phase 3 — Drop-in Neural Layers (`QuantLinear`) & Model Conversion

- Created `QuantLinear` (`quantbench.nn.quant_linear`):
  - A drop-in replacement for `torch.nn.Linear` storing compressed integer weights inside persistent buffers (`self.register_buffer("packed_weight", ...)`).
  - Supports INT4 (groupwise and per-channel), INT8, BitNet Ternary, and 2:4 Structured Sparsity.
  - Implements dynamic vector dequantization in `forward(x)`.
  - Overrides `_load_from_state_dict` to dynamically bind serialized buffer tensors during checkpoint restoration.
  - Computes exact layer parameter memory bytes and compression factor relative to FP16.
- Created `convert_model_to_quant` (`quantbench.nn.model_convert`):
  - In-place recursive model converter traversing PyTorch module trees.
  - Swaps targeted linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) while cleanly preserving sensitive layers (`lm_head`).
  - Reports layer conversion counts, memory reduction statistics, and elapsed conversion duration.
## 2026-10-26: Phase 4 — Advanced Calibration Algorithms (SmoothQuant, AWQ, GPTQ)

- Created `quantbench.quant.algorithms`:
  - `smoothquant.py`: Implemented activation outlier channel profiling and per-channel migration scale computation ($Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$). Mathematical verification confirms exact continuous output invariance before discrete rounding.
  - `awq.py`: Implemented Activation-aware Weight Quantization with mean activation channel profiling and grid-search over scale candidate $\alpha \in [0, 1]$ to minimize Frobenius reconstruction error $\| W X - \hat{W} X \|_F^2$.
  - `gptq.py`: Implemented Accurate Post-Training Quantization for Generative Pre-trained Transformers using empirical activation Hessian ($H = 2 X X^T$), diagonal damping regularization ($H_{\text{damp}} = H + \lambda \cdot \text{mean}(\text{diag}(H)) \cdot I$), Cholesky decomposition inversion ($H^{-1} = L^{-T} L^{-1}$), Optimal Brain Surgeon (OBS) coordinate updates, and vectorized block GEMM updates ($W_{:, j+B:} \mathrel{-}= E_{\text{block}} \cdot [H^{-1}]_{j:j+B, j+B:}$).
- Added comprehensive unit tests in `tests/test_algorithms.py`:
  - Outlier profiling and continuous mathematical invariance testing.
  - AWQ reconstruction error reduction testing on synthetic salient channels.
  - Hessian symmetry, positive semi-definiteness, and GPTQ OBS error reduction testing.
- 39/39 unit tests passing with 93% test coverage across the repository. All Ruff checks passing.

## 2026-10-26: Phase 5 & 6 — Sliding-Window Perplexity, SQNR & Pareto Benchmarking

- Created `quantbench.eval`:
  - `sqnr.py`: Implemented Signal-to-Quantization-Noise Ratio ($10 \log_{10}(P_{\text{signal}} / P_{\text{noise}})$) in decibels (dB) for layer-level numerical fidelity tracking without forward inference overhead.
  - `perplexity.py`: Implemented statistically rigorous sliding-window causal language model evaluation engine. Strided contexts ($S < L$) eliminate context boundary collapse by evaluating loss strictly on trailing $S$ tokens.
- Created `quantbench.bench.pareto`:
  - `ParetoBenchmark`: Automated sweep engine mapping (Compression Ratio, Footprint MB, Delta PPL, SQNR dB) across FP16 baseline, INT8, INT4 (G=128, G=64), BitNet Ternary, and 2:4 Structured Sparsity.
  - Formats GitHub Markdown tables and ASCII terminal scatter plots.
- Added CLI commands `eval` and `pareto` in `quantbench.cli`.
- Added unit tests in `tests/test_eval_and_pareto.py`:
  - 47/47 unit tests passing across entire repo with 95% test coverage in 3.33s. All Ruff checks passing.

## 2026-10-26: Phase 7 — Public Learning Guide & Packaging Complete

- Authoring of `GUIDE.md`:
  - Comprehensive educational engineering guide detailing arithmetic intensity rooflines, zero-preserving affine invariant derivation, banker's rounding drift, OCP FP8 specifications, vectorized sub-byte packing runtimes, SmoothQuant, AWQ, GPTQ second-order Hessian compensation, persistent buffer architecture, and dual-metric Pareto evaluation.
- Validated PyPI distribution packaging with `uv build`:
  - Successfully produced source distribution (`.tar.gz`) and platform-independent wheel (`.whl`).
- Repository verified across 47 automated test suites with 95% line coverage. All Ruff checks passing.
- All 7 staged roadmap milestones in `README.md` completed.






## 2026-10-29: Windows PyTorch Wheel Index Compatibility

- Added [tool.uv.sources] and [[tool.uv.index]] for pytorch-cu124 on Windows (win32) and pytorch-cpu on Linux (linux) in pyproject.toml.
- Resolves Windows Application Control (WinError 4551) DLL blocking for local development.
- Verified 47/47 tests passing in 60s with 95% coverage on Python 3.12.10.
