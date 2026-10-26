# Contributing to quantbench ⚖️

Thank you for your interest in contributing to `quantbench`!

`quantbench` is an open-source, first-principles quantization engine, bit-packing runtime, 2:4 sparsity toolkit, and Quality-vs-Speed Pareto evaluation benchmark written in pure Python and PyTorch. We welcome contributions that improve mathematical transparency, expand sub-byte compression layouts, add new post-training calibration algorithms, and bridge inference serving systems with precision engineering.

---

## 🛠️ Development Setup

We use [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Prajwal7-2002/quantbench.git
   cd quantbench
   ```

2. **Sync dependencies and virtual environment**:
   ```bash
   uv sync
   ```
   This automatically installs all runtime, benchmark, and development dependencies into a isolated `.venv`.

---

## 🧪 Testing Protocol

We maintain high test coverage across mathematical primitives, sub-byte bit manipulation, neural network layers, and calibration algorithms.

1. **Run the entire test suite**:
   ```bash
   uv run pytest
   ```

2. **Run with coverage report**:
   ```bash
   uv run pytest --cov=quantbench --cov-report=term-missing
   ```

3. **Run a specific test category**:
   ```bash
   # Test bit-packing roundtrip invariants
   uv run pytest tests/test_packing.py

   # Test calibration algorithms (SmoothQuant, AWQ, GPTQ)
   uv run pytest tests/test_algorithms.py

   # Test drop-in QuantLinear layers
   uv run pytest tests/test_quant_linear.py
   ```

### Invariant Testing Rules
When contributing new quantization schemes or packing runtimes, you **must** include tests verifying:
- **Roundtrip Fidelity**: $x = \text{unpack}(\text{pack}(x))$ for all valid integer representations.
- **Mathematical Invariance**: Continuous exactness proofs (e.g. SmoothQuant $(X s^{-1})(s W) = XW$ within $10^{-5}$ drift).
- **Error Bounds**: Maximum absolute error bounded by $\frac{\Delta}{2}$ for uniform quantizers.
- **Extreme Inputs**: Subnormal values, exact zeros, all-positive, and all-negative tensor ranges.

---

## 🎨 Code Quality & Style

We enforce modern Python conventions, strict type hints, and clean code formatting via [ruff](https://docs.astral.sh/ruff/):

1. **Lint and format checks**:
   ```bash
   uv run ruff check .
   ```

2. **Auto-fix formatting**:
   ```bash
   uv run ruff format .
   ```

### Architectural Principles

1. **Zero Compiled C++/CUDA Dependencies**:
   `quantbench` is designed for radical algorithmic transparency. Core quantization logic and bit-packing must run in pure Python and PyTorch without requiring NVCC or compiler toolchains.
2. **Vectorized PyTorch Operations**:
   Avoid Python `for` loops across tensor dimensions. Sub-byte bit-packing must be implemented via vectorized tensor operations (`&`, `|`, `>>`, `<<`) on `torch.uint8` containers.
3. **Hardware Agnostic**:
   All algorithms must run seamlessly on consumer CPUs, Apple Silicon, and CUDA GPUs without hardcoded device assumptions.
4. **Honest Pareto Benchmarks**:
   Never cherry-pick perplexity numbers. Always evaluate against standard validation corpora (WikiText-2, C4) using standardized sliding-window stride lengths.

---

## 🎯 High-Impact Areas for Contribution

We actively welcome contributions in the following domains:

### 1. New Sub-Byte Formats & Precision Schemes (`src/quantbench/quant/`)
- [ ] **NVIDIA Blackwell NVFP4 Emulation**: Bit-exact emulation of 4-bit floating point formats (E2M1) with micro-scaling factors.
- [ ] **2-bit Non-Linear Quantization**: Quantization via Lloyd-Max optimal non-uniform codebooks.
- [ ] **NF4 (NormalFloat4)**: Quantile-based optimal representation for zero-mean normally distributed weights.

### 2. Quantized KV-Cache Runtimes
- [ ] **INT4 / FP8 KV-Cache Page Allocator**: Interfacing with `mini-infer`'s paged memory manager to benchmark memory savings during multi-turn generation.
- [ ] **Dynamic Per-Token Key/Value Quantization**: Asymmetric scaling per token vector along the sequence dimension.

### 3. Advanced Calibration & Optimization (`src/quantbench/quant/algorithms/`)
- [ ] **SqueezeLLM Clustering**: Non-uniform sensitivity-weighted k-means clustering with sparse outlier matrices.
- [ ] **QuIP# Vector Quantization**: Randomized Hadamard transform pre-processing to eliminate activation outliers.
- [ ] **HQQ (Half-Quadratic Quantization)**: Closed-form optimization without calibration data.

### 4. Real-World Model Support (`src/quantbench/nn/`)
- [ ] **GGUF Dequantization Loader**: Direct parsing and loading of `.gguf` quantized tensors into `QuantLinear` modules.
- [ ] **SafeTensors Model Converter**: Auto-convert and shard Hugging Face SafeTensors checkpoints directly into disk-mapped quantized weights.

---

## 📬 Pull Request Rules & Submission Process

To keep `quantbench` reliable and maintainable, all contributions must follow these conventions:

### 1. Branch Naming Conventions
Create a descriptive branch off `main`:
* `feat/short-feature-name` (e.g. `feat/nvfp4-packing`)
* `fix/short-bug-description` (e.g. `fix/ternary-base3-rounding`)
* `perf/short-perf-description` (e.g. `perf/vectorized-hessian-chol`)
* `docs/short-doc-description` (e.g. `docs/smoothquant-proof-derivation`)
* `test/short-test-description` (e.g. `test/gptq-corner-cases`)

### 2. Conventional Commit Standards
Commits must follow the Conventional Commits specification:
```
feat(quant): add NVIDIA NVFP4 analytical emulator and bit-packing runtime
fix(packing): correct uint8 mask overflow in sparse24 index condensation
docs(guide): add mathematical derivation of optimal brain surgeon Hessian update
test(eval): add sliding window perplexity regression test on small corpus
```

### 3. PR Verification Checklist
Before submitting a PR, ensure:
- [ ] `uv run pytest` passes 100% of tests.
- [ ] `uv run ruff check .` reports 0 errors.
- [ ] Test coverage does not decrease.
- [ ] Relevant documentation or examples are updated.
- [ ] Single author attribution is preserved.

---

## 📜 License

By contributing to `quantbench`, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
