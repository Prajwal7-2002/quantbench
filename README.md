# quantbench ⚖️

[![CI](https://github.com/Prajwal7-2002/quantbench/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/quantbench/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU_%26_GPU-teal.svg)](#)

A **unified, first-principles quantization engine, bit-packing runtime, 2:4 sparsity toolkit, and Quality-vs-Speed Pareto evaluation benchmark** in pure Python and PyTorch.

Most developers treat LLM quantization as a black box: they download GGUF or AWQ checkpoints without knowing how scales and zero-points are calculated, how 4-bit signed integers are packed into memory bytes, or why activation outliers in attention layers make naive INT8 quantization collapse.

`quantbench` implements these production algorithms from scratch with **radical transparency and zero compiled C++/CUDA dependencies**, and provides a benchmark CLI that measures empirical perplexity degradation alongside generation throughput.

---

## ⚡ Core Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  quantbench                                            │
│            (Single Unified Repository: First-Principles Engine + Benchmark)            │
│                                                                                        │
│  ┌────────────────────────────────────────┐  ┌───────────────────────────────────────┐ │
│  │ 1. From-Scratch Quant Engine (quant)   │  │ 2. Drop-in Neural Layers (nn)         │ │
│  │ • Uniform Affine (Scale & Zero-Point)  │  │ • QuantLinear (PyTorch Module)        │ │
│  │ • Symmetric AbsMax (Per-Tensor/Channel)│  │ • In-Place Model Converter (Qwen2.5,  │ │
│  │ • Blockwise (Group Size 32, 64, 128)   │  │   SmolLM2-135M, GPT-2)                │ │
│  │ • INT4 Nibble Packing (uint8 vectors)  │  │ • Forward GEMM with dynamic dequant   │ │
│  │ • Sub-2-bit Ternary Packing (BitNet)   │  │ • Layer-by-Layer SQNR Diagnostics     │ │
│  │ • 2:4 Structured Sparsity (Ampere)     │  │                                       │ │
│  │ • SmoothQuant Outlier Migration        │  │                                       │ │
│  │ • AWQ Grid Search & GPTQ Hessian/Chol  │  │                                       │ │
│  └───────────────────┬────────────────────┘  └───────────────────┬───────────────────┘ │
│                      │                                           │                     │
│                      └─────────────────────┬─────────────────────┘                     │
│                                            │ Powers                                    │
│                                            ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │ 3. Rigorous Evaluation & Profiling (eval & bench)                                 │ │
│  │ • Perplexity Engine: Sliding-window evaluation on WikiText-2 & C4                 │ │
│  │ • Diagnostics: SQNR (dB), Max Error, Outlier Dimension Detection                  │ │
│  │ • Hardware Profiler: servebench bridge (Tokens/sec, TTFT, Peak VRAM in MB)        │ │
│  │ • Terminal ASCII Pareto Visualizer: Generates live Quality vs Speed tradeoff curve│ │
│  └───────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Foundations

### 1. Affine Asymmetric Quantization
Quantizes a floating-point tensor $x \in [x_{\min}, x_{\max}]$ into integer range $[q_{\min}, q_{\max}]$:
$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad z = \text{round}\left(-\frac{x_{\min}}{s}\right) + q_{\min}$$
$$q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z, \; q_{\min}, \; q_{\max}\right)$$
$$\hat{x} = (q - z) \cdot s$$

### 2. Symmetric AbsMax Quantization
Centers dynamic range at zero ($z = 0$), enabling direct hardware integer GEMM without per-element zero-point subtractions:
$$s = \frac{\max(|x|)}{2^{b-1} - 1}, \quad q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), \; -2^{b-1}, \; 2^{b-1}-1\right)$$
$$\hat{x} = q \cdot s$$

### 3. Block-Wise (Group-Wise) Quantization
Partitions weights into blocks of size $G \in \{32, 64, 128\}$ to isolate channel outliers and preserve precision across wide layers.

### 4. OCP FP8 Emulation
Bit-exact simulation of **E4M3FN** (higher mantissa precision, range $\pm 448$, no infinity) for GEMM weights/activations and **E5M2** (wider dynamic range $\pm 57344$) for gradients and KV caches.

---

## 🚀 Quickstart

Dependencies are managed via [uv](https://docs.astral.sh/uv/):

```bash
# Clone and sync environment
git clone https://github.com/Prajwal7-2002/quantbench.git
cd quantbench
uv sync

# Run the complete test suite
uv run pytest
```

---

## 🗺️ Roadmap & Staged Milestones

- [x] **Phase 1: Mathematical Primitives & Invariants**: Affine, AbsMax, Group-wise, and FP8 (E4M3/E5M2) simulation with rigorous error bound verification.
- [ ] **Phase 2: Integer Bit-Packing Runtimes**: Vectorized INT4 nibble packing (`uint8`), BitNet ternary packing, and 2:4 sparsity metadata.
- [ ] **Phase 3: Advanced Optimization Algorithms**: SmoothQuant outlier migration, AWQ grid search, and GPTQ second-order Hessian updates.
- [ ] **Phase 4: Drop-in Neural Layers**: `QuantLinear` PyTorch module, in-place Hugging Face model converter, and layer SQNR diagnostics.
- [ ] **Phase 5: Perplexity & Evaluation Engine**: Standardized sliding-window perplexity on WikiText-2 and C4.
- [ ] **Phase 6: servebench Bridge & Pareto Visualizer**: Empirical throughput (tok/s) profiling and terminal ASCII Pareto curves.
- [ ] **Phase 7: Public Learning Guide & Packaging**: Comprehensive educational walkthrough and PyPI release.

---

## 📜 License

Apache 2.0. See [LICENSE](LICENSE) for details.
