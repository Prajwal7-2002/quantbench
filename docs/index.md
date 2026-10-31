# quantbench ⚖️

[![PyPI](https://img.shields.io/pypi/v/quantbench-engine.svg)](https://pypi.org/project/quantbench-engine/)
[![CI](https://github.com/Prajwal7-2002/quantbench/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/quantbench/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU_%26_GPU-teal.svg)](#)

A **unified, first-principles quantization engine, bit-packing runtime, 2:4 sparsity toolkit, and Quality-vs-Speed Pareto evaluation benchmark** in pure Python and PyTorch.

---

## Why quantbench?

Most developers treat LLM quantization as a black box: they download GGUF or AWQ checkpoints without knowing how scales and zero-points are calculated, how 4-bit signed integers are packed into memory bytes, or why activation outliers in attention layers cause naive INT8 quantization to collapse.

`quantbench` implements these algorithms from scratch with **radical transparency and zero compiled C++/CUDA dependencies**, and provides a benchmark CLI that measures empirical perplexity degradation alongside generation throughput.

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

## 📦 Quick Installation

Install directly from [PyPI](https://pypi.org/project/quantbench-engine/):

```bash
pip install quantbench-engine

# Or via uv
uv add quantbench-engine
```

---

## 🚀 Navigation

* **[Mathematical Foundations](math-foundations.md)** — Zero-preserving affine invariants, symmetric AbsMax, blockwise grouping, and OCP FP8 emulation.
* **[Sub-Byte Bit-Packing](bit-packing.md)** — INT4 nibble packing, two's complement sign extension, and BitNet 1.58b ternary vectorization.
* **[2:4 Structured Sparsity](sparsity-24.md)** — NVIDIA Ampere/Ada/Hopper sparse tensor core layout and metadata encoding.
* **[Advanced Calibration](calibration.md)** — SmoothQuant channel migration, AWQ Frobenius grid search, and GPTQ second-order Hessian updates.
* **[Drop-in QuantLinear Layers](quant-linear.md)** — Replacing PyTorch `nn.Linear` with compressed sub-byte persistent buffers.
* **[Pareto Evaluation](pareto-eval.md)** — Layer SQNR (dB), sliding-window perplexity on WikiText-2/C4, and terminal Pareto curves.
