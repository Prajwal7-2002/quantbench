---
title: QuantBench - Sub-Byte LLM Quantization Studio
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Sub-byte quantization Pareto frontier & BitNet packer.
tags:
  - quantization
  - bitnet
  - int4
  - sparsity
  - sqnr
  - pareto-frontier
  - llm-compression
  - transformer
datasets:
  - prajwalpujari16/quantbench-pareto
models:
  - Qwen/Qwen2.5-0.5B-Instruct
---

# QuantBench 🔬 LLM Sub-Byte Quantization & Pareto Frontier Studio

Measured quality-vs-compression results for INT8, INT4, 2:4 sparsity and post-hoc ternary quantization
(one model, one GPU, one run; see the dataset card for the setup and caveats), plus interactive
visualizers for ternary, INT4 and 2:4 bit-packing and a live SQNR calculator for your own tensors.

* **GitHub Repository:** [Prajwal7-2002/quantbench](https://github.com/Prajwal7-2002/quantbench)
* **PyPI:** [`pip install quantbench-engine`](https://pypi.org/project/quantbench-engine/)
* **HF Benchmark Dataset:** [prajwalpujari16/quantbench-pareto](https://huggingface.co/datasets/prajwalpujari16/quantbench-pareto)
* **Documentation:** [prajwal7-2002.github.io/quantbench](https://prajwal7-2002.github.io/quantbench/)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))

---

### Citation & Attribution

If you use QuantBench or its measured quantization Pareto dataset, please cite:

```bibtex
@misc{pujari2026quantbench,
  author = {Pujari, Prajwal},
  title = {QuantBench: Sub-Byte LLM Quantization and Quality-Compression Pareto Benchmark},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/prajwalpujari16/quantbench}},
  url = {https://github.com/Prajwal7-2002/quantbench}
}
```
