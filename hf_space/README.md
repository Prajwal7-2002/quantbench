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
  - meta-llama/Llama-3.1-8B
---

# QuantBench 🔬 LLM Sub-Byte Quantization Studio

Interactive pre-flight analytical quantization exploration, BitNet 1.58b ternary bit-packing registers, and layer sensitivity simulator.

* **Interactive Features:** SQNR-vs-compression Pareto frontier explorer, live BitNet 1.58b / INT4 / 2:4 sparsity bit-packing registers, and Transformer layer sensitivity simulator.
* **GitHub Repository:** [Prajwal7-2002/quantbench](https://github.com/Prajwal7-2002/quantbench)
* **Dataset:** [prajwalpujari16/quantbench-pareto](https://huggingface.co/datasets/prajwalpujari16/quantbench-pareto)
* **PyPI:** [`pip install quantbench`](https://pypi.org/project/quantbench/)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))

---

### Citation & Attribution

If you use QuantBench or its empirical quantization Pareto dataset in your research, compression pipelines, or publications, please cite:

```bibtex
@misc{pujari2024quantbench,
  author = {Pujari, Prajwal},
  title = {QuantBench: Sub-Byte LLM Quantization and Quality-Compression Pareto Benchmark},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/prajwalpujari16/quantbench}},
  url = {https://github.com/Prajwal7-2002/quantbench}
}
```

