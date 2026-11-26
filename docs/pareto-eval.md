# Quality vs Speed Pareto Evaluation

Quantization always involves a trade-off between **model quality (perplexity / SQNR)** and **memory footprint and generation speed**.

---

## 1. Evaluation Metrics

### Signal-to-Quantization-Noise Ratio (SQNR)
Measures reconstruction fidelity across layers in decibels (dB):

$$\text{SQNR (dB)} = 10 \cdot \log_{10} \left( \frac{\sum |W|^2}{\sum |W - \hat{W}|^2 + \epsilon} \right)$$

SQNR is cheap to compute, but it is only a weight-level signal. On Qwen2.5-0.5B, round-to-nearest INT4
weights reached about 18–20 dB SQNR yet raised perplexity by 18–30%, and every projection type had almost the
same SQNR while their perplexity impact differed tenfold. Always confirm with perplexity on real text.

### Sliding-Window Perplexity
`compute_sliding_window_perplexity` evaluates perplexity with a sliding window stride $S < L$ so that every
token has preceding causal context. The CLI demo runs it on a synthetic corpus; the real-model script below
uses fixed 1,024-token windows of license-free text.

---

## 2. Running Pareto Sweeps

**Quick demo (toy model, CPU):** runs every scheme on a small randomly initialised Transformer and prints a
terminal table. Useful to see the pipeline work; the numbers say nothing about real models.

```bash
quantbench pareto --vocab-size 128 --hidden-dim 128 --seq-len 300
```

**Real model (CUDA GPU):** measures weight SQNR, stored size, resting VRAM, perplexity and decode speed for
BF16, INT8, INT4 (G=32/64/128), 2:4 sparsity and post-hoc ternary, plus a per-layer sensitivity sweep:

```bash
pip install quantbench-engine transformers accelerate
python scripts/measure_real_pareto.py --model Qwen/Qwen2.5-0.5B-Instruct
```

It writes `hf_dataset/` (JSON, CSV, run metadata and a dataset card). The published results are at
[prajwalpujari16/quantbench-pareto](https://huggingface.co/datasets/prajwalpujari16/quantbench-pareto).

### Measured on Qwen2.5-0.5B-Instruct (RTX 3050 Laptop GPU)

| Config | Linear-layer compression | Model VRAM | Weight SQNR | Perplexity change | Decode vs BF16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 1.00x | 950 MB | n/a | 0% | 1.00x |
| INT8 per-channel | 1.99x | 611 MB | 40.0 dB | +0.36% | 0.68x |
| INT4 G=32 | 3.20x | 483 MB | 19.8 dB | +18.4% | 0.23x |
| INT4 G=64 | 3.55x | 464 MB | 18.7 dB | +23.6% | 0.23x |
| INT4 G=128 | 3.76x | 451 MB | 17.8 dB | +30.1% | 0.23x |
| 2:4 structured sparsity | 0.94x | 1001 MB | 9.1 dB | model collapses | 0.34x |
| Ternary (post-hoc) | 8.00x | 364 MB | 5.0 dB | model collapses | 0.24x |

What it shows:
* INT8 is close to lossless; INT4 trades perplexity for size, and smaller groups cost less quality.
* Decode is slower for every scheme, because the reference runtime dequantizes full weights on each
  forward pass. Speed gains need fused dequant-matmul kernels.
* 2:4 sparsity stores values in the source dtype plus a mask, so it saves no memory here.
* Ternary applied after training destroys the model; BitNet-style models must be trained ternary.
