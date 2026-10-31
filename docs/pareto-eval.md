# Quality vs Speed Pareto Evaluation

Quantization always involves a trade-off between **model quality (perplexity / SQNR)** and **generation throughput (memory compression / latency)**.

---

## 1. Evaluation Metrics

### Signal-to-Quantization-Noise Ratio (SQNR)
Measures reconstruction fidelity across layers in decibels (dB):

$$\text{SQNR (dB)} = 10 \cdot \log_{10} \left( \frac{\sum |W|^2}{\sum |W - \hat{W}|^2 + \epsilon} \right)$$

* $> 30\text{ dB}$: High fidelity (lossless or near-lossless generation).
* $20\text{--}30\text{ dB}$: Good fidelity (acceptable for production serving).
* $< 15\text{ dB}$: Noticeable degradation; requires calibration tuning (AWQ / GPTQ).

### Sliding-Window Perplexity
Evaluates perplexity on WikiText-2 or C4 validation splits using sliding window stride $S < L$ to ensure every token has preceding causal context.

---

## 2. Running Pareto Sweeps

Run the benchmark CLI to evaluate compression vs accuracy and render a terminal Pareto frontier:

```bash
quantbench pareto --formats fp16,int8,int4,ternary,sparse24
```

### Visual ASCII Pareto Curve:
```text
  Quality (SQNR dB)
     ▲
  50 ┤   [FP16 Baseline]
  45 ┤            [FP8 E4M3]
  40 ┤                   [INT8 AbsMax]
  35 ┤                         [Ampere 2:4]
  30 ┤                                [AWQ INT4]
  25 ┤                                      [INT4 Groupwise]
  20 ┤
  15 ┤                                              [BitNet Ternary]
     └──────────────────────────────────────────────────────────────►
     1.0x        2.0x        3.5x        4.0x        7.5x   Compression
```
