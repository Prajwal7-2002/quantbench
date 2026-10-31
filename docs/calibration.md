# Advanced Calibration Algorithms

When quantizing large language models to 4 bits or 8 bits, naive rounding degrades generation quality. `quantbench` implements the three leading post-training calibration techniques from scratch.

---

## 1. SmoothQuant Outlier Migration

LLM activations display extreme channel outliers (up to $100\times$ larger than normal channels), causing INT8 activation quantization to clip or collapse resolution.

SmoothQuant migrates dynamic range from difficult activations $X$ to easy-to-quantize weights $W$ using a per-channel scaling vector $s$:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

Where migration scale $s_j$ is given by:
$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}}$$

The scaling factor $s$ is absorbed directly into the preceding LayerNorm or RMSNorm weights, leaving runtime matrix multiplication intact.

---

## 2. Activation-Aware Weight Quantization (AWQ)

AWQ observes that **not all weights are equally important**: only the top $1\%\text{--}2\%$ of weights corresponding to large activation magnitudes govern model perplexity.

Rather than quantizing salient weights into separate FP16 layers, AWQ applies a per-channel protection scale $s$:

$$W' = W \cdot \text{diag}(s)$$

Optimal scale $\alpha \in [0, 1]$ is discovered via fast grid search on calibration tokens to minimize Frobenius reconstruction error:

$$\min_\alpha \| W X - \text{dequantize}(\text{quantize}(W \cdot \text{diag}(s))) \cdot \text{diag}(s)^{-1} X \|_F^2$$

---

## 3. GPTQ Second-Order Hessian Updates

GPTQ frames quantization as an Optimal Brain Surgeon (OBS) error compensation problem using second-order Taylor expansion:

$$E \approx \frac{1}{2} (w - \hat{w})^T H (w - \hat{w}), \quad H = 2 X X^T$$

1. **Cholesky Inversion**: Inverts the damped Hessian $H_{\text{damp}} = H + \lambda \cdot \text{mean}(\text{diag}(H)) \cdot I$ using Cholesky decomposition $H = L L^T$.
2. **Sequential Greedy Rounding**: When quantizing column $q$, downstream remaining columns $F$ are updated:
   $$\Delta W_{:, F} = - \frac{W_{:, q} - \hat{W}_{:, q}}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{q, F}$$
3. **Block Updates**: Accumulates errors across blocks (e.g. 128 columns) and updates remaining weights using single dense GEMMs.
