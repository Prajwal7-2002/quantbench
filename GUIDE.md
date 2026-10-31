# The Comprehensive Guide to LLM Quantization & Hardware Sparsity ⚡
**From First-Principles Mathematical Formulations to Vectorized Bit-Packing Runtimes and Pareto Evaluation**

*Author: Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))*  
*Project: [quantbench](https://github.com/Prajwal7-2002/quantbench)*

---

## 1. Introduction: Why Quantization is Mandatory in LLM Serving

When serving modern Generative Large Language Models (LLMs), the primary inference bottleneck during autoregressive token generation (the decoding phase) is **memory bandwidth, not compute capacity**.

### The Arithmetic Intensity Bottleneck
In the autoregressive decode loop, the model processes tokens with batch size $B=1$ and query length 1:
* The GEMV (General Matrix-Vector multiplication) arithmetic intensity is:
  $$\text{Arithmetic Intensity} \approx \frac{2 \cdot P \text{ FLOPs}}{2 \cdot P \text{ Bytes}} = 1.0 \text{ FLOP/Byte} \quad (\text{in FP16})$$
* On an NVIDIA A100 SXM (312 TFLOPS FP16 Tensor Core, 2,039 GB/s HBM2e bandwidth), the hardware ridge point is:
  $$\text{Ridge Point} = \frac{312 \times 10^{12} \text{ FLOP/s}}{2,039 \times 10^9 \text{ B/s}} \approx 153 \text{ FLOPs/Byte}$$
* Since $1.0 \ll 153$, the GPU Tensor Cores sit **99.3% idle**, starved for weights streaming from high-bandwidth memory (HBM).

### Memory Bandwidth Reduction = Linear Decode Acceleration
By compressing weights from 16-bit to 4-bit (INT4):
* Weight memory footprint is cut by $\mathbf{4.0\times}$.
* The byte volume transferred over the memory bus per generated token drops by $\mathbf{4.0\times}$.
* Autoregressive decoding throughput increases nearly linearly with the compression factor!

---

## 2. Mathematical Foundations of Quantization

### 2.1. The Zero-Preserving Affine Invariant
Quantization maps continuous real numbers $x \in [\alpha, \beta]$ to a discrete integer grid $q \in [q_{\min}, q_{\max}]$.

$$\text{scale} = s = \frac{\beta - \alpha}{q_{\max} - q_{\min}}$$
$$\text{zero\_point} = z = \text{round}\left( - \frac{\alpha}{s} \right) + q_{\min}$$
$$q = \text{clamp}\left( \text{round}\left( \frac{x}{s} \right) + z, \; q_{\min}, \; q_{\max} \right)$$
$$\hat{x} = (q - z) \cdot s$$

#### Why Real $0.0$ Must Map Exactly to an Integer Zero-Point $z$:
In neural network operations:
1. **Padded Tokens**: Sequences of varying lengths use zero-padding ($x = 0.0$).
2. **ReLU / SwiGLU Zeros**: Non-linear activations introduce exact zero activations.
3. If $x = 0.0$ did not map to an integer $z$:
   $$\hat{x} = (\text{round}(0 + z_{\text{float}}) - z_{\text{float}}) \cdot s \ne 0.0$$
   Padding tokens would leak non-zero bias signals across self-attention projections, corrupting transformer output logits!

### 2.2. Symmetric AbsMax Quantization
When data distributions are symmetric around zero (e.g. weights centered at mean $0$):
* Restrict zero point to strictly $z = 0$.
* For signed $b$-bit integer ($q_{\max} = 2^{b-1} - 1$):
  $$s = \frac{\max(|x|)}{2^{b-1} - 1}$$
  $$q = \text{clamp}\left( \text{round}\left( \frac{x}{s} \right), \; -q_{\max}, \; q_{\max} \right)$$
  $$\hat{x} = q \cdot s$$
* In symmetric AbsMax, dequantization simplifies from $(q - z) s$ to pure multiplication $q \cdot s$, eliminating subtraction instructions from GPU GEMM inner loops.

### 2.3. Group-wise / Block-wise Quantization
Per-tensor quantization uses a single scale $s$ for the entire weight matrix. A single outlier value pushes $s$ up, severely degrading the precision of normal values.

Block-wise quantization slices the weight matrix into independent groups of size $G \in \{32, 64, 128\}$:
* Each block $k$ computes its own independent scale $s_k$ and zero-point $z_k$.
* **The Granularity Trade-off**:
  * Smaller $G$ (e.g. 32): Higher precision, but scales consume $16 / 32 = 0.5$ bits/weight (12.5% metadata overhead).
  * Larger $G$ (e.g. 128): Lower metadata overhead ($16 / 128 = 0.125$ bits/weight), but slightly reduced ability to isolate local outliers.
  * In practice, **$G = 128$ or $G = 64$** represents the Pareto-optimal operating point for LLMs.

---

## 3. Modern FP8 Formats (OCP Standard)

The Open Compute Project (OCP) standardizes two distinct 8-bit floating point formats:

| Specification | OCP FP8 E4M3FN | OCP FP8 E5M2 |
|---|---|---|
| **Bit Layout** | 1 sign, 4 exponent, 3 mantissa | 1 sign, 5 exponent, 2 mantissa |
| **Exponent Bias** | 7 | 15 |
| **Max Absolute Value** | $\pm 448.0$ | $\pm 57344.0$ |
| **Mantissa Resolution** | High ($2^{-3} = 0.125$) | Low ($2^{-2} = 0.25$) |
| **Primary Use-Case** | **Forward GEMM** (Weights & Activations) | **KV Cache, Gradients, Optimizer States** |

### Analytical Emulation on Older GPUs / CPUs
Hardware lacking native FP8 instructions (e.g. RTX 30-series, Apple Silicon, CPUs) cannot natively execute `torch.float8_e4m3fn`. `quantbench` provides a bit-exact software emulator that analytically simulates exponent bias, mantissa truncation, and saturation bounds:
```python
# Analytical bit-level reconstruction
e = clamp(floor(log2(|x|)) + bias, 0, 2^exp_bits - 1)
m = round((|x| / 2^(e - bias) - 1.0) * 2^mantissa_bits)
x_hat = sign(x) * (1.0 + m / 2^mantissa_bits) * 2^(e - bias)
```

---

## 4. Sub-Byte Vectorized Bit-Packing Runtimes

Modern CPU and GPU memory subsystems are byte-addressable (minimum addressable boundary is **8 bits**). Storing 4-bit, 2-bit, or ternary weights requires sub-byte bit packing into `uint8` containers.

### 4.1. Vectorized INT4 Packing
* Two 4-bit values $v_0$ and $v_1$ pack into 1 byte:
  $$\text{byte} = (v_0 \& 0x0F) \mid ((v_1 \& 0x0F) \ll 4)$$
* Unpacking:
  $$v_0 = \text{byte} \& 0x0F, \quad v_1 = (\text{byte} \gg 4) \& 0x0F$$
* **Signed Two's Complement Extension**:
  If bit 3 is set ($v \ge 8$), sign-extend to negative: $v_{\text{signed}} = v - 16$.

### 4.2. BitNet 1.58b Ternary Packing
Constrains weights strictly to $\{-1, 0, +1\}$:
1. **2-bit Biased SIMD Packing (Fastest)**:
   * Maps $u = v + 1 \in \{0, 1, 2\}$.
   * 4 weights fit into 1 byte:
     $$\text{byte} = u_0 \mid (u_1 \ll 2) \mid (u_2 \ll 4) \mid (u_3 \ll 6)$$
   * Decodes via parallel SIMD bit shifts without integer division.
2. **Radix-3 Base-3 Packing (Maximum Density)**:
   * 5 weights fit into 1 byte since $3^5 = 243 \le 255$:
     $$\text{byte} = u_0 + 3 u_1 + 9 u_2 + 27 u_3 + 81 u_4$$
   * Achieves theoretical $\log_2(3) \approx 1.585 \text{ bits/weight}$.

### 4.3. NVIDIA Ampere/Ada/Hopper 2:4 Structured Sparsity
NVIDIA Tensor Cores natively support 2:4 sparsity (in every 4 contiguous elements, at least 2 must be zero):
1. **Condensed Matrix**: Compresses non-zeros into dense matrix $(M, N/2)$ (50% reduction).
2. **Metadata Nibbles**: Records 2-bit column indices ($0..3$) for each non-zero element.
3. Two 4-element blocks (8 dense elements, 4 non-zeros) pack into **1 metadata byte**:
   $$\text{byte} = (idx_{A,0} \& 3) \mid ((idx_{A,1} \& 3) \ll 2) \mid ((idx_{B,0} \& 3) \ll 4) \mid ((idx_{B,1} \& 3) \ll 6)$$
4. Metadata overhead is strictly **2 bits per non-zero value** (only 12.5% of FP16 weight size).

---

## 5. Advanced Post-Training Calibration Algorithms

### 5.1. SmoothQuant: Activation Outlier Migration
* **Insight**: LLM activations have systematic outlier channels ($100\times$ normal values), while weights are smooth.
* **Invariant Transformation**:
  $$Y = X W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$
* **Scale Derivation**:
  $$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}} \quad (\alpha=0.5 \text{ balances activation and weight})$$
* Scale vector $s$ is folded into preceding LayerNorm/RMSNorm weights, enabling standard INT8 Tensor Core GEMM.

### 5.2. AWQ: Activation-Aware Weight Quantization
* **Insight**: Only top 1% salient weight channels preserve perplexity.
* Salience is determined by mean activation magnitude: $s_X = \frac{1}{N} \sum |X|$.
* Instead of keeping salient weights in FP16 (which ruins uniform SIMD kernels), AWQ applies a per-channel protection scale:
  $$s = s_X^\alpha$$
  where $\alpha \in [0, 1]$ is optimized via grid search on calibration tokens to minimize Frobenius reconstruction error:
  $$\min_\alpha \| W X - \text{dequantize}(\text{quantize}(W \text{diag}(s))) \text{diag}(s)^{-1} X \|_F^2$$

### 5.3. GPTQ: Second-Order Hessian Compensation
* Uses second-order Taylor expansion of quantization error:
  $$E \approx \frac{1}{2} (w - \hat{w})^T H (w - \hat{w}), \quad H = 2 X X^T$$
* **Damping Regularization**:
  $$H_{\text{damp}} = H + \lambda \cdot \text{mean}(\text{diag}(H)) \cdot I \quad (\lambda = 0.01)$$
* Inverts via Cholesky decomposition: $H = L L^T \implies H^{-1} = L^{-T} L^{-1}$.
* **Optimal Brain Surgeon (OBS) Update**:
  When rounding column $q$, updates remaining columns $F$:
  $$\Delta W_{:, F} = - \frac{W_{:, q} - \hat{W}_{:, q}}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{q, F}$$
* **Block Updates**: Accumulates errors in blocks of $B$ columns (e.g. 128) and performs dense matrix multiplications (GEMM) to update the remaining matrix.

---

## 6. Drop-in PyTorch Architecture & In-Place Conversion

### Why Persistent Buffers Beat `nn.Parameter` for Inference
1. `nn.Parameter` forces float dtypes and triggers Autograd gradient tracking (`requires_grad=True`).
2. If touched by an optimizer, it allocates momentum and gradient buffers ($2\times\text{--}3\times$ model size).
3. `QuantLinear` stores packed weights as persistent buffers (`self.register_buffer("packed_weight", ...)`):
   * Zero gradient tracking.
   * Zero optimizer memory overhead.
   * Automatically serializes into `state_dict()` and `safetensors`.
4. **Dynamic Checkpoint Hook (`_load_from_state_dict`)**:
   Overrides PyTorch's hook to register serialized buffer shapes dynamically before strict key validation runs.

---

## 7. Quality-vs-Speed Pareto Benchmarking

### Dual-Metric Evaluation
1. **Layer-Level SQNR (dB)**:
   $$\text{SQNR (dB)} = 10 \cdot \log_{10} \left( \frac{\sum |W|^2}{\sum |W - \hat{W}|^2 + \epsilon} \right)$$
   Diagnoses numerical reconstruction fidelity layer-by-layer without inference passes.
2. **Model-Level Sliding-Window Perplexity**:
   Uses context window $L$ with stride $S < L$, evaluating loss strictly on trailing $S$ tokens to eliminate chunk boundary context collapse.
3. **Pareto Table Output**:
   Maps configurations across (Compression Ratio, Footprint MB, Delta PPL, SQNR dB).

---

## 8. Summary Comparison of Quantization Schemes

| Scheme | Target Bits | Memory Reduction | Compute Speedup | Accuracy Impact | Recommended Use-Case |
|---|---|---|---|---|---|
| **FP16 Baseline** | 16-bit | $1.0\times$ | $1.0\times$ | Baseline | Reference quality |
| **FP8 (E4M3)** | 8-bit | $2.0\times$ | $1.8\times\text{--}2.0\times$ | Near zero ($\Delta \text{PPL} < 0.01$) | Standard serving on Ada/Hopper |
| **INT8 AbsMax** | 8-bit | $2.0\times$ | $1.7\times\text{--}1.9\times$ | Near zero ($\Delta \text{PPL} < 0.02$) | Universal CPU/GPU acceleration |
| **INT4 Blockwise** | 4-bit | $3.2\times\text{--}3.6\times$ | $2.5\times\text{--}3.5\times$ | Low ($\Delta \text{PPL} < 0.15$) | VRAM-constrained edge / consumer GPUs |
| **AWQ / GPTQ INT4** | 4-bit | $3.5\times$ | $3.0\times$ | Minimal ($\Delta \text{PPL} < 0.05$) | Production 4-bit LLM deployment |
| **Ampere 2:4 Sparse**| 8-bit eff. | $2.0\times$ | $2.0\times$ math | Negligible ($\Delta \text{PPL} < 0.08$) | Dense-throughput acceleration |
| **BitNet Ternary** | 1.58-bit | $7.5\times\text{--}8.0\times$ | $5.0\times\text{--}7.0\times$ | Requires fine-tuning | Ultra-low power / specialized silicon |
