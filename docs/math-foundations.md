# Mathematical Foundations of Quantization

Quantization projects continuous floating-point tensors into discrete low-bit integer or floating-point representations.

---

## 1. Affine Asymmetric Quantization

Maps floating-point interval $[x_{\min}, x_{\max}]$ to integer grid $[q_{\min}, q_{\max}]$:

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

$$z = \text{round}\left(-\frac{x_{\min}}{s}\right) + q_{\min}$$

$$q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z, \; q_{\min}, \; q_{\max}\right)$$

Dequantization reconstructs the real value:

$$\hat{x} = (q - z) \cdot s$$

### The Zero-Preserving Invariant
If a tensor is purely positive (e.g. ReLU activations in $[0.5, 12.0]$), a naive min-max scaling causes integer zero to represent a non-zero float. To preserve exact real-world zero:

$$x_{\min} = \min(x_{\min}, 0.0), \quad x_{\max} = \max(x_{\max}, 0.0)$$

This guarantees that zero-padding tokens and causal attention masks dequantize to exact $0.0$, preventing bias injection into transformer hidden states.

---

## 2. Symmetric AbsMax Quantization

Centers the dynamic range at zero ($z = 0$), eliminating per-element subtraction during hardware matrix multiplication:

$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$

$$q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), \; -(2^{b-1}), \; 2^{b-1}-1\right)$$

$$\hat{x} = q \cdot s$$

---

## 3. Block-Wise (Group-Wise) Quantization

Per-tensor quantization uses a single scale $s$ for an entire weight matrix. A single outlier forces $s$ to expand, destroying the resolution of normal channels.

Block-wise quantization slices the matrix into independent groups of size $G \in \{32, 64, 128\}$:
* Each block computes an independent scale $s_k$ and zero-point $z_k$.
* For $G = 128$, scale metadata adds only $16 / 128 = 0.125$ bits per weight, providing near-FP16 fidelity with INT4 weights.

---

## 4. OCP FP8 Formats (E4M3FN vs E5M2)

`quantbench` implements bit-exact software emulation of Open Compute Project (OCP) 8-bit floats:

| Format | Bit Layout | Exponent Bias | Max Finite Value | Primary Use Case |
| :--- | :--- | :---: | :---: | :--- |
| **E4M3FN** | 1 sign, 4 exp, 3 mantissa | 7 | 448.0 | GEMM weights and activations (high precision) |
| **E5M2** | 1 sign, 5 exp, 2 mantissa | 15 | 57344.0 | Gradients and attention KV-caches (wide dynamic range) |
