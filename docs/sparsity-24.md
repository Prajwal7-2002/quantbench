# NVIDIA 2:4 Structured Sparsity (`sparse_pack.py`)

NVIDIA Ampere, Ada Lovelace, and Hopper architectures support hardware-accelerated **2:4 structured sparsity**.

---

## 1. The Hardware Invariant

For every block of **4 consecutive values along a matrix row, exactly 2 values must be zero** (50% sparsity).

```
Dense 4-element block:   [ 0.0,  3.5,  0.0, -1.2 ]
Retained non-zero values: [ 3.5, -1.2 ]  (Column indices: 1 and 3)
2-bit column metadata:   idx0 = 1 (0b01), idx1 = 3 (0b11)
Metadata nibble:         (0b01 & 0x03) | ((0b11 & 0x03) << 2) = 0b1101 (13 decimal)
```

---

## 2. Memory & Math Acceleration

1. **Storage Reduction**: Retained non-zero weights form a compressed dense matrix of shape $(M, N/2)$ — slashing memory traffic by **50%**.
2. **Metadata Layout**: Two 2-bit indices form a 4-bit nibble per 4 elements. Eight matrix columns produce one 8-bit metadata byte.
3. **Sparse Tensor Cores**: Hardware skips zero multiply-accumulate operations, doubling theoretical peak TFLOPS ($2.0\times$ speedup).
