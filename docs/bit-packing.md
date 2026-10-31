# Sub-Byte Bit-Packing Runtimes

Standard CPU and GPU memory controllers address memory in 8-bit bytes. Storing 4-bit, 2-bit, or ternary weights requires bit-manipulation packing routines.

---

## 1. INT4 Nibble Packing (`int4_pack.py`)

Two 4-bit nibbles are packed into each `uint8` byte container:

$$\text{byte} = (v_0 \ \& \ \text{0x0F}) \mid ((v_1 \ \& \ \text{0x0F}) \ll 4)$$

```
Byte: [ b7 b6 b5 b4 | b3 b2 b1 b0 ]
         Value 1         Value 0
       (High Nibble)   (Low Nibble)
```

### Sign Extension in Two's Complement
Signed 4-bit values span $[-8, 7]$. When extracting the high nibble via arithmetic shift (`byte >> 4`), hardware handles sign propagation automatically. When extracting the low nibble:

$$v_{\text{signed}} = (v_{\text{raw}} \ge 8) \ ? \ (v_{\text{raw}} - 16) : v_{\text{raw}}$$

`quantbench` implements vectorized PyTorch tensor bit-shifts with zero CPU Python loops.

---

## 2. BitNet 1.58b Ternary Packing (`ternary_pack.py`)

BitNet weights belong to $\{-1, 0, +1\}$.

### Hardware-Friendly 2-Bit SIMD Packing
Maps ternary values to unsigned offset integers:
$$u = v + 1 \in \{0, 1, 2\}$$

Four weights are packed into one byte:
$$\text{byte} = u_0 \mid (u_1 \ll 2) \mid (u_2 \ll 4) \mid (u_3 \ll 6)$$
Achieves **$8\times$ compression** over FP16 with fast bitwise parallel dequantization.

### Base-3 Radix Packing
Because $3^5 = 243 \le 255$, five ternary values can theoretically fit into a single byte:
$$\text{byte} = \sum_{i=0}^4 u_i \cdot 3^i$$
Achieves theoretical maximal density ($\log_2(3) \approx 1.585$ bits per weight).
