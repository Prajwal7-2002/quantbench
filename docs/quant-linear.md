# Drop-in Neural Layers (`QuantLinear`)

In standard PyTorch, layers use `torch.nn.Parameter`, which forces 16-bit or 32-bit floats and tracks gradients.

`QuantLinear` (`quantbench.nn.quant_linear`) is a drop-in replacement for `torch.nn.Linear` that stores quantized weights directly inside sub-byte `uint8` persistent buffers (`register_buffer`).

---

## 1. Supported Formats

* **INT4 Groupwise**: Group sizes $G \in \{32, 64, 128\}$ with 4-bit nibbles packed into `uint8` arrays.
* **INT8 AbsMax**: Per-channel or per-tensor scales.
* **BitNet Ternary**: Biased 2-bit offset packing ($8\times$ compression).
* **Ampere 2:4 Structured Sparsity**: Compressed values + 4-bit column index nibbles.

---

## 2. Dynamic Forward Execution

During the forward pass, `QuantLinear` unpacks integer weights directly in memory and scales them dynamically:

```python
from quantbench.nn.quant_linear import QuantLinear
import torch.nn as nn

# Convert standard linear layer to INT4 blockwise (group size = 64)
linear = nn.Linear(1024, 4096)
quant_layer = QuantLinear.from_float(linear, format="int4", group_size=64)

# Forward pass with identical input dimensions
out = quant_layer(x)
```

---

## 3. Recursive Model Converter

Automatically replaces all `nn.Linear` layers across a Hugging Face model while excluding sensitive layers (such as the vocabulary `lm_head`):

```python
from quantbench.nn.model_convert import convert_model_to_quant

model = convert_model_to_quant(model, format="int4", group_size=128, exclude_modules=["lm_head"])
```
