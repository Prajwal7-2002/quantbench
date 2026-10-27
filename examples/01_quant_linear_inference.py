"""Example 1: Drop-in QuantLinear Replacement & Bit-Packing Invariants.

Demonstrates converting PyTorch nn.Linear to quantbench QuantLinear across:
1. INT4 blockwise packed nibbles (G=128).
2. BitNet 1.58b {-1, 0, +1} ternary packing.
3. NVIDIA Ampere 2:4 structured sparsity.
4. Memory footprint reduction and state_dict serialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quantbench.nn.model_convert import convert_model_to_quant
from quantbench.nn.quant_linear import QuantLinear


def main() -> None:
    print("=" * 70)
    print("Example 1: Drop-in QuantLinear Replacement & Bit-Packing Invariants")
    print("=" * 70)

    torch.manual_seed(42)
    in_features, out_features = 2048, 4096
    linear_layer = nn.Linear(in_features, out_features, bias=True)
    orig_bytes = sum(p.numel() * p.element_size() for p in linear_layer.parameters())

    print(f"\n[1] Baseline PyTorch nn.Linear({in_features}, {out_features}):")
    print(f"    Parameters       : {in_features * out_features:,} weights")
    print(f"    Memory Footprint : {orig_bytes / (1024 * 1024):.2f} MB (FP32)")

    # 1. INT4 Blockwise QuantLinear (G=128)
    q_int4 = QuantLinear.from_linear(linear_layer, scheme="int4", group_size=128)
    int4_footprint = q_int4.memory_footprint_bytes()
    int4_bytes = int4_footprint["compressed_bytes"]
    print("\n[2] Converted to QuantLinear (INT4 Blockwise G=128):")
    print(
        f"    Packed Buffer    : {list(q_int4.packed_weight.shape)} ({q_int4.packed_weight.dtype})"
    )
    r_int4_fp32 = orig_bytes / int4_bytes
    r_int4_fp16 = int4_footprint["compression_ratio"]
    mb_int4 = int4_bytes / (1024 * 1024)
    print(
        f"    Memory Footprint : {mb_int4:.2f} MB "
        f"({r_int4_fp32:.2f}x vs FP32, {r_int4_fp16:.2f}x vs FP16)"
    )

    # 2. BitNet 1.58b Ternary QuantLinear
    q_ternary = QuantLinear.from_linear(linear_layer, scheme="ternary")
    ternary_footprint = q_ternary.memory_footprint_bytes()
    ternary_bytes = ternary_footprint["compressed_bytes"]
    r_t_fp32 = orig_bytes / ternary_bytes
    r_t_fp16 = ternary_footprint["compression_ratio"]
    mb_t = ternary_bytes / (1024 * 1024)
    print("\n[3] Converted to QuantLinear (BitNet 1.58b {-1, 0, +1}):")
    print(f"    Packed Buffer    : {list(q_ternary.packed_weight.shape)} (4 weights/byte)")
    print(
        f"    Memory Footprint : {mb_t:.2f} MB ({r_t_fp32:.2f}x vs FP32, {r_t_fp16:.2f}x vs FP16)"
    )

    # 3. Ampere 2:4 Structured Sparsity QuantLinear
    q_sparse = QuantLinear.from_linear(linear_layer, scheme="sparse24")
    sparse_footprint = q_sparse.memory_footprint_bytes()
    sparse_bytes = sparse_footprint["compressed_bytes"]
    r_s_fp32 = orig_bytes / sparse_bytes
    r_s_fp16 = sparse_footprint["compression_ratio"]
    mb_s = sparse_bytes / (1024 * 1024)
    print("\n[4] Converted to QuantLinear (NVIDIA Ampere 2:4 Structured Sparsity):")
    print(f"    Condensed Buffer : {list(q_sparse.condensed_weight.shape)} (50% non-zeros)")
    print(f"    Index Metadata   : {list(q_sparse.packed_metadata.shape)} (2-bit column nibbles)")
    print(
        f"    Memory Footprint : {mb_s:.2f} MB ({r_s_fp32:.2f}x vs FP32, {r_s_fp16:.2f}x vs FP16)"
    )

    # 4. Verify Forward Pass & Reconstruction SQNR
    x = torch.randn(4, in_features)
    y_orig = linear_layer(x)
    y_int4 = q_int4(x)
    y_ternary = q_ternary(x)

    def sqnr(orig: torch.Tensor, recon: torch.Tensor) -> float:
        signal = torch.sum(orig**2).item()
        noise = torch.sum((orig - recon) ** 2).item()
        return 10.0 * torch.log10(torch.tensor(signal / (noise + 1e-10))).item()

    print("\n[5] Forward Pass Execution (Batch=4):")
    print(f"    Output Shape     : {list(y_int4.shape)}")
    print(f"    INT4 SQNR (dB)   : {sqnr(y_orig, y_int4):.2f} dB")
    print(f"    BitNet SQNR (dB) : {sqnr(y_orig, y_ternary):.2f} dB")

    # 5. Full Model Conversion Demo
    class SimpleMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(1024, 512)
            self.lm_head = nn.Linear(512, 32000, bias=False)  # Excluded

    mlp = SimpleMLP()
    converted_mlp, stats = convert_model_to_quant(
        mlp,
        scheme="int4",
        group_size=128,
        exclude_modules=["lm_head"],
    )
    print("\n[6] Recursive Model Conversion (with lm_head exclusion):")
    print(f"    Converted Layers : {stats['converted_layers']}")
    print(f"    Excluded Layers  : {stats['skipped_layers']} (lm_head preserved in float)")
    c_ratio = stats["compression_ratio"]
    print(
        f"    Compression      : {c_ratio:.2f}x "
        f"({stats['orig_memory_mb']}MB -> {stats['quant_memory_mb']}MB)"
    )
    print(f"    Layer 1 Type     : {type(converted_mlp.fc1).__name__}")
    print(f"    Head Type        : {type(converted_mlp.lm_head).__name__}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
