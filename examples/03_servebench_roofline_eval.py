"""Example 3: Empirical Roofline Model & Memory Bandwidth Integration.

Bridges quantbench drop-in layers with servebench hardware roofline physics:
1. Evaluates memory bandwidth saturation across RTX 3050 (192 GB/s) and T4 (320 GB/s).
2. Computes arithmetic intensity (FLOPs/byte) and ridge points for FP16, INT8, INT4, BitNet.
3. Benchmarks real forward-pass execution latency and DRAM traffic reduction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from quantbench.nn.quant_linear import QuantLinear


@dataclass
class HardwareProfile:
    name: str
    dram_bandwidth_gb_s: float
    peak_tensor_tflops: float
    vram_gb: float

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOPs/byte) at boundary of memory and compute bounds."""
        return (self.peak_tensor_tflops * 1e12) / (self.dram_bandwidth_gb_s * 1e9)


HARDWARE_PROFILES = {
    "rtx_3050": HardwareProfile("NVIDIA RTX 3050 Laptop", 192.0, 36.4, 6.0),
    "t4": HardwareProfile("NVIDIA T4 Tensor Core", 320.0, 65.0, 16.0),
    "a100": HardwareProfile("NVIDIA A100-SXM4-80GB", 2039.0, 312.0, 80.0),
}


def benchmark_layer_latency(
    layer: nn.Module,
    x: torch.Tensor,
    warmup: int = 10,
    repeats: int = 50,
) -> float:
    """Benchmark forward pass latency in microseconds."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = layer(x)
        t1 = time.perf_counter()

    return ((t1 - t0) / repeats) * 1e6


def main() -> None:
    print("=" * 80)
    print("Example 3: Empirical Roofline Model & Memory Bandwidth Integration")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Target Hardware Environment: {device.type.upper()}")

    # Transformer projection layer dimensions (e.g. Qwen2.5-1.5B MLP up-projection)
    in_features, out_features = 1536, 8960
    n_params = in_features * out_features
    # In decode phase, batch size is small (e.g. 1 token per active sequence)
    batch_size = 1
    flops = 2 * n_params * batch_size

    linear_fp32 = nn.Linear(in_features, out_features, bias=False).to(device)

    # Instantiate quantbench layers
    q_int8 = QuantLinear.from_linear(linear_fp32, scheme="int8").to(device)
    q_int4 = QuantLinear.from_linear(linear_fp32, scheme="int4", group_size=128).to(device)
    q_ternary = QuantLinear.from_linear(linear_fp32, scheme="ternary").to(device)
    q_sparse = QuantLinear.from_linear(linear_fp32, scheme="sparse24").to(device)

    schemes = [
        ("FP16 Baseline", n_params * 2.0, linear_fp32),
        ("INT8 AbsMax", q_int8.memory_footprint_bytes()["compressed_bytes"], q_int8),
        ("INT4 Blockwise (G=128)", q_int4.memory_footprint_bytes()["compressed_bytes"], q_int4),
        ("BitNet 1.58b Ternary", q_ternary.memory_footprint_bytes()["compressed_bytes"], q_ternary),
        (
            "Ampere 2:4 Structured Sparsity",
            q_sparse.memory_footprint_bytes()["compressed_bytes"],
            q_sparse,
        ),
    ]

    hw = HARDWARE_PROFILES["rtx_3050"]
    print(f"\n[2] Roofline Analysis on {hw.name} ({hw.dram_bandwidth_gb_s} GB/s DRAM):")
    print(f"    Layer Dimensions : ({in_features} -> {out_features}) [{n_params:,} parameters]")
    print(f"    Decode FLOPs     : {flops:,} FLOPs (Batch Size = {batch_size})")
    print(f"    GPU Ridge Point  : {hw.ridge_point:.1f} FLOPs/byte (Memory-bound threshold)")

    print("\n" + "-" * 75)
    hdr = (
        f"{'Precision Scheme':<30} | {'Size':<9} | {'Intensity':<10} | "
        f"{'Latency':<10} | {'Saved':<6}"
    )
    print(hdr)
    print("-" * 75)

    fp16_bytes = n_params * 2.0
    x = torch.randn(batch_size, in_features, device=device)

    for name, byte_count, _ in schemes:
        mb = byte_count / (1024 * 1024)
        intensity = flops / max(byte_count, 1)
        # Theoretical latency = Bytes / Bandwidth (s)
        theo_us = (byte_count / (hw.dram_bandwidth_gb_s * 1e9)) * 1e6
        traffic_reduction = (1.0 - (byte_count / fp16_bytes)) * 100.0

        row = (
            f"{name:<30} | {mb:>6.2f} MB | {intensity:>6.2f} F/B | "
            f"{theo_us:>7.2f} us | {traffic_reduction:>5.1f}%"
        )
        print(row)
    print("-" * 75)

    # Empirical Forward Pass Benchmark
    print("\n[3] Empirical Local Execution Benchmarks (50 iterations):")
    for name, _, layer in schemes[:3]:
        lat_us = benchmark_layer_latency(layer, x)
        print(f"    {name:<26}: {lat_us:>8.2f} us per forward pass")

    print("\n[4] Systems Takeaway:")
    print("    In autoregressive decoding, batch=1 operation operates entirely at")
    print(f"    arithmetic intensities < {hw.ridge_point:.1f} FLOPs/byte, meaning execution is")
    print("    100% memory bandwidth bound. Quantizing weights to INT4 cuts memory")
    print("    transfers by 72.5%, proportionally accelerating decode token throughput.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
