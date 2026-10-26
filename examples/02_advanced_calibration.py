"""Example 2: Advanced Calibration Algorithms (SmoothQuant, AWQ & GPTQ).

Demonstrates first-principles implementation of:
1. SmoothQuant: Migration of activation outlier dynamic range into weights.
2. AWQ: Activation-aware grid search for salient channel protection scales.
3. GPTQ: Hessian covariance estimation, Cholesky inversion with damping, and OBS updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quantbench.quant.algorithms.awq import (
    compute_activation_channel_means,
    search_best_awq_scale,
)
from quantbench.quant.algorithms.gptq import compute_activation_hessian, gptq_quantize_weights
from quantbench.quant.algorithms.smoothquant import compute_smoothquant_scale


def main() -> None:
    print("=" * 75)
    print("Example 2: Advanced Calibration Algorithms (SmoothQuant, AWQ & GPTQ)")
    print("=" * 75)

    torch.manual_seed(42)
    in_features, out_features = 256, 512
    linear = nn.Linear(in_features, out_features, bias=False)
    w = linear.weight.data.clone()

    # Create synthetic calibration batch with realistic activation outlier channels
    batch_size, seq_len = 8, 32
    x = torch.randn(batch_size, seq_len, in_features)
    # Inject severe outlier channels (e.g. channels 10, 42, 100 have 50x magnitude)
    outlier_channels = [10, 42, 100]
    for ch in outlier_channels:
        x[..., ch] *= 50.0

    print(f"\n[1] Calibration Profiling ({batch_size} batches x {seq_len} tokens):")
    print(f"    Weight Matrix Shape   : {list(w.shape)}")
    print(f"    Activation Shape      : {list(x.shape)}")
    print(
        f"    Outlier Channels      : {outlier_channels} (magnitude ~{x[..., 10].abs().max():.1f})"
    )

    # =========================================================================
    # 1. SmoothQuant Outlier Migration
    # =========================================================================
    print("\n[2] SmoothQuant Channel Migration (alpha=0.5):")
    act_max = x.abs().view(-1, in_features).max(dim=0).values
    weight_max = w.abs().max(dim=0).values
    sq_scale = compute_smoothquant_scale(act_max, weight_max, alpha=0.5)

    x_smooth = x / sq_scale.view(1, 1, -1)
    w_smooth = w * sq_scale.view(1, -1)

    # Invariance check: Y = X * W = (X * s^-1) * (s * W)
    y_orig = torch.matmul(x, w.T)
    y_smooth = torch.matmul(x_smooth, w_smooth.T)
    max_inv_err = (y_orig - y_smooth).abs().max().item()

    print(f"    Peak Activation Mag   : {act_max.max():.2f} -> {x_smooth.abs().max():.2f} (smooth)")
    print(
        f"    Peak Weight Mag       : {weight_max.max():.2f} -> "
        f"{w_smooth.abs().max():.2f} (absorbed)"
    )
    print(f"    Mathematical Drift    : {max_inv_err:.2e} (Continuous Exact Invariance)")

    # =========================================================================
    # 2. AWQ Grid Search
    # =========================================================================
    print("\n[3] AWQ Frobenius Grid Search (alpha in [0.0, 1.0]):")
    act_means = compute_activation_channel_means(x)
    awq_scales, best_alpha = search_best_awq_scale(w, act_means, x.view(-1, in_features), n_grid=20)
    print(f"    Optimal Alpha Found   : {best_alpha:.2f}")
    print(f"    Computed AWQ Scales   : Shape {list(awq_scales.shape)}")
    print(f"    Scale Range           : Min {awq_scales.min():.4f}, Max {awq_scales.max():.4f}")
    ch_10_scale = awq_scales[10]
    mean_scale = awq_scales.mean()
    print(f"    Scale on Outlier Ch 10: {ch_10_scale:.4f} (protected vs mean {mean_scale:.4f})")

    # =========================================================================
    # 3. GPTQ Second-Order Hessian Compensation
    # =========================================================================
    print("\n[4] GPTQ Block-wise Hessian Optimization:")
    h = compute_activation_hessian(x, in_features=in_features)
    h_rank = torch.linalg.matrix_rank(h).item()
    print(f"    Hessian Matrix Shape  : {list(h.shape)} (Rank {h_rank})")

    # Quantize with GPTQ (INT4, Block Size=32, Damping=0.01)
    w_gptq = gptq_quantize_weights(w, h, bits=4, block_size=32, damping_factor=0.01)

    # Compare reconstruction error: Round-to-Nearest vs GPTQ
    from quantbench.quant.affine import quantize_absmax

    w_rtn_q, s_rtn = quantize_absmax(w, num_bits=4, dim=0)
    w_rtn = w_rtn_q.to(torch.float32) * s_rtn

    err_rtn = torch.norm(torch.matmul(x, w.T) - torch.matmul(x, w_rtn.T), p="fro").item()
    err_gptq = torch.norm(torch.matmul(x, w.T) - torch.matmul(x, w_gptq.T), p="fro").item()
    err_ratio = err_rtn / max(err_gptq, 1e-6)

    print(f"    Frobenius Error (RTN) : {err_rtn:.4f}")
    print(f"    Frobenius Error (GPTQ): {err_gptq:.4f} ({err_ratio:.2f}x error reduction!)")
    print("    Damping Regularization: 1.0% of Hessian diagonal mean")
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
