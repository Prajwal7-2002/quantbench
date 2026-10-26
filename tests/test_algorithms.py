# Copyright 2026 Prajwal7-2002
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive test suite for advanced calibration algorithms (SmoothQuant, AWQ, GPTQ)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from quantbench.quant.affine import dequantize_absmax, quantize_absmax
from quantbench.quant.algorithms.awq import (
    awq_quantize_weights,
    compute_activation_channel_means,
    search_best_awq_scale,
)
from quantbench.quant.algorithms.gptq import (
    compute_activation_hessian,
    gptq_quantize_weights,
)
from quantbench.quant.algorithms.smoothquant import (
    compute_smoothquant_scale,
    extract_activation_channel_scales,
    smooth_linear_weights,
)


class TestSmoothQuant:
    """Mathematical verification and invariant tests for SmoothQuant."""

    def test_outlier_extraction_and_scale_computation(self) -> None:
        torch.manual_seed(42)
        # 100 tokens, 16 features
        x = torch.randn(100, 16)
        # Create an intentional outlier at channel 5
        x[:, 5] *= 80.0

        act_max = extract_activation_channel_scales(nn.Identity(), x)
        assert act_max.shape == (16,)
        assert act_max[5] > 70.0
        assert act_max[0] < 10.0

        weight = torch.randn(32, 16)
        w_max = weight.abs().max(dim=0).values

        # Test alpha=0.5 (balanced)
        scale = compute_smoothquant_scale(act_max, w_max, alpha=0.5)
        assert scale.shape == (16,)
        # Channel 5 should have higher scale to migrate activation difficulty to weight
        assert scale[5] > scale[0]

    def test_mathematical_continuous_invariance(self) -> None:
        """Verify that before discrete rounding, (X * s^-1) @ (W * s)^T == X @ W^T."""
        torch.manual_seed(123)
        b, s, c_in, c_out = 2, 8, 32, 64
        x = torch.randn(b, s, c_in) * 2.0
        x[:, :, 3] *= 50.0  # outlier channel
        w = torch.randn(c_out, c_in)

        act_scales = extract_activation_channel_scales(nn.Identity(), x)
        w_smoothed, scale = smooth_linear_weights(w, act_scales, alpha=0.5)

        # Scale activation: X_hat = X * diag(s)^-1
        x_smoothed = x / scale.unsqueeze(0).unsqueeze(0)

        # Reference output: X @ W^T
        y_ref = x @ w.t()
        # Smoothed output: X_hat @ W_hat^T
        y_smooth = x_smoothed @ w_smoothed.t()

        # In continuous precision, the outputs must be mathematically identical
        torch.testing.assert_close(y_smooth, y_ref, rtol=1e-4, atol=1e-4)

    def test_alpha_boundaries(self) -> None:
        act = torch.tensor([10.0, 1.0])
        w = torch.tensor([1.0, 10.0])

        # alpha = 0.0 -> s = 1.0 / w
        s_0 = compute_smoothquant_scale(act, w, alpha=0.0)
        torch.testing.assert_close(s_0, 1.0 / w)

        # alpha = 1.0 -> s = act
        s_1 = compute_smoothquant_scale(act, w, alpha=1.0)
        torch.testing.assert_close(s_1, act)

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_smoothquant_scale(act, w, alpha=1.5)


class TestAWQ:
    """Verification for Activation-aware Weight Quantization."""

    def test_activation_means_and_shape(self) -> None:
        x1 = torch.ones(10, 8) * 2.0
        x2 = torch.ones(10, 8) * 4.0
        means = compute_activation_channel_means([x1, x2])
        assert means.shape == (8,)
        torch.testing.assert_close(means, torch.ones(8) * 3.0)

    def test_awq_reconstruction_error_reduction(self) -> None:
        """Verify that AWQ search produces equal or lower output MSE than unscaled quantization."""
        torch.manual_seed(99)
        c_in, c_out = 64, 32
        n_samples = 128

        # Create activation distribution with sharp salient channels
        x = torch.randn(n_samples, c_in)
        x[:, 4] *= 25.0
        x[:, 12] *= 30.0

        w = torch.randn(c_out, c_in) * 0.1

        act_means = compute_activation_channel_means(x)

        # Base uniform quantization (alpha = 0.0)
        q_naive, scale_naive = quantize_absmax(w, num_bits=4, dim=1)
        w_naive = dequantize_absmax(q_naive, scale_naive)
        err_naive = torch.norm((x @ w.t()) - (x @ w_naive.t()), p="fro").item()

        # AWQ grid search
        best_scale, best_alpha = search_best_awq_scale(
            w,
            act_means,
            calibration_x=x,
            n_grid=10,
        )
        assert best_scale.shape == (c_in,)
        assert 0.0 <= best_alpha <= 1.0

        w_awq, _ = awq_quantize_weights(w, best_scale)
        err_awq = torch.norm((x @ w.t()) - (x @ w_awq.t()), p="fro").item()

        # AWQ should achieve competitive or lower reconstruction error on calibration data
        assert err_awq <= err_naive * 1.05


class TestGPTQ:
    """Mathematical verification and second-order Hessian error compensation tests."""

    def test_hessian_properties(self) -> None:
        torch.manual_seed(77)
        in_dim = 16
        x = torch.randn(50, in_dim)

        H = compute_activation_hessian(x, in_features=in_dim)
        assert H.shape == (in_dim, in_dim)

        # Hessian must be symmetric: H == H^T
        torch.testing.assert_close(H, H.t(), atol=1e-5, rtol=1e-5)

        # Eigenvalues must be non-negative (positive semi-definite)
        eigvals = torch.linalg.eigvalsh(H)
        assert (eigvals >= -1e-5).all()

    def test_gptq_quantize_weights_error_reduction(self) -> None:
        """Verify that GPTQ second-order compensation improves output reconstruction."""
        torch.manual_seed(42)
        in_dim, out_dim = 64, 32
        n_tokens = 256

        x = torch.randn(n_tokens, in_dim)
        w = torch.randn(out_dim, in_dim) * 0.2

        H = compute_activation_hessian(x, in_features=in_dim)

        # Naive INT4 quantization without OBS compensation
        q_naive, scale_naive = quantize_absmax(w, num_bits=4, dim=1)
        w_naive = dequantize_absmax(q_naive, scale_naive)
        err_naive = torch.norm((x @ w.t()) - (x @ w_naive.t()), p="fro").item()

        # GPTQ quantization with Hessian compensation
        w_gptq = gptq_quantize_weights(
            w,
            H=H,
            bits=4,
            group_size=32,
            block_size=16,
            damping_factor=0.01,
        )
        assert w_gptq.shape == w.shape
        err_gptq = torch.norm((x @ w.t()) - (x @ w_gptq.t()), p="fro").item()

        # GPTQ compensation must produce lower error than naive rounding
        assert err_gptq < err_naive

    def test_gptq_non_divisible_block_size(self) -> None:
        """Verify robust behavior when in_features is not evenly divisible by block_size."""
        torch.manual_seed(10)
        in_dim, out_dim = 25, 15
        x = torch.randn(40, in_dim)
        w = torch.randn(out_dim, in_dim)

        H = compute_activation_hessian(x, in_features=in_dim)
        # block_size = 8 (25 is not divisible by 8)
        w_gptq = gptq_quantize_weights(w, H, bits=4, group_size=-1, block_size=8)
        assert w_gptq.shape == (out_dim, in_dim)
        assert not torch.isnan(w_gptq).any()
