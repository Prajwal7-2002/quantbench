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

"""AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.

Mathematical Foundation:
    Not all weights in a neural network are equally critical to perplexity.
    AWQ discovers that protecting the top ~1% of salient weights avoids perplexity
    degradation even at aggressive INT4/INT3 regimes. Crucially, salient weights
    directly correspond to channels that process activations with high magnitudes:
        s_X = (1 / N) * sum(|X|)   (mean absolute activation per input channel)

    Instead of maintaining mixed-precision floating-point formats (which incur runtime
    overhead and degrade hardware SIMD execution efficiency), AWQ applies a per-channel
    protection scale s:
        W' = W * diag(s)
        X' = diag(s)^-1 * X

    When W' is quantized to sub-byte integers (e.g. INT4 blockwise), the relative quantization
    error on salient channels is inversely proportional to s_j:
        Delta W_j / W_j ~ 1 / s_j

    AWQ searches for the optimal per-channel scale s_j = (s_{X, j})^alpha over a grid of
    alpha in [0.0, 1.0] by directly minimizing the Frobenius output reconstruction error:
        min_alpha || W * X - [Quantize(W * diag(s)) * diag(s)^-1] * X ||_F^2
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise


def compute_activation_channel_means(
    calibration_inputs: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Calculate mean absolute activation magnitude per input channel.

    Args:
        calibration_inputs: Tensor of shape (..., in_features) or list of tensors.

    Returns:
        Tensor of shape (in_features,) containing per-channel mean absolute activation.
    """
    if isinstance(calibration_inputs, torch.Tensor):
        calibration_inputs = [calibration_inputs]

    total_sum: Optional[torch.Tensor] = None
    total_tokens = 0

    for batch in calibration_inputs:
        flat = batch.detach().view(-1, batch.shape[-1]).abs()
        batch_sum = flat.sum(dim=0)
        n_tokens = flat.shape[0]

        if total_sum is None:
            total_sum = batch_sum
        else:
            total_sum = total_sum + batch_sum
        total_tokens += n_tokens

    if total_sum is None or total_tokens == 0:
        raise ValueError("calibration_inputs cannot be empty.")

    return total_sum / total_tokens


def search_best_awq_scale(
    weight: torch.Tensor,
    act_means: torch.Tensor,
    calibration_x: torch.Tensor,
    quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    n_grid: int = 20,
    max_alpha: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """Perform grid search for optimal AWQ protection scale alpha.

    Args:
        weight: Float tensor of shape (out_features, in_features).
        act_means: Channel-wise activation mean of shape (in_features,).
        calibration_x: Sample calibration activations of shape (N_samples, in_features).
        quant_fn: Quantize-and-dequantize callable. Defaults to INT4 blockwise (group_size=128).
        n_grid: Number of search points in alpha grid [0, max_alpha].
        max_alpha: Maximum alpha value for search.

    Returns:
        Tuple of (optimal_scale_vector, best_alpha):
            - optimal_scale_vector: Per-channel scale s in R^{in_features}.
            - best_alpha: Optimal alpha hyperparameter.
    """
    if quant_fn is None:
        def default_quant(w: torch.Tensor) -> torch.Tensor:
            q_b, s_b, zp_b, meta = quantize_blockwise(
                w, group_size=min(128, w.shape[1]), num_bits=4
            )
            return dequantize_blockwise(q_b, s_b, zp_b, meta)
        quant_fn = default_quant

    # Reference unquantized linear output
    # calibration_x: (N, in_features), weight: (out_features, in_features)
    # y_ref: (N, out_features)
    y_ref = calibration_x @ weight.t()

    best_error = float("inf")
    best_scale = torch.ones(weight.shape[1], device=weight.device, dtype=weight.dtype)
    best_alpha = 0.0

    act_means = act_means.clamp(min=1e-5)

    alphas = torch.linspace(0.0, max_alpha, steps=n_grid)

    for alpha_val in alphas:
        alpha = float(alpha_val.item())
        # Candidate scale s = act_means^alpha
        cand_scale = act_means.pow(alpha)
        # Normalize scale by geometric mean of min/max to maintain weight scale stability
        geom_mean = (cand_scale.max() * cand_scale.min()).sqrt().clamp(min=1e-5)
        cand_scale = (cand_scale / geom_mean).clamp(min=1e-4)

        # Scale weights: W' = W * diag(s)
        # weight: (out, in), cand_scale: (in,) -> broadcast along rows
        w_scaled = weight * cand_scale.unsqueeze(0)

        # Quantize and dequantize
        w_dequant = quant_fn(w_scaled)

        # Inverse scale: W_cand = w_dequant * diag(s)^-1
        w_cand = w_dequant / cand_scale.unsqueeze(0)

        # Output error || y_ref - x @ w_cand.t() ||_F^2
        y_cand = calibration_x @ w_cand.t()
        err = torch.norm(y_ref - y_cand, p="fro").item()

        if err < best_error:
            best_error = err
            best_scale = cand_scale
            best_alpha = alpha

    return best_scale, best_alpha


def awq_quantize_weights(
    weight: torch.Tensor,
    scale: torch.Tensor,
    quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply optimal AWQ scale and quantize weight matrix.

    Args:
        weight: Original weight of shape (out_features, in_features).
        scale: Optimal per-channel protection scale of shape (in_features,).
        quant_fn: Quantization-dequantization function.

    Returns:
        Tuple of (dequantized_weight, scale).
    """
    if quant_fn is None:
        def default_quant(w: torch.Tensor) -> torch.Tensor:
            q_b, s_b, zp_b, meta = quantize_blockwise(
                w, group_size=min(128, w.shape[1]), num_bits=4
            )
            return dequantize_blockwise(q_b, s_b, zp_b, meta)
        quant_fn = default_quant

    w_scaled = weight * scale.unsqueeze(0)
    w_dequant = quant_fn(w_scaled)
    # Invert scale so layer can be run directly with unscaled input:
    # y = x @ (w_dequant / s).t()
    w_effective = w_dequant / scale.unsqueeze(0)
    return w_effective, scale
