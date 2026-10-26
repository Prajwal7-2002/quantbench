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

"""SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.

Mathematical Foundation:
    LLM activations exhibit severe outlier channels whose magnitudes can be 100x
    larger than typical channels, making activation quantization to INT8 prone to
    catastrophic precision loss under per-tensor or per-token absmax scaling.
    Conversely, weights exhibit smooth channel distributions and are easy to quantize.

    SmoothQuant introduces a mathematically invariant per-channel migration scale s:
        Y = X * W = (X * diag(s)^-1) * (diag(s) * W) = X_hat * W_hat

    The scale vector s in R^C is chosen to balance the maximum dynamic ranges:
        s_j = (max(|X_j|)^alpha) / (max(|W_j|)^(1 - alpha))

    where alpha in [0, 1] controls migration strength:
        - alpha = 0.5: Symmetrically divides difficulty between activations and weights.
        - alpha = 1.0: Migrates all difficulty to the weights (activations become easy to quantize).
        - alpha = 0.0: Preserves weights completely (no difficulty migration).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def compute_smoothquant_scale(
    act_max: torch.Tensor,
    weight_max: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute the per-channel SmoothQuant migration scale vector.

    Args:
        act_max: Tensor of shape (C,) representing the maximum absolute activation
            magnitude per input channel across calibration tokens.
        weight_max: Tensor of shape (C,) representing the maximum absolute weight
            magnitude per input channel: max_i |W_{i, j}|.
        alpha: Migration strength hyperparameter in [0.0, 1.0].
        eps: Small epsilon to prevent division by zero or degenerate scales.

    Returns:
        Tensor of shape (C,) containing per-channel migration scales s_j.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    act_max = act_max.clamp(min=eps)
    weight_max = weight_max.clamp(min=eps)

    # s_j = (s_X^alpha) / (s_W^(1 - alpha))
    numerator = act_max.pow(alpha)
    denominator = weight_max.pow(1.0 - alpha)
    scale = (numerator / denominator).clamp(min=eps)
    return scale


def extract_activation_channel_scales(
    module: nn.Module,
    calibration_inputs: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Profile activation outlier channel scales across calibration data.

    Args:
        module: Neural network module (e.g. nn.Linear).
        calibration_inputs: A tensor or list of tensors of shape (B, S, C_in) or (N, C_in).

    Returns:
        Tensor of shape (C_in,) containing per-channel max absolute activation values.
    """
    if isinstance(calibration_inputs, torch.Tensor):
        calibration_inputs = [calibration_inputs]

    channel_max: Optional[torch.Tensor] = None

    for batch in calibration_inputs:
        # Reshape to (N_tokens, C_in)
        flat = batch.detach().view(-1, batch.shape[-1]).abs()
        batch_max = flat.max(dim=0).values
        if channel_max is None:
            channel_max = batch_max
        else:
            channel_max = torch.maximum(channel_max, batch_max)

    if channel_max is None:
        raise ValueError("calibration_inputs cannot be empty.")

    return channel_max


def smooth_linear_weights(
    weight: torch.Tensor,
    act_scales: torch.Tensor,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply SmoothQuant per-channel migration to weight tensor.

    Args:
        weight: Float tensor of shape (out_features, in_features).
        act_scales: Per-channel activation max tensor of shape (in_features,).
        alpha: Migration factor in [0.0, 1.0].

    Returns:
        Tuple of (smoothed_weight, migration_scale):
            - smoothed_weight: W * diag(s) of shape (out_features, in_features).
            - migration_scale: s of shape (in_features,).
    """
    # W shape: (out_features, in_features)
    # Channel max across output rows: max_i |W_{i, j}| -> shape (in_features,)
    weight_max = weight.abs().max(dim=0).values
    scale = compute_smoothquant_scale(act_scales, weight_max, alpha=alpha)

    # W_hat = W * diag(s) -> broadcasting along dim 1
    smoothed_weight = weight * scale.unsqueeze(0)
    return smoothed_weight, scale
