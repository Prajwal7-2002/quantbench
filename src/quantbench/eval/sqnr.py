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

"""Signal-to-Quantization-Noise Ratio (SQNR) evaluation metrics."""

from __future__ import annotations

import torch


def compute_sqnr(
    original: torch.Tensor,
    quantized: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB).

    Mathematical Formulation:
        SQNR (dB) = 10 * log10( P_signal / P_noise )
                  = 10 * log10( sum(|W|^2) / (sum(|W - W_hat|^2) + eps) )

    Interpretation:
        - Higher SQNR indicates superior numerical fidelity.
        - FP16 -> INT8 typically achieves 40 dB to 55 dB.
        - FP16 -> INT4 blockwise typically achieves 22 dB to 32 dB.
        - Severe breakdown or NaN yields <= 0 dB.

    Args:
        original: Reference unquantized floating-point tensor.
        quantized: Reconstructed/dequantized floating-point tensor of matching shape.
        eps: Numerical stability constant to prevent division by zero for exact equality.

    Returns:
        Float value of SQNR in decibels (dB).
    """
    if original.shape != quantized.shape:
        raise ValueError(
            f"Shape mismatch in SQNR calculation: {original.shape} vs {quantized.shape}"
        )

    orig_f = original.detach().float()
    quant_f = quantized.detach().float()

    signal_power = torch.sum(orig_f.pow(2)).item()
    noise_power = torch.sum((orig_f - quant_f).pow(2)).item()

    if noise_power <= eps:
        return 100.0  # Cap at +100 dB for near-lossless / exact reconstruction

    ratio = signal_power / (noise_power + eps)
    if ratio <= 0.0:
        return 0.0

    sqnr_db = 10.0 * torch.log10(torch.tensor(ratio)).item()
    return float(sqnr_db)
