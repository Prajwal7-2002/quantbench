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

"""GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

Mathematical Foundation:
    Post-training quantization aims to minimize the layer-wise Frobenius reconstruction error:
        argmin_{W_hat} || W * X - W_hat * X ||_F^2

    Using a second-order Taylor expansion around the continuous weights W, the loss
    increase induced by quantizing weights is governed by the input activation Hessian:
        H = 2 * X * X^T in R^{d_in x d_in}

    Optimal Brain Surgeon (OBS) establishes that rounding column j to W_hat_{:, j} produces
    an error delta w_j = W_{:, j} - W_hat_{:, j}. The optimal linear compensation applied to
    the remaining unquantized columns F = {j+1, ..., d_in - 1} is given by:
        Delta W_{:, F} = - (W_{:, j} - W_hat_{:, j}) / [H^-1]_{j, j} * [H^-1]_{j, F}

    To prevent numerical divergence from zero/near-zero eigenvalues, the empirical Hessian
    is regularized with diagonal damping before inversion:
        H_damp = H + lambda * mean(diag(H)) * I    (typically lambda = 0.01)

    GPTQ implements a vectorized block-wise update (block size B, e.g. 32 or 128)
    converting column-by-column corrections into dense matrix multiplications (GEMM)
    while avoiding O(d^3) naive Cholesky recomputations.
"""

from __future__ import annotations

from typing import Optional

import torch

from quantbench.quant.affine import dequantize_absmax, quantize_absmax
from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise


def compute_activation_hessian(
    calibration_inputs: list[torch.Tensor] | torch.Tensor,
    in_features: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute empirical input activation Hessian H = 2 * X * X^T across calibration batches.

    Args:
        calibration_inputs: Tensor of shape (..., in_features) or list of tensors.
        in_features: Number of input features (columns of weight matrix).
        device: Torch device for computation.

    Returns:
        Tensor of shape (in_features, in_features) representing the accumulated Hessian.
    """
    if isinstance(calibration_inputs, torch.Tensor):
        calibration_inputs = [calibration_inputs]

    dev = device or calibration_inputs[0].device
    H = torch.zeros((in_features, in_features), device=dev, dtype=torch.float32)
    n_samples = 0

    for batch in calibration_inputs:
        # Reshape to (N_tokens, in_features)
        x = batch.to(dev).float().view(-1, in_features)
        # H += 2 * x^T * x
        H += 2.0 * (x.t() @ x)
        n_samples += x.shape[0]

    if n_samples == 0:
        raise ValueError("calibration_inputs contains 0 tokens.")

    # Normalize by total sample count
    return H / n_samples


def gptq_quantize_weights(
    weight: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 32,
    damping_factor: float = 0.01,
) -> torch.Tensor:
    """Quantize weight matrix using GPTQ second-order Hessian error compensation.

    Args:
        weight: Float tensor of shape (out_features, in_features).
        H: Input Hessian matrix of shape (in_features, in_features).
        bits: Target integer bit-width (e.g. 4 or 8).
        group_size: Quantization block/group size for scales (-1 for per-channel).
        block_size: GPTQ column block update size (e.g. 32).
        damping_factor: Diagonal damping regularization factor lambda.

    Returns:
        Quantized and dequantized weight tensor in continuous precision with
        second-order error compensation applied.
    """
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype

    # Work in float32 for numerical stability
    W = weight.clone().float()
    H = H.clone().float()

    # Damping regularization: H_reg = H + lambda * mean(diag(H)) * I
    diag_mean = torch.mean(torch.diag(H)).item()
    if diag_mean <= 0.0:
        diag_mean = 1.0
    damp = damping_factor * diag_mean
    H += damp * torch.eye(in_features, device=device)

    # Invert Hessian via Cholesky decomposition: H = L * L^T -> H^-1 = L^-T * L^-1
    try:
        L = torch.linalg.cholesky(H)
        L_inv = torch.linalg.inv(L)
        H_inv = L_inv.t() @ L_inv
    except RuntimeError:
        # Fallback to pseudo-inverse if Cholesky fails due to singular sub-blocks
        H_inv = torch.linalg.pinv(H)

    # Quantization helper for column vectors / sub-blocks
    def quantize_dequantize_cols(col_w: torch.Tensor) -> torch.Tensor:
        # col_w: (out_features, num_cols)
        if group_size > 0:
            q_b, s_b, zp_b, meta = quantize_blockwise(
                col_w, group_size=group_size, num_bits=bits
            )
            return dequantize_blockwise(q_b, s_b, zp_b, meta)
        else:
            q, scale = quantize_absmax(col_w, num_bits=bits, dim=0)
            return dequantize_absmax(q, scale)

    Q = torch.zeros_like(W)

    # Iterate through column blocks
    for col_start in range(0, in_features, block_size):
        col_end = min(col_start + block_size, in_features)
        count = col_end - col_start

        # Sub-block weights and sub-block Hessian inverse
        W_block = W[:, col_start:col_end].clone()
        Q_block = torch.zeros_like(W_block)
        err_block = torch.zeros_like(W_block)

        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        # Column-by-column inside the block
        for i in range(count):
            w_col = W_block[:, i]
            d = H_inv_block[i, i]

            # Quantize individual column
            q_col = quantize_dequantize_cols(w_col.unsqueeze(1)).squeeze(1)
            Q_block[:, i] = q_col

            # OBS column error
            err = (w_col - q_col) / d.clamp(min=1e-7)
            err_block[:, i] = err

            # Propagate error to remaining columns in the current block
            if i + 1 < count:
                W_block[:, i + 1 :] -= err.unsqueeze(1) * H_inv_block[i, i + 1 :].unsqueeze(0)

        Q[:, col_start:col_end] = Q_block

        # Propagate block error to all remaining columns outside the block (GEMM)
        if col_end < in_features:
            # W[:, col_end:] -= err_block @ H_inv[col_start:col_end, col_end:]
            W[:, col_end:] -= err_block @ H_inv[col_start:col_end, col_end:]

    return Q.to(dtype=dtype)
