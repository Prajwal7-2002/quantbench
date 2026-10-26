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

"""Sliding-window causal language model perplexity evaluation engine.

Mathematical Foundation:
    Perplexity (PPL) is the standard metric for language model evaluation:
        PPL = exp( - (1 / N) * sum_{t=1}^N log P(w_t | w_{<t}) )

    Standard non-overlapping chunk evaluation splits text into discrete blocks of
    length L, discarding past context for the initial tokens of every block and
    artificially inflating measured perplexity.

    The sliding-window approach uses a context window of length L and strides forward
    by S < L tokens:
        - The model sees the full preceding context of length L.
        - Cross-entropy loss is evaluated strictly over the trailing S tokens.
        - Boundary effects are eliminated, guaranteeing statistically rigorous
          comparisons across FP16, INT8, INT4, and sparse model checkpoints.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sliding_window_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    window_size: int = 512,
    stride: int = 256,
    device: Optional[torch.device] = None,
) -> float:
    """Compute exact sliding-window perplexity for a causal language model.

    Args:
        model: Causal language model producing logits of shape (B, S, V).
        input_ids: 1D or 2D tensor of token IDs.
        window_size: Maximum context window size L (e.g. 512 or 1024).
        stride: Stride step S < L between consecutive evaluation windows (e.g. 256).
        device: Device to run evaluation on.

    Returns:
        Perplexity value as a positive float.
    """
    if stride >= window_size:
        raise ValueError(
            f"stride ({stride}) must be strictly less than window_size ({window_size})"
        )

    model.eval()
    dev = device or next(model.parameters()).device

    # Ensure 1D flat token vector
    flat_tokens = input_ids.detach().view(-1).to(dev)
    seq_len = flat_tokens.size(0)

    if seq_len <= 1:
        raise ValueError("input_ids must contain at least 2 tokens for causal language modeling.")

    nlls: list[torch.Tensor] = []
    total_eval_tokens = 0

    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + window_size, seq_len)
        trg_len = end_loc - prev_end_loc  # Number of newly evaluated tokens in this window

        input_chunk = flat_tokens[begin_loc:end_loc].unsqueeze(0)  # (1, chunk_len)
        target_chunk = input_chunk.clone()

        # Mask out context tokens that were already evaluated in previous windows
        target_chunk[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Shift logits and targets for next-token causal prediction:
            # logits[:, :-1, :] predicts targets[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_chunk[:, 1:].contiguous()

            # Cross entropy loss with ignore_index=-100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

        nlls.append(loss)
        # Count non-masked target tokens
        num_valid = (shift_targets != -100).sum().item()
        total_eval_tokens += num_valid

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if total_eval_tokens == 0:
        return float("inf")

    total_nll = torch.stack(nlls).sum().item()
    mean_nll = total_nll / total_eval_tokens
    perplexity = math.exp(mean_nll)

    return float(perplexity)


def generate_synthetic_corpus(
    vocab_size: int = 1000,
    seq_length: int = 1024,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a deterministic synthetic token sequence for offline testing.

    Args:
        vocab_size: Vocabulary cardinality.
        seq_length: Number of tokens to generate.
        seed: Random seed for repeatability.

    Returns:
        Tensor of shape (seq_length,) containing token IDs.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(low=0, high=vocab_size, size=(seq_length,), generator=g)
