"""In-place model conversion runtime for PyTorch transformer architectures.

Recursively scans any PyTorch neural network (such as Hugging Face CausalLM models)
and replaces target torch.nn.Linear layers with QuantLinear modules.
"""

from __future__ import annotations

import time
from typing import Any

import torch.nn as nn

from quantbench.nn.quant_linear import QuantLinear


def convert_model_to_quant(
    model: nn.Module,
    target_modules: list[str] | None = None,
    exclude_modules: list[str] | None = None,
    scheme: str = "int4",
    group_size: int = 64,
    symmetric: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """In-place recursive conversion of model linear layers to QuantLinear.

    Args:
        model: PyTorch model (e.g. Hugging Face model or custom transformer).
        target_modules: Optional list of substring names to convert (e.g. ['q_proj', 'v_proj']).
            If None, all linear layers are targeted except those in exclude_modules.
        exclude_modules: Module names to explicitly preserve unquantized (defaults to ['lm_head']).
        scheme: Target quantization scheme ('int4', 'int8', 'ternary', 'sparse24').
        group_size: Group size for blockwise quantization (-1 for per-channel).
        symmetric: Whether to use symmetric quantization.

    Returns:
        Tuple of (converted_model, conversion_statistics_dictionary).
    """
    if exclude_modules is None:
        exclude_modules = ["lm_head"]

    t0 = time.perf_counter()

    # Calculate pre-conversion memory
    orig_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    converted_count = 0
    skipped_count = 0
    layer_names: list[str] = []

    # Helper function to recursively replace layers
    def _replace_layers(parent: nn.Module, prefix: str = ""):
        nonlocal converted_count, skipped_count
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this child is an explicit exclusion
            if any(exc in full_name for exc in exclude_modules):
                skipped_count += 1
                continue

            if isinstance(child, nn.Linear):
                # Check if target matches
                if target_modules is not None:
                    is_target = any(tgt in full_name for tgt in target_modules)
                else:
                    is_target = True

                if is_target:
                    # Convert to QuantLinear
                    quant_child = QuantLinear.from_linear(
                        child,
                        scheme=scheme,
                        group_size=group_size,
                        symmetric=symmetric,
                    )
                    setattr(parent, name, quant_child)
                    converted_count += 1
                    layer_names.append(full_name)
                else:
                    skipped_count += 1
            else:
                _replace_layers(child, full_name)

    _replace_layers(model)

    dt = time.perf_counter() - t0

    # Calculate post-conversion memory (buffers + parameters)
    quant_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    quant_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())

    orig_mb = orig_bytes / (1024 * 1024)
    quant_mb = quant_bytes / (1024 * 1024)
    ratio = orig_mb / max(quant_mb, 1e-6)

    stats = {
        "converted_layers": converted_count,
        "skipped_layers": skipped_count,
        "scheme": scheme,
        "group_size": group_size,
        "orig_memory_mb": round(orig_mb, 2),
        "quant_memory_mb": round(quant_mb, 2),
        "compression_ratio": round(ratio, 2),
        "duration_sec": round(dt, 4),
        "layer_names": layer_names,
    }

    return model, stats
