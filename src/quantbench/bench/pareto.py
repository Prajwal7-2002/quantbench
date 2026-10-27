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

"""Pareto evaluation engine: Quality vs Compression trade-off analysis."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from quantbench.eval.perplexity import compute_sliding_window_perplexity
from quantbench.eval.sqnr import compute_sqnr
from quantbench.nn.model_convert import convert_model_to_quant
from quantbench.nn.quant_linear import QuantLinear


@dataclass
class ParetoPoint:
    """Single configuration measurement on the Pareto frontier."""

    name: str
    scheme: str
    group_size: int
    compression_ratio: float
    memory_mb: float
    perplexity: float
    perplexity_delta: float
    avg_sqnr_db: float


class ParetoBenchmark:
    """Benchmark model configurations across precision, memory, and perplexity frontiers."""

    def __init__(
        self,
        base_model: nn.Module,
        test_tokens: torch.Tensor,
        window_size: int = 256,
        stride: int = 128,
    ) -> None:
        self.base_model = base_model
        self.test_tokens = test_tokens
        self.window_size = window_size
        self.stride = stride

    def run_sweep(
        self,
        schemes: Optional[List[Dict[str, Any]]] = None,
        exclude_modules: Optional[List[str]] = None,
    ) -> List[ParetoPoint]:
        """Run Pareto sweep across specified quantization schemes."""
        if schemes is None:
            schemes = [
                {"name": "FP16 Baseline", "scheme": "fp16", "group_size": -1},
                {"name": "INT8 AbsMax", "scheme": "int8", "group_size": -1},
                {"name": "INT4 (G=128)", "scheme": "int4", "group_size": 128},
                {"name": "INT4 (G=64)", "scheme": "int4", "group_size": 64},
                {"name": "BitNet Ternary", "scheme": "ternary", "group_size": -1},
                {"name": "Ampere 2:4 Sparse", "scheme": "sparse24", "group_size": -1},
            ]

        # 1. Evaluate baseline FP16
        base_ppl = compute_sliding_window_perplexity(
            self.base_model,
            self.test_tokens,
            window_size=self.window_size,
            stride=self.stride,
        )

        base_bytes = sum(p.numel() * 2 for p in self.base_model.parameters())
        base_mb = base_bytes / (1024 * 1024)

        points: List[ParetoPoint] = []

        for cfg in schemes:
            name = cfg["name"]
            scheme = cfg["scheme"]
            group_size = cfg.get("group_size", 64)

            if scheme == "fp16":
                points.append(
                    ParetoPoint(
                        name=name,
                        scheme=scheme,
                        group_size=group_size,
                        compression_ratio=1.0,
                        memory_mb=round(base_mb, 2),
                        perplexity=round(base_ppl, 3),
                        perplexity_delta=0.0,
                        avg_sqnr_db=100.0,
                    )
                )
                continue

            # Clone base model for isolated in-place conversion
            cloned_model = copy.deepcopy(self.base_model)

            # Extract unquantized linear weights for layer-level SQNR comparison
            orig_weights: Dict[str, torch.Tensor] = {}
            for m_name, module in cloned_model.named_modules():
                if isinstance(module, nn.Linear):
                    orig_weights[m_name] = module.weight.detach().clone()

            # In-place quantization
            _, stats = convert_model_to_quant(
                cloned_model,
                scheme=scheme,
                group_size=group_size,
                exclude_modules=exclude_modules,
            )

            # Compute layer-level SQNR
            sqnrs: List[float] = []
            for m_name, module in cloned_model.named_modules():
                if isinstance(module, QuantLinear) and m_name in orig_weights:
                    dequant_w = module.dequantize_weight()
                    sqnrs.append(compute_sqnr(orig_weights[m_name], dequant_w))

            avg_sqnr = sum(sqnrs) / len(sqnrs) if sqnrs else 0.0

            # Evaluate quantized perplexity
            q_ppl = compute_sliding_window_perplexity(
                cloned_model,
                self.test_tokens,
                window_size=self.window_size,
                stride=self.stride,
            )

            points.append(
                ParetoPoint(
                    name=name,
                    scheme=scheme,
                    group_size=group_size,
                    compression_ratio=round(stats["compression_ratio"], 2),
                    memory_mb=round(stats["quant_memory_mb"], 2),
                    perplexity=round(q_ppl, 3),
                    perplexity_delta=round(q_ppl - base_ppl, 3),
                    avg_sqnr_db=round(avg_sqnr, 1),
                )
            )

        return points

    @staticmethod
    def render_ascii_table(points: List[ParetoPoint]) -> str:
        """Render a clean GitHub Markdown / terminal ASCII table of the Pareto frontier."""
        lines = [
            "| Configuration | Scheme | Group | Memory (MB) | Compression "
            "| Perplexity | Delta PPL | SQNR (dB) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for p in points:
            grp = str(p.group_size) if p.group_size > 0 else "None"
            delta = (
                f"+{p.perplexity_delta:.3f}"
                if p.perplexity_delta > 0
                else f"{p.perplexity_delta:.3f}"
            )
            row = (
                f"| {p.name:<18} | {p.scheme:<8} | {grp:<5} | {p.memory_mb:>10.2f} | "
                f"{p.compression_ratio:>10.2f}x | {p.perplexity:>10.3f} | {delta:>9} | "
                f"{p.avg_sqnr_db:>8.1f} |"
            )
            lines.append(row)
        return "\n".join(lines)
