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

"""Test suite for evaluation metrics (SQNR, Sliding Perplexity) and Pareto benchmarking."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from quantbench.bench.pareto import ParetoBenchmark
from quantbench.eval.perplexity import (
    compute_sliding_window_perplexity,
    generate_synthetic_corpus,
)
from quantbench.eval.sqnr import compute_sqnr
from quantbench.quant.affine import dequantize_absmax, quantize_absmax


class TestSQNR:
    """Mathematical validation for Signal-to-Quantization-Noise Ratio (dB)."""

    def test_identical_tensors_yield_max_sqnr(self) -> None:
        x = torch.randn(64, 64)
        sqnr = compute_sqnr(x, x)
        assert sqnr == 100.0

    def test_known_signal_noise_ratio(self) -> None:
        # Signal power = 10.0, noise power = 1.0 -> 10 * log10(10 / 1) = 10.0 dB
        orig = torch.tensor([3.0, 1.0])  # sum(sq) = 9 + 1 = 10.0
        # Add noise vector of power 1.0: diff = [1.0, 0.0] -> sum(sq) = 1.0
        quant = torch.tensor([2.0, 1.0])
        sqnr = compute_sqnr(orig, quant)
        assert pytest.approx(sqnr, rel=1e-3) == 10.0

    def test_int8_has_higher_sqnr_than_int4(self) -> None:
        torch.manual_seed(42)
        w = torch.randn(128, 128)

        # INT8
        q8, s8 = quantize_absmax(w, num_bits=8, dim=1)
        w8 = dequantize_absmax(q8, s8)
        sqnr8 = compute_sqnr(w, w8)

        # INT4
        q4, s4 = quantize_absmax(w, num_bits=4, dim=1)
        w4 = dequantize_absmax(q4, s4)
        sqnr4 = compute_sqnr(w, w4)

        assert sqnr8 > sqnr4
        assert sqnr8 > 35.0  # INT8 typically > 35 dB
        assert sqnr4 > 15.0  # INT4 typically > 15 dB

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_sqnr(torch.zeros(10), torch.zeros(12))


class MockCausalLM(nn.Module):
    """Minimal deterministic causal language model for unit testing."""

    def __init__(self, vocab_size: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding(input_ids)
        h = torch.relu(self.q_proj(h))
        h = self.o_proj(h)
        logits = self.lm_head(h)
        return logits


class TestPerplexity:
    """Sliding-window perplexity evaluation tests."""

    def test_sliding_window_perplexity_calculation(self) -> None:
        torch.manual_seed(123)
        model = MockCausalLM(vocab_size=64, hidden_dim=32)
        tokens = generate_synthetic_corpus(vocab_size=64, seq_length=200, seed=42)

        ppl = compute_sliding_window_perplexity(
            model,
            tokens,
            window_size=64,
            stride=32,
        )
        assert ppl > 1.0
        assert not torch.isnan(torch.tensor(ppl))

    def test_stride_validation(self) -> None:
        model = MockCausalLM()
        tokens = torch.randint(0, 10, (50,))
        with pytest.raises(ValueError, match="stride .* must be strictly less"):
            compute_sliding_window_perplexity(model, tokens, window_size=64, stride=64)


class TestParetoBenchmark:
    """Pareto quality-vs-compression analysis tests."""

    def test_full_pareto_sweep_and_ordering(self) -> None:
        torch.manual_seed(42)
        vocab_size = 64
        hidden_dim = 64
        model = MockCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
        tokens = generate_synthetic_corpus(vocab_size=vocab_size, seq_length=150, seed=1)

        bench = ParetoBenchmark(
            base_model=model,
            test_tokens=tokens,
            window_size=64,
            stride=32,
        )

        schemes = [
            {"name": "FP16 Baseline", "scheme": "fp16", "group_size": -1},
            {"name": "INT8 AbsMax", "scheme": "int8", "group_size": -1},
            {"name": "INT4 (G=32)", "scheme": "int4", "group_size": 32},
            {"name": "BitNet Ternary", "scheme": "ternary", "group_size": -1},
            {"name": "Ampere 2:4 Sparse", "scheme": "sparse24", "group_size": -1},
        ]

        points = bench.run_sweep(schemes=schemes, exclude_modules=["lm_head"])
        assert len(points) == 5

        # Check compression ratios
        p_fp16 = next(p for p in points if p.scheme == "fp16")
        p_int8 = next(p for p in points if p.scheme == "int8")
        p_int4 = next(p for p in points if p.scheme == "int4")
        p_ternary = next(p for p in points if p.scheme == "ternary")

        assert p_fp16.compression_ratio == 1.0
        assert p_fp16.compression_ratio < p_int8.compression_ratio
        assert p_int8.compression_ratio < p_int4.compression_ratio
        assert p_int4.compression_ratio < p_ternary.compression_ratio

        # Check table rendering
        table_str = ParetoBenchmark.render_ascii_table(points)
        assert "| Configuration" in table_str
        assert "FP16 Baseline" in table_str
        assert "INT4 (G=32)" in table_str
        assert "BitNet Ternary" in table_str

    def test_cli_eval_and_pareto_commands(self) -> None:
        from typer.testing import CliRunner

        from quantbench.cli import app

        runner = CliRunner()

        # Test eval command
        res_eval = runner.invoke(app, ["eval", "--shape", "64x64"])
        assert res_eval.exit_code == 0
        assert "Quantization Fidelity Benchmark" in res_eval.stdout
        assert "INT8 AbsMax" in res_eval.stdout

        # Test eval invalid shape
        res_eval_err = runner.invoke(app, ["eval", "--shape", "bad_shape"])
        assert res_eval_err.exit_code != 0

        # Test pareto command
        res_pareto = runner.invoke(
            app, ["pareto", "--vocab-size", "32", "--hidden-dim", "32", "--seq-len", "80"]
        )
        assert res_pareto.exit_code == 0
        assert "Pareto Frontier" in res_pareto.stdout
        assert "FP16" in res_pareto.stdout
