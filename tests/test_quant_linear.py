"""Unit tests for QuantLinear drop-in module and model conversion runtimes."""

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantbench.nn.model_convert import convert_model_to_quant
from quantbench.nn.quant_linear import QuantLinear


class TestQuantLinear:
    """Test suite for QuantLinear drop-in neural layer."""

    def test_int4_blockwise_forward_pass(self) -> None:
        """Verify blockwise INT4 QuantLinear forward pass, output shape, and MSE."""
        torch.manual_seed(42)
        in_f, out_f = 64, 128
        linear = nn.Linear(in_f, out_f, bias=True)

        quant_linear = QuantLinear.from_linear(
            linear, scheme="int4", group_size=32, symmetric=True
        )

        # 128 * 64 = 8192 elements -> 8192 / 32 = 256 blocks of 32 elements
        # 32 elements pack into 16 uint8 bytes
        assert quant_linear.packed_weight.dtype == torch.uint8
        assert quant_linear.packed_weight.shape == (256, 16)

        x = torch.randn(4, in_f)
        out_orig = linear(x)
        out_quant = quant_linear(x)

        assert out_quant.shape == (4, out_f)
        mse = torch.mean((out_orig - out_quant) ** 2).item()
        assert mse < 0.1

    def test_int4_per_channel_forward_pass(self) -> None:
        """Verify per-channel INT4 QuantLinear forward pass and packed shape."""
        torch.manual_seed(42)
        in_f, out_f = 64, 128
        linear = nn.Linear(in_f, out_f, bias=True)

        quant_linear = QuantLinear.from_linear(
            linear, scheme="int4", group_size=-1, symmetric=True
        )

        # Per-channel: (out_f, in_f // 2) = (128, 32)
        assert quant_linear.packed_weight.shape == (out_f, in_f // 2)

        x = torch.randn(4, in_f)
        out_orig = linear(x)
        out_quant = quant_linear(x)

        assert out_quant.shape == (4, out_f)
        mse = torch.mean((out_orig - out_quant) ** 2).item()
        assert mse < 0.15

    def test_int8_forward_pass_low_error(self) -> None:
        """Verify INT8 QuantLinear achieves minimal quantization error."""
        torch.manual_seed(42)
        in_f, out_f = 64, 64
        linear = nn.Linear(in_f, out_f, bias=False)

        quant_linear = QuantLinear.from_linear(linear, scheme="int8", symmetric=True)
        assert quant_linear.packed_weight.dtype == torch.int8

        x = torch.randn(8, in_f)
        out_orig = linear(x)
        out_quant = quant_linear(x)

        mse = torch.mean((out_orig - out_quant) ** 2).item()
        assert mse < 0.005

    def test_ternary_forward_pass(self) -> None:
        """Verify BitNet 1.58b ternary QuantLinear forward execution."""
        torch.manual_seed(42)
        in_f, out_f = 64, 128
        linear = nn.Linear(in_f, out_f, bias=True)

        quant_linear = QuantLinear.from_linear(linear, scheme="ternary")
        # 4 weights per byte -> in_f // 4
        assert quant_linear.packed_weight.shape == (out_f, in_f // 4)

        x = torch.randn(2, in_f)
        out = quant_linear(x)
        assert out.shape == (2, out_f)

    def test_sparse24_forward_pass(self) -> None:
        """Verify 2:4 structured sparsity QuantLinear forward execution."""
        torch.manual_seed(42)
        in_f, out_f = 64, 128
        linear = nn.Linear(in_f, out_f, bias=True)

        quant_linear = QuantLinear.from_linear(linear, scheme="sparse24")
        assert quant_linear.condensed_weight.shape == (out_f, in_f // 2)
        assert quant_linear.packed_metadata.shape == (out_f, in_f // 8)

        x = torch.randn(3, in_f)
        out = quant_linear(x)
        assert out.shape == (3, out_f)

    def test_memory_footprint_calculation(self) -> None:
        """Verify memory_footprint_bytes reports >3x compression factor for INT4."""
        in_f, out_f = 128, 256
        linear = nn.Linear(in_f, out_f)
        quant_linear = QuantLinear.from_linear(linear, scheme="int4", group_size=64)

        stats = quant_linear.memory_footprint_bytes()
        assert stats["compression_ratio"] > 3.0
        assert stats["compressed_bytes"] < stats["fp16_bytes"]

    def test_state_dict_serialization_roundtrip(self) -> None:
        """Verify packed buffers correctly serialize and deserialize with bit-exact outputs."""
        torch.manual_seed(101)
        in_f, out_f = 64, 64
        linear = nn.Linear(in_f, out_f)
        quant_linear = QuantLinear.from_linear(linear, scheme="int4", group_size=32)

        x = torch.randn(2, in_f)
        out_before = quant_linear(x)

        # Save to buffer
        buffer = io.BytesIO()
        torch.save(quant_linear.state_dict(), buffer)
        buffer.seek(0)

        # Create fresh layer and load state_dict
        new_layer = QuantLinear(in_f, out_f, scheme="int4", group_size=32)
        new_layer.load_state_dict(torch.load(buffer, weights_only=True))

        out_after = new_layer(x)
        assert torch.equal(out_before, out_after)

    def test_invalid_scheme_raises(self) -> None:
        """Verify unsupported schemes raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scheme"):
            QuantLinear(32, 32, scheme="unsupported_format")


class TestModelConvert:
    """Test suite for in-place transformer model conversion."""

    def test_convert_transformer_mlp_block(self) -> None:
        """Verify recursive conversion swaps targeted linear layers while respecting exclusions."""
        class MockTransformerMLP(nn.Module):
            def __init__(self, dim: int, hidden_dim: int):
                super().__init__()
                self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
                self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
                self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
                self.lm_head = nn.Linear(dim, 1000, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate = self.gate_proj(x)
                up = self.up_proj(x)
                hidden = F.silu(gate) * up
                out = self.down_proj(hidden)
                return self.lm_head(out)

        torch.manual_seed(42)
        model = MockTransformerMLP(dim=64, hidden_dim=128)
        x = torch.randn(2, 64)
        out_unquant = model(x)
        assert out_unquant.shape == (2, 1000)

        # Convert to INT4, excluding lm_head
        converted_model, stats = convert_model_to_quant(
            model,
            exclude_modules=["lm_head"],
            scheme="int4",
            group_size=32,
        )

        # Verify layer types
        assert isinstance(converted_model.gate_proj, QuantLinear)
        assert isinstance(converted_model.up_proj, QuantLinear)
        assert isinstance(converted_model.down_proj, QuantLinear)
        assert isinstance(converted_model.lm_head, nn.Linear)  # Kept unquantized

        # Verify stats
        assert stats["converted_layers"] == 3
        assert stats["skipped_layers"] == 1
        assert stats["compression_ratio"] > 1.0

        # Run forward pass through converted model
        out_quant = converted_model(x)
        assert out_quant.shape == (2, 1000)
        # Verify cosine similarity is high (>0.95)
        cos_sim = F.cosine_similarity(out_unquant, out_quant, dim=-1).mean().item()
        assert cos_sim > 0.95
