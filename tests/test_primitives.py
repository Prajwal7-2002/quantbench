"""Unit tests for Phase 1: Mathematical Primitives & Invariants."""

import torch
from typer.testing import CliRunner

from quantbench.cli import app
from quantbench.quant.affine import (
    check_quantization_invariants,
    dequantize_absmax,
    dequantize_affine,
    quantize_absmax,
    quantize_affine,
)
from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise
from quantbench.quant.fp8 import FP8Format, dequantize_fp8, quantize_fp8


def test_affine_quantize_dequantize_invariants():
    """Verify uniform affine quantization error bound: |x - x_hat| <= scale / 2."""
    torch.manual_seed(42)
    x = torch.randn(256, 256, dtype=torch.float32) * 5.0 + 2.0

    # 8-bit unsigned
    q, scale, zp = quantize_affine(x, num_bits=8, signed=False)
    assert q.dtype == torch.uint8
    assert q.min() >= 0
    assert q.max() <= 255

    x_hat = dequantize_affine(q, scale, zp)
    assert check_quantization_invariants(x, x_hat, scale, tolerance_factor=0.51)

    # 8-bit signed
    q_s, scale_s, zp_s = quantize_affine(x, num_bits=8, signed=True)
    assert q_s.dtype == torch.int8
    assert q_s.min() >= -128
    assert q_s.max() <= 127

    x_hat_s = dequantize_affine(q_s, scale_s, zp_s)
    assert check_quantization_invariants(x, x_hat_s, scale_s, tolerance_factor=0.51)


def test_affine_per_channel():
    """Verify per-channel scaling calculates independent scales per row."""
    torch.manual_seed(42)
    # Row 0 has small scale, Row 1 has huge scale
    x = torch.zeros(2, 64)
    x[0] = torch.linspace(-1.0, 1.0, 64)
    x[1] = torch.linspace(-100.0, 100.0, 64)

    q, scale, zp = quantize_affine(x, num_bits=8, dim=1)
    assert scale.shape == (2, 1)
    assert zp.shape == (2, 1)
    # Row 1 scale should be approximately 100x larger than Row 0 scale
    assert scale[1].item() > 50.0 * scale[0].item()

    x_hat = dequantize_affine(q, scale, zp)
    err = (x - x_hat).abs()
    assert err.max().item() < 1.0


def test_absmax_symmetry_and_zero():
    """Verify AbsMax maps 0.0 to 0 and preserves symmetric bounds."""
    x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=torch.float32)
    q, scale = quantize_absmax(x, num_bits=8, symmetric_range=True)

    assert q[2].item() == 0  # 0.0 must map to 0
    assert q[0].item() == -q[4].item()  # symmetric bounds
    assert q.dtype == torch.int8

    x_hat = dequantize_absmax(q, scale)
    assert torch.allclose(x, x_hat, atol=scale.item())


def test_constant_tensor_division_by_zero():
    """Verify constant tensors don't produce NaN or zero division errors."""
    x_zeros = torch.zeros(32, 32)
    q_z, s_z, zp_z = quantize_affine(x_zeros)
    assert not torch.isnan(q_z).any()
    assert not torch.isnan(s_z).any()

    x_ones = torch.ones(32, 32) * 42.0
    q_o, s_o, zp_o = quantize_affine(x_ones)
    assert not torch.isnan(q_o).any()
    x_hat_o = dequantize_affine(q_o, s_o, zp_o)
    assert torch.allclose(x_ones, x_hat_o, atol=1e-4)


def test_blockwise_quantization_group_sizes():
    """Verify group-wise quantization across group sizes 32, 64, 128 and padding."""
    torch.manual_seed(42)
    for group_size in [32, 64, 128]:
        # Exact multiple
        x = torch.randn(4, 256)
        q_blocks, scales, zps, meta = quantize_blockwise(
            x,
            group_size=group_size,
            num_bits=4,
            symmetric=True,
        )
        assert q_blocks.shape[1] == group_size
        assert scales.shape[1] == 1
        assert meta["pad_len"] == 0

        recon = dequantize_blockwise(q_blocks, scales, zps, meta)
        assert recon.shape == x.shape
        # MSE should be reasonable for 4-bit blockwise
        mse = torch.mean((x - recon) ** 2).item()
        assert mse < 0.15


def test_blockwise_with_padding():
    """Verify tensor element counts not divisible by group_size are padded and restored."""
    x_odd = torch.randn(7, 13)  # 91 elements, not divisible by 32
    q_b, s_b, zp_b, meta = quantize_blockwise(x_odd, group_size=32, num_bits=4)

    assert meta["n_elements"] == 91
    assert meta["pad_len"] == (32 - (91 % 32))  # 5 elements padded
    assert q_b.numel() == 91 + 5

    recon = dequantize_blockwise(q_b, s_b, zp_b, meta)
    assert recon.shape == (7, 13)
    assert recon.numel() == 91


def test_blockwise_asymmetric():
    """Verify asymmetric blockwise quantization captures positive skewed blocks."""
    x = torch.rand(4, 128) * 10.0 + 5.0  # purely positive [5, 15]
    q_b, s_b, zp_b, meta = quantize_blockwise(
        x,
        group_size=64,
        num_bits=4,
        symmetric=False,
    )
    assert zp_b is not None
    recon = dequantize_blockwise(q_b, s_b, zp_b, meta)
    assert recon.shape == x.shape
    mse = torch.mean((x - recon) ** 2).item()
    assert mse < 0.1


def test_fp8_e4m3_and_e5m2_simulation():
    """Verify FP8 formats, dynamic scaling, and saturation bounds."""
    x = torch.randn(100, 100) * 10.0

    # E4M3
    q_e4m3, s_e4m3 = quantize_fp8(x, format=FP8Format.E4M3)
    recon_e4m3 = dequantize_fp8(q_e4m3, s_e4m3)
    assert recon_e4m3.shape == x.shape
    err_e4m3 = (x - recon_e4m3).abs().mean().item()
    assert err_e4m3 < 0.2

    # E5M2
    q_e5m2, s_e5m2 = quantize_fp8(x, format=FP8Format.E5M2)
    recon_e5m2 = dequantize_fp8(q_e5m2, s_e5m2)
    assert recon_e5m2.shape == x.shape
    err_e5m2 = (x - recon_e5m2).abs().mean().item()
    assert err_e5m2 < 0.4  # E5M2 has fewer mantissa bits than E4M3


def test_fp8_analytical_emulator():
    """Directly test bit-exact FP8 emulator for platforms without native float8."""
    from quantbench.quant.fp8 import _emulate_fp8

    # Test values including zero, subnormals, and large saturation values
    x = torch.tensor([-500.0, -10.0, -0.005, 0.0, 0.005, 10.0, 500.0], dtype=torch.float32)

    # E4M3 (max 448.0)
    q_e4m3 = _emulate_fp8(x, exp_bits=4, mantissa_bits=3, bias=7, max_val=448.0)
    assert q_e4m3[0].item() == -448.0  # saturated at -max
    assert q_e4m3[6].item() == 448.0   # saturated at +max
    assert q_e4m3[3].item() == 0.0     # exact zero

    # E5M2 (max 57344.0)
    q_e5m2 = _emulate_fp8(x, exp_bits=5, mantissa_bits=2, bias=15, max_val=57344.0)
    assert q_e5m2[3].item() == 0.0
    assert torch.allclose(q_e5m2[1], torch.tensor(-10.0), atol=0.5)


def test_cli_runner():
    """Verify CLI basic info, version, and pack-inspect outputs."""
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "quantbench Capabilities" in result.stdout

    result_v = runner.invoke(app, ["--version"])
    assert result_v.exit_code == 0
    assert "0.1.0" in result_v.stdout

    # Test pack-inspect commands
    res_int4 = runner.invoke(app, ["pack-inspect", "--format", "int4"])
    assert res_int4.exit_code == 0
    assert "INT4" in res_int4.stdout

    res_ternary = runner.invoke(app, ["pack-inspect", "--format", "ternary"])
    assert res_ternary.exit_code == 0
    assert "BitNet" in res_ternary.stdout

    res_sparse = runner.invoke(app, ["pack-inspect", "--format", "sparse24"])
    assert res_sparse.exit_code == 0
    assert "2:4" in res_sparse.stdout

    # Test invalid format
    res_err = runner.invoke(app, ["pack-inspect", "--format", "invalid_format"])
    assert res_err.exit_code != 0

