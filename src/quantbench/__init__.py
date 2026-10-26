"""quantbench: Unified Quantization Engine, Bit-Packing Runtime & Quality-vs-Speed Benchmark."""

__version__ = "0.1.0"

from quantbench.quant.affine import (
    dequantize_absmax,
    dequantize_affine,
    quantize_absmax,
    quantize_affine,
)
from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise
from quantbench.quant.fp8 import FP8Format, dequantize_fp8, quantize_fp8

__all__ = [
    "__version__",
    "quantize_affine",
    "dequantize_affine",
    "quantize_absmax",
    "dequantize_absmax",
    "quantize_blockwise",
    "dequantize_blockwise",
    "quantize_fp8",
    "dequantize_fp8",
    "FP8Format",
]
