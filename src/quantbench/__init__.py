"""quantbench: Unified Quantization Engine, Bit-Packing Runtime & Quality-vs-Speed Benchmark."""

__version__ = "0.1.1"

from quantbench.bench import ParetoBenchmark, ParetoPoint
from quantbench.eval import (
    compute_sliding_window_perplexity,
    compute_sqnr,
    generate_synthetic_corpus,
)
from quantbench.nn.model_convert import convert_model_to_quant
from quantbench.nn.quant_linear import QuantLinear
from quantbench.quant.affine import (
    dequantize_absmax,
    dequantize_affine,
    quantize_absmax,
    quantize_affine,
)
from quantbench.quant.algorithms import (
    awq_quantize_weights,
    compute_activation_channel_means,
    compute_activation_hessian,
    compute_smoothquant_scale,
    extract_activation_channel_scales,
    gptq_quantize_weights,
    search_best_awq_scale,
    smooth_linear_weights,
)
from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise
from quantbench.quant.fp8 import FP8Format, dequantize_fp8, quantize_fp8
from quantbench.quant.packing import (
    decompress_2_4,
    pack_2_4_indices,
    pack_int4,
    pack_ternary,
    prune_and_compress_2_4,
    unpack_2_4_indices,
    unpack_int4,
    unpack_ternary,
)

__all__ = [
    "__version__",
    "QuantLinear",
    "convert_model_to_quant",
    "quantize_affine",
    "dequantize_affine",
    "quantize_absmax",
    "dequantize_absmax",
    "quantize_blockwise",
    "dequantize_blockwise",
    "quantize_fp8",
    "dequantize_fp8",
    "FP8Format",
    "pack_int4",
    "unpack_int4",
    "pack_ternary",
    "unpack_ternary",
    "pack_2_4_indices",
    "unpack_2_4_indices",
    "prune_and_compress_2_4",
    "decompress_2_4",
    "compute_smoothquant_scale",
    "extract_activation_channel_scales",
    "smooth_linear_weights",
    "compute_activation_channel_means",
    "search_best_awq_scale",
    "awq_quantize_weights",
    "compute_activation_hessian",
    "gptq_quantize_weights",
    "compute_sqnr",
    "compute_sliding_window_perplexity",
    "generate_synthetic_corpus",
    "ParetoBenchmark",
    "ParetoPoint",
]
