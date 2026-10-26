"""Drop-in quantized neural network modules and model conversion runtimes."""

from quantbench.nn.model_convert import convert_model_to_quant
from quantbench.nn.quant_linear import QuantLinear

__all__ = [
    "QuantLinear",
    "convert_model_to_quant",
]
