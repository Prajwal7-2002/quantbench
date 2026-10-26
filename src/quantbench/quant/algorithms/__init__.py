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

"""Advanced calibration algorithms for post-training quantization."""

from quantbench.quant.algorithms.awq import (
    awq_quantize_weights,
    compute_activation_channel_means,
    search_best_awq_scale,
)
from quantbench.quant.algorithms.gptq import (
    compute_activation_hessian,
    gptq_quantize_weights,
)
from quantbench.quant.algorithms.smoothquant import (
    compute_smoothquant_scale,
    extract_activation_channel_scales,
    smooth_linear_weights,
)

__all__ = [
    "compute_smoothquant_scale",
    "extract_activation_channel_scales",
    "smooth_linear_weights",
    "compute_activation_channel_means",
    "search_best_awq_scale",
    "awq_quantize_weights",
    "compute_activation_hessian",
    "gptq_quantize_weights",
]
