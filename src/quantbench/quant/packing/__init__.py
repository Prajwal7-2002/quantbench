"""Bit-packing runtimes for sub-byte quantization formats.

Provides vectorized bit-shift packing and unpacking for:
- INT4 (4-bit integers: unsigned [0, 15] or signed [-8, 7]) -> 2 values per uint8 byte.
- Ternary (BitNet 1.58b: {-1, 0, +1}) -> 4 values per uint8 byte.
- 2:4 Structured Sparsity Metadata -> two 2-bit indices per 4-bit nibble.
"""

from quantbench.quant.packing.int4_pack import pack_int4, unpack_int4
from quantbench.quant.packing.sparse_pack import (
    decompress_2_4,
    pack_2_4_indices,
    prune_and_compress_2_4,
    unpack_2_4_indices,
)
from quantbench.quant.packing.ternary_pack import pack_ternary, unpack_ternary

__all__ = [
    "pack_int4",
    "unpack_int4",
    "pack_ternary",
    "unpack_ternary",
    "pack_2_4_indices",
    "unpack_2_4_indices",
    "prune_and_compress_2_4",
    "decompress_2_4",
]
