## Description
<!-- Briefly describe what this PR does, why it is needed, and link any related issues. -->

Fixes #(issue)

## Type of Change
<!-- Please check all options that apply -->
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (new quantization scheme, calibration algorithm, or packing format)
- [ ] ⚡ Performance optimization (vectorization speedup or memory compression)
- [ ] 📚 Documentation update (guides, docstrings, README)
- [ ] 🧪 Tests (coverage expansion or mathematical invariant verification)
- [ ] 💥 Breaking change (fix or feature that breaks existing `QuantLinear` or packing contracts)

## Quantization & Bit-Packing Invariants
<!-- If this PR introduces or modifies a bit-packing format, packing/unpacking roundtrip tests are mandatory. -->
* **Precision Scheme**: (e.g. INT4 / INT8 / BitNet Ternary / FP8 / Ampere 2:4)
* **Packing Ratio**: `____ bits/weight` | `____ x compression vs FP16`
* **SQNR / Perplexity Impact**: `____ dB SQNR` | `____ perplexity on WikiText-2`

## Architectural Invariants Checklist
<!-- You MUST verify and check every item below before submitting -->
- [ ] **Exact Scalar & Tensor Roundtripping**: Packing and unpacking functions tested for bit-exact reconstruction within precision limits.
- [ ] **Pure PyTorch Vectorization**: Implemented with vectorized bitwise shifts/masks (`torch.bitwise_left_shift`, `torch.bitwise_right_shift`). Zero scalar loops over tensor elements.
- [ ] **Unit Tests & Coverage**: Ran `uv run python -m pytest` (100% pass rate, maintained >= 90% coverage).
- [ ] **Formatting & Linting**: Ran `uv run ruff check .` and `uv run ruff format --check .` with zero errors.
- [ ] **Documentation**: Updated `README.md` or docstrings if modifying CLI subcommands or quantization arguments.
- [ ] **Clean Commits**: Follows conventional commit format (`feat:`, `fix:`, `perf:`, `docs:`, `test:`).
