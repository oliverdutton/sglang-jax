# Pallas Sampling Kernels Integration

This directory contains high-performance Pallas-based sampling kernels for TPU inference, ported from the [tpu-inference](https://github.com/oliverdutton/tpu-inference) repository's `pallas_sampling` branch.

## Overview

The Tallax sampling implementation provides significant performance improvements for token sampling on TPU hardware:
- **15-75× faster** than standard sampling approaches
- Optimized divide-and-filter top-k algorithm
- Fused top-p filtering and categorical sampling
- **NEW:** min_p support added to the sampling kernels

## Components

### Core Algorithms

1. **divide_and_filter_topk.py** - Adaptive binned top-k computation
   - Uses progressive refinement with early convergence detection
   - 15× speedup on small batches, 75× on large batches
   - Handles variable per-token k values

2. **top_p_and_sample.py** - Fused sampling kernel
   - Combines temperature scaling, top-p filtering, and sampling
   - **Enhanced with min_p filtering support**
   - Optimized for TPU memory hierarchy

3. **bitonic_topk.py** - Bitonic sorting for top-k
   - TPU-optimized sorting using compressed transpose format
   - Reduces lane permutation overhead

### Supporting Utilities

4. **cumsum.py** - TPU-optimized cumulative sum for top-p
5. **sparse_random.py** - Efficient sparse random number generation
6. **gather.py** - Optimized take_along_axis operations
7. **utils.py** - Common TPU utilities and constants
8. **topk_convergence_theory.py** - Convergence probability calculations
9. **sampling.py** - Main entry point combining top-k and top-p

## Usage

### Basic Usage

```python
from sgl_jax.srt.kernels.sampling import topk_topp_and_sample

# Sample with Pallas kernels
next_tokens = topk_topp_and_sample(
    rng_key,
    logits,
    sampling_metadata,
    max_k=64,
    sampling_eps=1e-6,
    replace_val=-1e12,
)
```

### Integration with Sampler

The Pallas kernels are automatically used when:
1. Running on TPU hardware
2. `SGLANG_USE_PALLAS_SAMPLING=true` environment variable is set
3. Or explicitly enabled: `Sampler(use_pallas_sampling=True)`

```python
from sgl_jax.srt.layers.sampler import Sampler

# Enable Pallas sampling
sampler = Sampler(use_pallas_sampling=True)

# Or via environment variable
# export SGLANG_USE_PALLAS_SAMPLING=true
sampler = Sampler()  # Auto-detects based on env var
```

### min_p Support

The `top_p_and_sample` kernel now supports min_p filtering:

```python
from sgl_jax.srt.kernels.sampling import top_p_and_sample

next_tokens = top_p_and_sample(
    topk_logits,
    topk_idx,
    rng_key,
    top_p=top_p_values,
    temperature=temperatures,
    min_p=min_p_values,  # NEW: min_p support
    vocab_size=vocab_size,
    replace_val=-1e12,
    sampling_eps=1e-6,
)
```

## Configuration

Enable Pallas sampling with environment variable:
```bash
export SGLANG_USE_PALLAS_SAMPLING=true
```

Configure top-k threshold for auto-switching:
```bash
export PALLAS_SAMPLING_TOPK_THRESHOLD=64
```

## Performance

Benchmarked on TPU v5e with Gemini 3 Pro settings (top-k=64, top-p=0.95, 262K vocab):
- **Small batches (16 tokens):** 15× average speedup (10× minimum)
- **Large batches (128 tokens):** 75× average speedup (45× minimum)

## Differences from tpu-inference

1. **Import paths:** Changed from `tpu_inference.kernels.sampling` to `sgl_jax.srt.kernels.sampling`
2. **min_p support:** Added min_p filtering to `top_p_and_sample.py`
3. **Integration:** Seamlessly integrated into existing `Sampler` class with fallback support

## Testing

Basic integration tests are available in:
```
python/sgl_jax/test/kernels/sampling/test_pallas_sampling.py
```

Run tests:
```bash
pytest python/sgl_jax/test/kernels/sampling/
```

## References

- Original implementation: [tpu-inference/pallas_sampling](https://github.com/oliverdutton/tpu-inference/tree/pallas_sampling)
- Tallax sampling documentation: See README.md in this directory
- JAX Pallas documentation: https://jax.readthedocs.io/en/latest/pallas.html

## Future Work

- Full min_p integration into `topk_topp_and_sample` main entry point
- Additional sampling strategies (e.g., typical-p, tail-free sampling)
- Performance benchmarking on different TPU generations
- Extended test coverage for edge cases
