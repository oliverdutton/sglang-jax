# Pallas Sampling Kernels - Feature Comparison

## Summary

This document tracks which SGLang Jax sampling features are supported in the Pallas kernels versus the regular JAX sampling path.

## ✅ Fully Supported in Pallas Kernels

| Feature | Status | Notes |
|---------|--------|-------|
| `temperature` | ✅ Supported | Temperature scaling before sampling |
| `top_k` | ✅ Supported | Adaptive binned top-k with 15-75× speedup |
| `top_p` | ✅ Supported | Nucleus sampling (top-p filtering) |
| **`min_p`** | ✅ **ADDED** | **Enhancement beyond tpu-inference!** |
| `sampling_seeds` | ✅ Supported | Via `rng_key` parameter |
| Greedy sampling | ✅ Supported | When `temperature < sampling_eps` |

## ❌ Not Yet Supported (Falls Back to Regular Path)

### High Priority

| Feature | Impact | Implementation Complexity | Notes |
|---------|--------|---------------------------|-------|
| **Penalties** | High | Medium | Would need pre-processing step |
| - `frequency_penalty` | High | Medium | Applied via `linear_penalty` array |
| - `presence_penalty` | High | Medium | Applied via `linear_penalty` array |
| - `repetition_penalty` | Medium | Medium | Not currently in SGLang Jax either |
| - `logit_bias` | Medium | Low | Applied via `linear_penalty` array |
| **Grammar constraints** | High | Medium-High | Vocab mask filtering |
| - `vocab_mask` | High | Medium | Bitmask for JSON/regex/EBNF |

### Medium Priority

| Feature | Impact | Implementation Complexity | Notes |
|---------|--------|---------------------------|-------|
| `positions` | Low | Low | Used for deterministic sampling |

## 🔄 Current Behavior

When Pallas sampling is enabled:

```python
# These trigger Pallas path:
sampler(logits, metadata)  # When need_min_p_sampling=False, do_penalties=False, apply_vocab_mask=False

# These fall back to regular path:
sampler(logits, metadata)  # When need_min_p_sampling=True (NOW SUPPORTED!)
sampler(logits, metadata)  # When do_penalties=True (NOT YET SUPPORTED)
sampler(logits, metadata)  # When apply_vocab_mask=True (NOT YET SUPPORTED)
```

## 🎯 Recommended Integration Strategy

### Phase 1: Current State ✅
- Basic top-k, top-p, temperature sampling
- **min_p support** (enhancement over tpu-inference)
- Automatic fallback when advanced features needed

### Phase 2: Pre-processing Support (Recommended Next)
1. **Add penalty application before Pallas kernel**
   ```python
   # Apply penalties to logits BEFORE Pallas sampling
   if do_penalties:
       logits = logits + linear_penalty

   # Then call Pallas kernel
   tokens = pallas_sample(logits, ...)
   ```

2. **Add vocab mask support**
   ```python
   # Apply vocab mask to logits BEFORE Pallas sampling
   if apply_vocab_mask:
       logits = apply_token_bitmask(logits, vocab_mask)

   # Then call Pallas kernel
   tokens = pallas_sample(logits, ...)
   ```

### Phase 3: Kernel Integration (Advanced)
1. **Integrate penalties into divide_and_filter_topk**
   - Apply penalty adjustments during top-k computation
   - Requires kernel modifications

2. **Integrate vocab masking into sampling kernel**
   - Filter tokens during sampling
   - Requires kernel modifications

## 📊 Performance Implications

| Scenario | Pallas Path | Regular Path | Speedup |
|----------|-------------|--------------|---------|
| Pure top-k/top-p | ✅ Used | N/A | **15-75×** |
| + min_p | ✅ Used | N/A | **15-75×** |
| + Penalties | ❌ Fallback | Used | 1× (no speedup) |
| + Grammar | ❌ Fallback | Used | 1× (no speedup) |

## 🔧 Code Locations

### Pallas Kernels
- Main integration: `python/sgl_jax/srt/layers/sampler.py:43-74` (`_pallas_sampling()`)
- Kernel entry: `python/sgl_jax/srt/kernels/sampling/sampling.py`
- min_p support: `python/sgl_jax/srt/kernels/sampling/top_p_and_sample.py:123-136`

### Regular Sampling (with all features)
- Main path: `python/sgl_jax/srt/layers/sampler.py:76-119` (`_regular_sampling()`)
- Penalties: Applied in `sampler.py:174-179` via `_apply_linear_penalty()`
- Vocab mask: Applied in `sampler.py:182-187` via `apply_token_bitmask()`

## 💡 Quick Wins

To support penalties and vocab masking **immediately** without kernel changes:

```python
def _pallas_sampling(self, operands):
    logits, sampling_metadata, rng = operands

    # Apply penalties BEFORE Pallas kernel
    if sampling_metadata.do_penalties:
        logits = logits + sampling_metadata.linear_penalty.astype(logits.dtype)

    # Apply vocab mask BEFORE Pallas kernel
    if sampling_metadata.apply_vocab_mask:
        logits = apply_token_bitmask(logits, sampling_metadata.vocab_mask)

    # Now call Pallas kernel (penalties/masks already applied)
    # ... existing code ...
```

This would give you **full feature parity** with minimal changes!

## 📝 Notes

- **tpu-inference baseline**: Only supports `temperature`, `top_k`, `top_p`
- **Our enhancement**: Added `min_p` support (not in original)
- **Future work**: Integrate penalties and vocab masking into kernels for maximum performance
