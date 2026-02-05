"""Basic integration tests for Pallas sampling kernels.

These tests verify that the sampling kernels can be imported and have the correct
interface. Full functional testing should be done in the tpu-inference repository.
"""

import pytest

# Test imports
def test_imports():
    """Test that sampling kernels can be imported."""
    try:
        from sgl_jax.srt.kernels.sampling import (
            topk_topp_and_sample,
            top_p_and_sample,
            top_k,
            top_bounded_k,
        )
        assert topk_topp_and_sample is not None
        assert top_p_and_sample is not None
        assert top_k is not None
        assert top_bounded_k is not None
    except ImportError as e:
        pytest.skip(f"Pallas sampling not available: {e}")


def test_utils_import():
    """Test that sampling utilities can be imported."""
    try:
        from sgl_jax.srt.kernels.sampling.utils import NUM_LANES, NUM_SUBLANES
        assert NUM_LANES == 128
        assert NUM_SUBLANES == 8
    except ImportError as e:
        pytest.skip(f"Sampling utils not available: {e}")


def test_sampler_pallas_integration():
    """Test that Sampler can use Pallas sampling when enabled."""
    try:
        from sgl_jax.srt.layers.sampler import Sampler, PALLAS_SAMPLING_AVAILABLE

        # Test that the flag is set correctly
        assert isinstance(PALLAS_SAMPLING_AVAILABLE, bool)

        # Test that Sampler can be instantiated with pallas sampling option
        sampler = Sampler(use_pallas_sampling=False)
        assert hasattr(sampler, 'use_pallas_sampling')

    except ImportError as e:
        pytest.skip(f"Sampler integration not available: {e}")


if __name__ == "__main__":
    test_imports()
    test_utils_import()
    test_sampler_pallas_integration()
    print("All basic integration tests passed!")
