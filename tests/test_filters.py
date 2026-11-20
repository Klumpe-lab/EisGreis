import numpy as np
import pytest
import scipy.fft as fft

from eisgreis.filters import (
    FILTER_CHOICES,
    LowPassFilter,
    _mask_from_f2,
    contrast_limits,
    gaussian_bandpass_filter,
    resolution_bounds,
)


def reference_lowpass(data: np.ndarray, voxel: float, resolution: float, filter_type: str) -> np.ndarray:
    """The low-pass on the full, centred frequency grid: what the fast filter must equal."""
    axes = np.meshgrid(*[fft.fftshift(np.fft.fftfreq(n, d=voxel)) for n in data.shape], indexing="ij")
    f2 = sum(a**2 for a in axes).astype(np.float32)
    mask = _mask_from_f2(f2, resolution, filter_type)
    spectrum = fft.fftshift(fft.fftn(data.astype(np.float64)))
    return np.real(fft.ifftn(fft.ifftshift(spectrum * mask))).astype(np.float32)


@pytest.mark.parametrize("filter_type", FILTER_CHOICES)
def test_lowpass_matches_full_grid(filter_type):
    data = np.random.default_rng(1).normal(size=(9, 12, 10)).astype(np.float32)
    lowpass = LowPassFilter(data, 5.0, workers=2, cache_spectrum=True)
    np.testing.assert_allclose(
        lowpass.apply(25.0, filter_type), reference_lowpass(data, 5.0, 25.0, filter_type), atol=2e-4
    )


def test_masks_shape():
    c2 = (1 / 20.0) ** 2
    f2 = np.array([0.0, 0.25 * c2, 0.81 * c2, c2 * 1.0001, 100 * c2], dtype=np.float32)
    for filter_type in FILTER_CHOICES:
        mask = _mask_from_f2(f2.copy(), 20.0, filter_type)
        assert mask[0] == pytest.approx(1.0)
        assert mask[-1] == pytest.approx(0.0, abs=1e-3)
    hard = _mask_from_f2(f2.copy(), 20.0, "hard")
    assert list(hard) == [1, 1, 1, 0, 0]
    cosine = _mask_from_f2(f2.copy(), 20.0, "cosine")
    assert cosine[1] == pytest.approx(1.0) and 0 < cosine[2] < 1 and cosine[3] == 0


def test_spectrum_cache():
    data = np.random.default_rng(2).normal(size=(8, 8, 8)).astype(np.float32)
    cached = LowPassFilter(data, 4.0, workers=1, cache_spectrum=True)
    first = cached.apply(16.0, "gaussian")
    assert cached._spectrum is not None
    np.testing.assert_allclose(cached.apply(16.0, "gaussian"), first)
    cached.release()
    assert cached._spectrum is None
    uncached = LowPassFilter(data, 4.0, workers=1, cache_spectrum=False)
    np.testing.assert_allclose(uncached.apply(16.0, "gaussian"), first, atol=1e-5)
    assert uncached._spectrum is None
    cached.set_voxel_size(8.0)
    assert not np.allclose(cached.apply(16.0, "gaussian"), first)


def test_resolution_bounds_and_contrast():
    low, high = resolution_bounds((100, 200, 50), 5.0)
    assert low == 10.0 and high == 1000.0
    data = np.arange(1_000_000, dtype=np.float32).reshape(100, 100, 100)
    lo, hi = contrast_limits(data, max_samples=10_000)
    assert lo < hi and 0 <= lo < 50_000 and hi > 950_000


def test_bandpass():
    image = np.random.default_rng(3).normal(size=(30, 40)).astype(np.float32)
    filtered, mask = gaussian_bandpass_filter(image, 20.0, 0.0, 2.0)
    assert filtered.shape == image.shape and mask.shape == (40, 40)
    assert gaussian_bandpass_filter(image, 50.0, 20.0, 2.0, log=lambda _m: None) is None
