"""Tests for the feature matching module."""

import numpy as np
import pytest

from symbac.imaging.feature_matching import (
    ImageFeatures,
    feature_distance,
    compare_images,
    OptimizationBounds,
    _normalize_image,
    _radial_power_spectrum,
    _spectral_slope,
    _compute_glcm_features,
    _region_statistics,
    _skewness,
    _kurtosis,
)


# ---- Test fixtures ----

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def uniform_image():
    """A uniform gray image."""
    return np.full((64, 64), 0.5, dtype=np.float64)


@pytest.fixture
def noisy_image(rng):
    """An image with Gaussian noise."""
    return rng.normal(0.5, 0.1, size=(64, 64)).clip(0, 1)


@pytest.fixture
def gradient_image():
    """A horizontal gradient image."""
    return np.tile(np.linspace(0, 1, 64), (64, 1))


@pytest.fixture
def synthetic_pair(rng):
    """A pair of slightly different images (simulates real vs synthetic)."""
    base = rng.normal(0.5, 0.15, size=(128, 48)).clip(0, 1)
    # Add a gradient and slight blur to simulate differences
    gradient = np.linspace(0.9, 1.1, 128)[:, np.newaxis]
    shifted = base * gradient
    shifted = shifted.clip(0, 1)
    return base, shifted


@pytest.fixture
def image_with_masks(rng):
    """An image with instance masks and device mask."""
    img = rng.normal(0.5, 0.1, size=(64, 64)).clip(0, 1)
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[20:30, 20:44] = 1  # cell 1
    masks[35:45, 20:44] = 2  # cell 2
    device = np.zeros((64, 64), dtype=np.uint8)
    device[:, :15] = 1  # device walls
    device[:, 49:] = 1
    return img, masks, device


# ---- Helper tests ----

class TestNormalize:
    def test_already_normalized(self):
        img = np.array([[0.0, 0.5], [0.5, 1.0]])
        result = _normalize_image(img)
        np.testing.assert_allclose(result, img)

    def test_uint8(self):
        img = np.array([[0, 128], [128, 255]], dtype=np.uint8)
        result = _normalize_image(img)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_constant_image(self):
        img = np.full((10, 10), 42.0)
        result = _normalize_image(img)
        assert result.min() == result.max() == 0.0

    def test_3d_to_2d(self):
        img = np.random.rand(10, 10, 3)
        result = _normalize_image(img)
        assert result.ndim == 2


class TestStatistics:
    def test_skewness_symmetric(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        assert abs(_skewness(x)) < 0.1

    def test_skewness_positive(self):
        rng = np.random.default_rng(0)
        x = rng.exponential(1.0, size=10000)
        assert _skewness(x) > 0.5

    def test_kurtosis_normal(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=50000)
        assert abs(_kurtosis(x)) < 0.2  # excess kurtosis ≈ 0 for normal

    def test_constant_input(self):
        x = np.ones(100)
        assert _skewness(x) == 0.0
        assert _kurtosis(x) == 0.0


class TestRadialSpectrum:
    def test_output_shape(self, noisy_image):
        spec = _radial_power_spectrum(noisy_image, n_bins=32)
        assert spec.shape == (32,)

    def test_all_positive(self, noisy_image):
        spec = _radial_power_spectrum(noisy_image, n_bins=32)
        assert np.all(spec >= 0)

    def test_uniform_image_dc_dominant(self, uniform_image):
        spec = _radial_power_spectrum(uniform_image, n_bins=16)
        # DC component should dominate for uniform image
        assert spec[0] > spec[-1]


class TestSpectralSlope:
    def test_noisy_has_slope(self, noisy_image):
        spec = _radial_power_spectrum(noisy_image, n_bins=32)
        slope, intercept = _spectral_slope(spec)
        # Natural-like images have negative slope in log-log
        assert isinstance(slope, float)
        assert isinstance(intercept, float)

    def test_zeros(self):
        slope, intercept = _spectral_slope(np.zeros(10))
        assert slope == 0.0
        assert intercept == 0.0


class TestGLCM:
    def test_uniform_image(self, uniform_image):
        props = _compute_glcm_features(uniform_image, levels=16)
        assert props["contrast"] == 0.0  # No texture variation
        assert props["energy"] > 0

    def test_noisy_has_texture(self, noisy_image):
        props = _compute_glcm_features(noisy_image, levels=16)
        assert props["contrast"] > 0  # Noise creates texture
        assert 0 <= props["homogeneity"] <= 1

    def test_gradient_has_structure(self, gradient_image):
        props = _compute_glcm_features(gradient_image, levels=16)
        assert props["contrast"] > 0

    def test_all_keys_present(self, noisy_image):
        props = _compute_glcm_features(noisy_image, levels=16)
        for key in ["contrast", "correlation", "energy", "homogeneity"]:
            assert key in props


class TestRegionStatistics:
    def test_with_masks(self, image_with_masks):
        img, masks, device = image_with_masks
        means, stds = _region_statistics(img, masks, device)
        assert "media" in means
        assert "cell" in means
        assert "device" in means
        for v in means.values():
            assert 0 <= v <= 1
        for v in stds.values():
            assert v >= 0

    def test_no_masks_auto(self, noisy_image):
        means, stds = _region_statistics(noisy_image, None, None)
        # Should auto-segment; may or may not find 3 regions
        assert isinstance(means, dict)
        assert isinstance(stds, dict)


# ---- ImageFeatures tests ----

class TestImageFeatures:
    def test_extract_basic(self, noisy_image):
        feats = ImageFeatures.extract(noisy_image)
        assert 0 <= feats.intensity_mean <= 1
        assert feats.intensity_std > 0
        assert len(feats.radial_spectrum) > 0
        assert feats.gradient_mean > 0

    def test_extract_with_masks(self, image_with_masks):
        img, masks, device = image_with_masks
        feats = ImageFeatures.extract(img, masks=masks, device_mask=device)
        assert "cell" in feats.region_means
        assert "media" in feats.region_means
        assert "device" in feats.region_means

    def test_to_vector(self, noisy_image):
        feats = ImageFeatures.extract(noisy_image)
        vec = feats.to_vector()
        assert vec.ndim == 1
        assert len(vec) > 19  # 19 scalars + spectrum bins
        assert not np.any(np.isnan(vec))

    def test_to_vector_no_spectrum(self, noisy_image):
        feats = ImageFeatures.extract(noisy_image)
        vec = feats.to_vector(include_spectrum=False)
        assert len(vec) == 19

    def test_uniform_low_variation(self, uniform_image):
        feats = ImageFeatures.extract(uniform_image)
        assert feats.intensity_std == 0.0
        assert feats.edge_density == 0.0
        assert feats.gradient_mean == 0.0


# ---- feature_distance tests ----

class TestFeatureDistance:
    def test_identical_images(self, noisy_image):
        feats = ImageFeatures.extract(noisy_image)
        dist = feature_distance(feats, feats)
        assert dist == 0.0

    def test_different_images(self, synthetic_pair):
        img_a, img_b = synthetic_pair
        feats_a = ImageFeatures.extract(img_a)
        feats_b = ImageFeatures.extract(img_b)
        dist = feature_distance(feats_a, feats_b)
        assert dist > 0

    def test_symmetric(self, synthetic_pair):
        img_a, img_b = synthetic_pair
        feats_a = ImageFeatures.extract(img_a)
        feats_b = ImageFeatures.extract(img_b)
        dist_ab = feature_distance(feats_a, feats_b)
        dist_ba = feature_distance(feats_b, feats_a)
        np.testing.assert_allclose(dist_ab, dist_ba, atol=1e-10)

    def test_custom_weights(self, synthetic_pair):
        img_a, img_b = synthetic_pair
        feats_a = ImageFeatures.extract(img_a)
        feats_b = ImageFeatures.extract(img_b)

        # All-zero weights → zero distance
        zero_weights = {k: 0.0 for k in [
            "intensity_mean", "intensity_std", "intensity_skew",
            "intensity_kurtosis", "intensity_p5", "intensity_p25",
            "intensity_p50", "intensity_p75", "intensity_p95",
            "spectral_slope", "spectral_intercept",
            "glcm_contrast", "glcm_correlation", "glcm_energy",
            "glcm_homogeneity", "edge_density", "gradient_mean",
            "gradient_std", "laplacian_var", "radial_spectrum",
        ]}
        dist = feature_distance(feats_a, feats_b, weights=zero_weights)
        assert dist == 0.0

    def test_higher_noise_larger_distance(self):
        # Use a larger image and fixed seed for stable statistics
        rng = np.random.default_rng(123)
        base = rng.normal(0.5, 0.1, size=(128, 128)).clip(0, 1)
        noisy_small = (base + rng.normal(0, 0.005, size=(128, 128))).clip(0, 1)
        noisy_large = (base + rng.normal(0, 0.15, size=(128, 128))).clip(0, 1)

        feats_base = ImageFeatures.extract(base)
        feats_small = ImageFeatures.extract(noisy_small)
        feats_large = ImageFeatures.extract(noisy_large)

        dist_small = feature_distance(feats_base, feats_small)
        dist_large = feature_distance(feats_base, feats_large)
        assert dist_large > dist_small


# ---- compare_images tests ----

class TestCompareImages:
    def test_basic(self, synthetic_pair):
        img_a, img_b = synthetic_pair
        report = compare_images(img_a, img_b)
        assert "overall_distance" in report
        assert report["overall_distance"] > 0
        assert "intensity" in report
        assert "texture" in report
        assert "edges" in report
        assert "frequency" in report

    def test_identical_low_distance(self, noisy_image):
        report = compare_images(noisy_image, noisy_image)
        assert report["overall_distance"] == 0.0

    def test_ssim_present(self, synthetic_pair):
        img_a, img_b = synthetic_pair
        report = compare_images(img_a, img_b)
        assert "ssim" in report
        assert 0 <= report["ssim"] <= 1


# ---- OptimizationBounds tests ----

class TestOptimizationBounds:
    def test_defaults(self):
        bounds = OptimizationBounds()
        assert bounds.media_multiplier[0] < bounds.media_multiplier[1]
        assert bounds.cell_multiplier[0] < bounds.cell_multiplier[1]
        assert bounds.device_multiplier[0] < bounds.device_multiplier[1]

    def test_custom(self):
        bounds = OptimizationBounds(
            media_multiplier=(10.0, 50.0),
            defocus=(0.5, 5.0),
        )
        assert bounds.media_multiplier == (10.0, 50.0)
        assert bounds.defocus == (0.5, 5.0)
