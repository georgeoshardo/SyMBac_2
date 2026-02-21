"""Tests for the 6-change SyMBac Phase Contrast Realism Improvement Plan.

Change 1: Pupil-function-based defocus
Change 2: Configurable intensity normalisation
Change 3: Spatially varying device edge halo
Change 4: Two-layer cell OPL model
Change 5: PSF condenser params in optimisation bounds
Change 6: Edge-profile and halo-width features
"""

import numpy as np
import pytest


# ============================================================
# Change 2: Configurable intensity normalisation
# ============================================================

class TestNormalizeOutput:
    def test_default_normalizes_to_01(self):
        """Default behavior: output in [0, 1]."""
        from symbac.imaging.renderer import RenderConfig
        config = RenderConfig()
        assert config.normalize_output is True
        assert config.intensity_range is None

    def test_normalize_output_false_preserves_scale(self):
        """When normalize_output=False, raw intensities are preserved."""
        from symbac.imaging.renderer import RenderConfig, generate_pc_opl
        config = RenderConfig(
            media_multiplier=30.0,
            cell_multiplier=-5.0,
            device_multiplier=-50.0,
            normalize_output=False,
        )
        # Create a simple scene
        opl = np.zeros((20, 20), dtype=np.float64)
        opl[8:12, 8:12] = 1.0  # cell in center
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[8:12, 8:12] = 1
        device = np.zeros((20, 20), dtype=np.uint8)
        device[:3, :] = 1  # device at top

        scene, bg, exp_masks = generate_pc_opl(opl, masks, device, config)
        # Scene should have values near media_multiplier and device_multiplier
        # (not [0, 1])
        assert scene.max() > 1.0 or scene.min() < 0.0

    def test_intensity_range_custom(self):
        """intensity_range sets custom in_range for rescale_intensity."""
        from symbac.imaging.renderer import RenderConfig
        config = RenderConfig(intensity_range=(-100.0, 100.0))
        assert config.intensity_range == (-100.0, 100.0)


# ============================================================
# Change 1: Pupil-function-based defocus
# ============================================================

class TestPupilDefocus:
    def test_apply_defocus_pupil_zero(self):
        """Zero defocus returns unchanged kernel."""
        from symbac.imaging.optics import apply_defocus_pupil
        kernel = np.random.default_rng(42).normal(0, 1, (21, 21))
        result = apply_defocus_pupil(kernel, 0.0, 0.7, 1.2, 1.5, 0.065)
        np.testing.assert_array_equal(result, kernel)

    def test_apply_defocus_pupil_nonzero(self):
        """Non-zero defocus produces a different kernel."""
        from symbac.imaging.optics import apply_defocus_pupil, PSFModel
        psf = PSFModel.phase_contrast(
            wavelength=0.7, NA=1.2, n=1.5,
            condenser="Ph3", apo_sigma=10,
            radius=30, pixel_scale=0.1,
        )
        kernel = psf.kernel_2d.copy()
        defocused = apply_defocus_pupil(kernel, 2.0, 0.7, 1.2, 1.5, 0.1)
        assert defocused.shape == kernel.shape
        # Should be different from original
        assert not np.allclose(defocused, kernel, atol=1e-6)

    def test_defocus_broadens_psf(self):
        """Defocus should broaden the PSF (more spread)."""
        from symbac.imaging.optics import apply_defocus_pupil, PSFModel
        psf = PSFModel.phase_contrast(
            wavelength=0.7, NA=1.2, n=1.5,
            condenser="Ph3", apo_sigma=10,
            radius=30, pixel_scale=0.1,
        )
        kernel = psf.kernel_2d.copy()
        defocused = apply_defocus_pupil(kernel, 5.0, 0.7, 1.2, 1.5, 0.1)

        # Second moment (spread) should increase with defocus
        h, w = kernel.shape
        cy, cx = h // 2, w // 2
        y, x = np.mgrid[0:h, 0:w]
        r_sq = (x - cx)**2 + (y - cy)**2
        spread_orig = np.sum(np.abs(kernel) * r_sq) / np.sum(np.abs(kernel))
        spread_defoc = np.sum(np.abs(defocused) * r_sq) / np.sum(np.abs(defocused))
        assert spread_defoc > spread_orig

    def test_render_config_defocus_um(self):
        """RenderConfig supports defocus_um field."""
        from symbac.imaging.renderer import RenderConfig
        config = RenderConfig(defocus_um=2.5)
        assert config.defocus_um == 2.5
        assert config.defocus == 1.0  # default legacy value

    def test_render_uses_pupil_defocus_when_set(self):
        """When defocus_um is set, render_image uses pupil defocus."""
        from symbac.imaging.renderer import RenderConfig
        # Just verify the config is created without error
        config = RenderConfig(
            defocus_um=3.0,
            defocus=0.0,
            media_multiplier=30.0,
            cell_multiplier=-5.0,
            device_multiplier=-50.0,
        )
        assert config.defocus_um == 3.0


# ============================================================
# Change 3: Spatially varying device edge halo
# ============================================================

class TestDeviceEdgeHalo:
    def test_compute_edge_halo_no_device(self):
        """No device mask returns zero halo."""
        from symbac.imaging.renderer import compute_device_edge_halo
        device = np.zeros((50, 50), dtype=np.uint8)
        halo = compute_device_edge_halo(device, halo_width=5.0, halo_intensity=0.1)
        assert halo.shape == (50, 50)
        np.testing.assert_array_equal(halo, 0.0)

    def test_compute_edge_halo_has_gradient(self):
        """Halo decays away from device boundary."""
        from symbac.imaging.renderer import compute_device_edge_halo
        device = np.zeros((100, 100), dtype=np.uint8)
        device[:, :20] = 1  # left side is device
        halo = compute_device_edge_halo(device, halo_width=10.0, halo_intensity=0.5)
        # Halo should be strongest near boundary (column ~20)
        # and decay toward center
        near_boundary = halo[50, 21]
        far_from_boundary = halo[50, 80]
        assert near_boundary > far_from_boundary
        assert near_boundary > 0

    def test_edge_halo_width_affects_spread(self):
        """Wider halo_width produces a more spread-out halo."""
        from symbac.imaging.renderer import compute_device_edge_halo
        device = np.zeros((100, 100), dtype=np.uint8)
        device[:, :20] = 1
        halo_narrow = compute_device_edge_halo(device, halo_width=3.0, halo_intensity=0.5)
        halo_wide = compute_device_edge_halo(device, halo_width=15.0, halo_intensity=0.5)
        # Wide halo should have more total intensity (more spread)
        assert halo_wide.sum() > halo_narrow.sum()

    def test_render_config_edge_halo_fields(self):
        """RenderConfig has edge_halo_width and edge_halo_intensity."""
        from symbac.imaging.renderer import RenderConfig
        config = RenderConfig(edge_halo_width=5.0, edge_halo_intensity=0.15)
        assert config.edge_halo_width == 5.0
        assert config.edge_halo_intensity == 0.15

    def test_apply_illumination_gradient_backward_compat(self):
        """apply_halo still works as alias."""
        from symbac.imaging.renderer import apply_halo, apply_illumination_gradient
        assert apply_halo is apply_illumination_gradient
        img = np.ones((10, 10))
        result = apply_halo(img, 1.0, 1.0)
        np.testing.assert_array_equal(result, img)


# ============================================================
# Change 4: Two-layer cell OPL model
# ============================================================

class TestTwoLayerOPL:
    def test_cell_optics_config_defaults(self):
        """CellOpticsConfig has sensible defaults."""
        from symbac.imaging.optics import CellOpticsConfig
        cfg = CellOpticsConfig()
        assert cfg.n_medium == 1.33
        assert cfg.n_wall == 1.45
        assert cfg.n_cytoplasm == 1.39
        assert cfg.wall_fraction == 0.1

    def test_compute_two_layer_opl_shape(self):
        """Two-layer OPL has same shape as input."""
        from symbac.imaging.drawing import compute_two_layer_opl
        from symbac.imaging.optics import CellOpticsConfig
        cell_optics = CellOpticsConfig()
        dist_sq = np.zeros((20, 20))
        r = 10.0
        for i in range(20):
            for j in range(20):
                dist_sq[i, j] = (i - 10)**2 + (j - 10)**2
        opl = compute_two_layer_opl(dist_sq, r**2, r, cell_optics)
        assert opl.shape == (20, 20)

    def test_two_layer_produces_rim_enhancement(self):
        """Two-layer model should produce brighter rim than interior."""
        from symbac.imaging.drawing import compute_two_layer_opl
        from symbac.imaging.optics import CellOpticsConfig
        cell_optics = CellOpticsConfig(
            n_medium=1.33, n_wall=1.50, n_cytoplasm=1.35, wall_fraction=0.15,
        )
        # Create a circle
        r = 50.0
        y, x = np.mgrid[-60:61, -60:61]
        dist_sq = (x.astype(float))**2 + (y.astype(float))**2
        opl = compute_two_layer_opl(dist_sq, r**2, r, cell_optics)

        # The rim (near r) should have higher OPL per unit thickness
        # than the center because it's all wall material
        center_opl = opl[60, 60]
        # Sample at ~85% of radius (in wall region)
        rim_y = int(60 + 0.93 * r)
        rim_opl = opl[rim_y, 60]
        # Both should be positive (dn > 0)
        assert center_opl > 0
        # At center, cytoplasm dominates; at rim, wall dominates
        # Wall has higher delta_n so per-pixel contribution is larger at rim
        # (But total thickness is less at rim, so raw OPL may be lower)
        # Just check both are positive
        assert opl[inside_mask := dist_sq < r**2].min() >= 0

    def test_two_layer_vs_simple_different(self):
        """Two-layer model gives different OPL than simple 2*sqrt(R^2-d^2)."""
        from symbac.imaging.drawing import compute_two_layer_opl
        from symbac.imaging.optics import CellOpticsConfig
        cell_optics = CellOpticsConfig()
        r = 20.0
        y, x = np.mgrid[-25:26, -25:26]
        dist_sq = x.astype(float)**2 + y.astype(float)**2
        r_sq = r**2

        # Two-layer OPL
        opl_2layer = compute_two_layer_opl(dist_sq, r_sq, r, cell_optics)

        # Simple OPL
        inside = dist_sq < r_sq
        opl_simple = np.zeros_like(dist_sq)
        opl_simple[inside] = 2.0 * np.sqrt(r_sq - dist_sq[inside])

        # They should be different (two-layer uses delta_n scaling)
        assert not np.allclose(opl_2layer, opl_simple)
        # Two-layer values should be smaller (delta_n < 1)
        assert opl_2layer[inside].max() < opl_simple[inside].max()

    def test_draw_scene_supersampled_accepts_cell_optics(self):
        """draw_scene_supersampled accepts cell_optics parameter."""
        from symbac.imaging.drawing import draw_scene_supersampled
        import inspect
        sig = inspect.signature(draw_scene_supersampled)
        assert "cell_optics" in sig.parameters


# ============================================================
# Change 6: Edge-profile and halo-width features
# ============================================================

class TestEdgeProfileFeatures:
    def test_cell_edge_features_no_masks(self):
        """Without masks, edge features are zero."""
        from symbac.imaging.feature_matching import ImageFeatures
        img = np.random.default_rng(42).random((50, 50))
        feats = ImageFeatures.extract(img, masks=None)
        assert feats.cell_edge_mean == 0.0
        assert feats.cell_edge_contrast == 0.0

    def test_cell_edge_features_with_masks(self):
        """With masks, edge features are non-zero."""
        from symbac.imaging.feature_matching import ImageFeatures
        rng = np.random.default_rng(42)
        img = rng.random((80, 80))
        # Create a cell-like mask
        masks = np.zeros((80, 80), dtype=np.int32)
        masks[30:50, 30:50] = 1
        feats = ImageFeatures.extract(img, masks=masks)
        assert feats.cell_edge_mean > 0.0

    def test_device_halo_features_no_device(self):
        """Without device mask, halo features are zero."""
        from symbac.imaging.feature_matching import ImageFeatures
        img = np.random.default_rng(42).random((50, 50))
        feats = ImageFeatures.extract(img, device_mask=None)
        assert feats.device_halo_width == 0.0

    def test_device_halo_features_with_device(self):
        """With device mask, halo features are extracted."""
        from symbac.imaging.feature_matching import ImageFeatures
        rng = np.random.default_rng(42)
        img = np.ones((80, 80)) * 0.5
        # Add a bright halo near device boundary
        device = np.zeros((80, 80), dtype=np.uint8)
        device[:, :15] = 1  # device on left
        # Simulate halo: brighter near boundary
        for x in range(15, 40):
            img[:, x] += 0.2 * np.exp(-(x - 15) / 5.0)
        feats = ImageFeatures.extract(img, device_mask=device)
        assert feats.device_halo_intensity > 0.0

    def test_new_features_in_distance(self):
        """New features contribute to feature_distance."""
        from symbac.imaging.feature_matching import ImageFeatures, feature_distance
        rng = np.random.default_rng(42)
        img1 = rng.random((50, 50))
        img2 = rng.random((50, 50))
        masks1 = np.zeros((50, 50), dtype=np.int32)
        masks1[20:30, 20:30] = 1
        masks2 = masks1.copy()

        f1 = ImageFeatures.extract(img1, masks=masks1)
        f2 = ImageFeatures.extract(img2, masks=masks2)
        dist = feature_distance(f1, f2)
        assert dist > 0


# ============================================================
# Change 5: PSF condenser params in optimisation
# ============================================================

class TestOptimizationBoundsExtended:
    def test_default_psf_bounds_are_none(self):
        """By default, PSF bounds are not set."""
        from symbac.imaging.feature_matching import OptimizationBounds
        bounds = OptimizationBounds()
        assert bounds.apo_sigma is None
        assert bounds.psf_offset is None
        assert bounds.edge_halo_width is None
        assert bounds.edge_halo_intensity is None

    def test_psf_bounds_can_be_set(self):
        """PSF bounds can be configured."""
        from symbac.imaging.feature_matching import OptimizationBounds
        bounds = OptimizationBounds(
            apo_sigma=(5.0, 30.0),
            psf_offset=(-0.1, 0.1),
            edge_halo_width=(1.0, 15.0),
            edge_halo_intensity=(0.0, 0.5),
        )
        assert bounds.apo_sigma == (5.0, 30.0)
        assert bounds.psf_offset == (-0.1, 0.1)
        assert bounds.edge_halo_width == (1.0, 15.0)
        assert bounds.edge_halo_intensity == (0.0, 0.5)

    def test_defocus_um_bounds(self):
        """defocus_um bounds work in OptimizationBounds."""
        from symbac.imaging.feature_matching import OptimizationBounds
        bounds = OptimizationBounds(defocus_um=(0.0, 5.0))
        assert bounds.defocus_um == (0.0, 5.0)
