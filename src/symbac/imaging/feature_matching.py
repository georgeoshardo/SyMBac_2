"""Feature matching between synthetic and real microscopy images.

Provides multiple techniques to measure and minimize the perceptual distance
between SyMBac-generated synthetic images and real microscopy data. The core
idea is: SyMBac's rendering pipeline has tunable parameters (intensity
multipliers, defocus, noise, halo, PSF sigma), and we need automated methods
to find values that produce synthetic images indistinguishable from real ones.

Three tiers of approach are provided, from lightweight to heavyweight:

1. **Handcrafted feature extraction** (``ImageFeatures``): Extracts
   interpretable statistics — intensity distributions, frequency content,
   texture (GLCM), edge profiles — and computes a scalar distance.

2. **Gradient-free parameter optimization** (``optimize_render_params``):
   Uses scipy's ``differential_evolution`` or ``dual_annealing`` to search
   the RenderConfig parameter space, minimizing the feature distance.

3. **Perceptual (neural) feature matching** (``PerceptualMatcher``):
   Uses a pre-trained CNN (VGG-16) to compare Gram-matrix texture statistics,
   similar to neural style transfer. This captures subtle texture patterns
   that handcrafted features miss.

Typical usage::

    from symbac.imaging.feature_matching import (
        ImageFeatures, feature_distance,
        optimize_render_params, PerceptualMatcher,
    )

    # Quick comparison
    real_feats = ImageFeatures.extract(real_image)
    synth_feats = ImageFeatures.extract(synth_image)
    dist = feature_distance(real_feats, synth_feats)

    # Automatic parameter tuning
    best_config = optimize_render_params(
        real_image, opl_scene, masks, device_mask,
        psf_model, base_config, supersampling=3,
    )

    # Neural texture matching (requires torch)
    matcher = PerceptualMatcher()
    ploss = matcher.perceptual_loss(synth_image, real_image)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

import numpy as np
from numpy import fft
from scipy import ndimage
from skimage.exposure import rescale_intensity
from skimage.transform import resize


# ============================================================
# 1. Handcrafted Feature Extraction
# ============================================================


@dataclass
class ImageFeatures:
    """Extracted feature vector from a microscopy image.

    Each field captures a different aspect of image appearance.
    The full feature vector can be compared via ``feature_distance()``.
    """

    # --- Intensity statistics ---
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    intensity_skew: float = 0.0
    intensity_kurtosis: float = 0.0
    intensity_p5: float = 0.0       # 5th percentile
    intensity_p25: float = 0.0      # 25th percentile
    intensity_p50: float = 0.0      # median
    intensity_p75: float = 0.0      # 75th percentile
    intensity_p95: float = 0.0      # 95th percentile

    # --- Frequency domain ---
    radial_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_slope: float = 0.0     # power-law slope of radial spectrum
    spectral_intercept: float = 0.0

    # --- Texture (GLCM-derived) ---
    glcm_contrast: float = 0.0
    glcm_correlation: float = 0.0
    glcm_energy: float = 0.0
    glcm_homogeneity: float = 0.0

    # --- Edge / gradient ---
    edge_density: float = 0.0       # fraction of pixels that are edges
    gradient_mean: float = 0.0      # mean gradient magnitude
    gradient_std: float = 0.0       # std of gradient magnitude
    laplacian_var: float = 0.0      # variance of Laplacian (focus measure)

    # --- Cell edge profile ---
    cell_edge_mean: float = 0.0         # mean intensity at cell boundaries
    cell_edge_contrast: float = 0.0     # contrast between cell edge and interior
    cell_edge_width: float = 0.0        # characteristic width of edge transition

    # --- Device halo ---
    device_halo_width: float = 0.0      # characteristic width of halo near device edges
    device_halo_intensity: float = 0.0  # peak intensity of the halo

    # --- Region statistics (if masks provided) ---
    region_means: dict = field(default_factory=dict)
    region_stds: dict = field(default_factory=dict)

    @classmethod
    def extract(
        cls,
        image: np.ndarray,
        masks: Optional[np.ndarray] = None,
        device_mask: Optional[np.ndarray] = None,
        n_spectrum_bins: int = 64,
        glcm_levels: int = 64,
        glcm_distance: int = 1,
    ) -> ImageFeatures:
        """Extract features from an image.

        Args:
            image: 2D grayscale image (float or uint).
            masks: Optional instance segmentation mask (cells > 0).
            device_mask: Optional device region mask.
            n_spectrum_bins: Number of radial frequency bins.
            glcm_levels: Number of gray levels for GLCM.
            glcm_distance: Pixel distance for GLCM.

        Returns:
            ImageFeatures with all fields populated.
        """
        feats = cls()

        # Normalize to [0, 1] float
        img = _normalize_image(image)

        # --- Intensity statistics ---
        feats.intensity_mean = float(np.mean(img))
        feats.intensity_std = float(np.std(img))
        feats.intensity_skew = float(_skewness(img))
        feats.intensity_kurtosis = float(_kurtosis(img))
        percentiles = np.percentile(img, [5, 25, 50, 75, 95])
        feats.intensity_p5 = float(percentiles[0])
        feats.intensity_p25 = float(percentiles[1])
        feats.intensity_p50 = float(percentiles[2])
        feats.intensity_p75 = float(percentiles[3])
        feats.intensity_p95 = float(percentiles[4])

        # --- Frequency domain ---
        feats.radial_spectrum = _radial_power_spectrum(img, n_bins=n_spectrum_bins)
        slope, intercept = _spectral_slope(feats.radial_spectrum)
        feats.spectral_slope = slope
        feats.spectral_intercept = intercept

        # --- Texture (GLCM) ---
        glcm_props = _compute_glcm_features(img, levels=glcm_levels, distance=glcm_distance)
        feats.glcm_contrast = glcm_props["contrast"]
        feats.glcm_correlation = glcm_props["correlation"]
        feats.glcm_energy = glcm_props["energy"]
        feats.glcm_homogeneity = glcm_props["homogeneity"]

        # --- Edge / gradient ---
        gy, gx = np.gradient(img)
        grad_mag = np.sqrt(gx**2 + gy**2)
        feats.gradient_mean = float(np.mean(grad_mag))
        feats.gradient_std = float(np.std(grad_mag))

        # Edge density via Canny-like thresholding on gradient
        threshold = np.mean(grad_mag) + 2 * np.std(grad_mag)
        feats.edge_density = float(np.mean(grad_mag > threshold))

        # Laplacian variance (focus measure)
        laplacian = ndimage.laplace(img)
        feats.laplacian_var = float(np.var(laplacian))

        # --- Cell edge profile ---
        if masks is not None:
            edge_feats = _cell_edge_features(img, masks)
            feats.cell_edge_mean = edge_feats["edge_mean"]
            feats.cell_edge_contrast = edge_feats["edge_contrast"]
            feats.cell_edge_width = edge_feats["edge_width"]

        # --- Device halo ---
        if device_mask is not None:
            halo_feats = _device_halo_features(img, device_mask)
            feats.device_halo_width = halo_feats["halo_width"]
            feats.device_halo_intensity = halo_feats["halo_intensity"]

        # --- Region statistics ---
        if masks is not None or device_mask is not None:
            feats.region_means, feats.region_stds = _region_statistics(
                img, masks, device_mask,
            )

        return feats

    def to_vector(self, include_spectrum: bool = True) -> np.ndarray:
        """Convert features to a flat numpy vector for distance computation.

        Args:
            include_spectrum: Whether to include the radial spectrum.

        Returns:
            1D numpy array of feature values.
        """
        scalars = [
            self.intensity_mean, self.intensity_std,
            self.intensity_skew, self.intensity_kurtosis,
            self.intensity_p5, self.intensity_p25, self.intensity_p50,
            self.intensity_p75, self.intensity_p95,
            self.spectral_slope, self.spectral_intercept,
            self.glcm_contrast, self.glcm_correlation,
            self.glcm_energy, self.glcm_homogeneity,
            self.edge_density, self.gradient_mean, self.gradient_std,
            self.laplacian_var,
            self.cell_edge_mean, self.cell_edge_contrast,
            self.cell_edge_width,
            self.device_halo_width, self.device_halo_intensity,
        ]
        vec = np.array(scalars, dtype=np.float64)
        if include_spectrum and len(self.radial_spectrum) > 0:
            # Normalize spectrum for scale invariance
            spec = self.radial_spectrum.copy()
            spec_max = spec.max()
            if spec_max > 0:
                spec = spec / spec_max
            vec = np.concatenate([vec, spec])
        return vec


def feature_distance(
    feats_a: ImageFeatures,
    feats_b: ImageFeatures,
    weights: Optional[dict[str, float]] = None,
    include_spectrum: bool = True,
) -> float:
    """Compute weighted distance between two feature vectors.

    Args:
        feats_a: First feature set (typically real image).
        feats_b: Second feature set (typically synthetic image).
        weights: Optional per-component weights. Keys match
            ``ImageFeatures`` field names. Unspecified fields get weight 1.0.
        include_spectrum: Whether to include radial spectrum in distance.

    Returns:
        Scalar distance (lower = more similar).
    """
    if weights is None:
        weights = DEFAULT_FEATURE_WEIGHTS

    # Compute element-wise weighted distance
    total = 0.0

    # Scalar features
    scalar_names = [
        "intensity_mean", "intensity_std", "intensity_skew",
        "intensity_kurtosis", "intensity_p5", "intensity_p25",
        "intensity_p50", "intensity_p75", "intensity_p95",
        "spectral_slope", "spectral_intercept",
        "glcm_contrast", "glcm_correlation", "glcm_energy",
        "glcm_homogeneity", "edge_density", "gradient_mean",
        "gradient_std", "laplacian_var",
        "cell_edge_mean", "cell_edge_contrast", "cell_edge_width",
        "device_halo_width", "device_halo_intensity",
    ]

    for name in scalar_names:
        a_val = getattr(feats_a, name)
        b_val = getattr(feats_b, name)
        w = weights.get(name, 1.0)
        # Relative error, clamped to avoid division by zero
        denom = max(abs(a_val), abs(b_val), 1e-8)
        total += w * ((a_val - b_val) / denom) ** 2

    # Radial spectrum distance (normalized L2)
    if include_spectrum:
        spec_a = feats_a.radial_spectrum
        spec_b = feats_b.radial_spectrum
        if len(spec_a) > 0 and len(spec_b) > 0:
            # Ensure same length
            min_len = min(len(spec_a), len(spec_b))
            sa = spec_a[:min_len]
            sb = spec_b[:min_len]
            sa_max = sa.max() if sa.max() > 0 else 1.0
            sb_max = sb.max() if sb.max() > 0 else 1.0
            w = weights.get("radial_spectrum", 2.0)
            total += w * np.mean(((sa / sa_max) - (sb / sb_max)) ** 2)

    # Region statistics distance
    for region in feats_a.region_means:
        if region in feats_b.region_means:
            a_mean = feats_a.region_means[region]
            b_mean = feats_b.region_means[region]
            denom = max(abs(a_mean), abs(b_mean), 1e-8)
            w = weights.get(f"region_mean_{region}", 3.0)
            total += w * ((a_mean - b_mean) / denom) ** 2

    for region in feats_a.region_stds:
        if region in feats_b.region_stds:
            a_std = feats_a.region_stds[region]
            b_std = feats_b.region_stds[region]
            denom = max(abs(a_std), abs(b_std), 1e-8)
            w = weights.get(f"region_std_{region}", 2.0)
            total += w * ((a_std - b_std) / denom) ** 2

    return float(np.sqrt(total))


#: Default feature weights emphasizing perceptually important features.
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    # Intensity distribution (most important for visual match)
    "intensity_mean": 5.0,
    "intensity_std": 5.0,
    "intensity_skew": 2.0,
    "intensity_kurtosis": 1.0,
    "intensity_p5": 3.0,
    "intensity_p25": 3.0,
    "intensity_p50": 5.0,
    "intensity_p75": 3.0,
    "intensity_p95": 3.0,
    # Frequency content
    "spectral_slope": 3.0,
    "spectral_intercept": 1.0,
    "radial_spectrum": 4.0,
    # Texture
    "glcm_contrast": 3.0,
    "glcm_correlation": 2.0,
    "glcm_energy": 2.0,
    "glcm_homogeneity": 2.0,
    # Edge / gradient
    "edge_density": 2.0,
    "gradient_mean": 3.0,
    "gradient_std": 2.0,
    "laplacian_var": 2.0,
    # Cell edge profile (important for phase contrast realism)
    "cell_edge_mean": 4.0,
    "cell_edge_contrast": 4.0,
    "cell_edge_width": 3.0,
    # Device halo
    "device_halo_width": 3.0,
    "device_halo_intensity": 3.0,
    # Region-specific (high weight — these are the most direct comparisons)
    "region_mean_media": 5.0,
    "region_mean_cell": 5.0,
    "region_mean_device": 4.0,
    "region_std_media": 3.0,
    "region_std_cell": 3.0,
    "region_std_device": 2.0,
}


# ============================================================
# 2. Gradient-Free Parameter Optimization
# ============================================================


@dataclass
class OptimizationBounds:
    """Parameter bounds for rendering optimization.

    Each field is a (min, max) tuple defining the search space.

    PSF-related bounds (apo_sigma, psf_offset, edge_halo_width, edge_halo_intensity)
    are None by default. When set, the optimizer will include them in the search.
    """
    media_multiplier: tuple[float, float] = (5.0, 100.0)
    cell_multiplier: tuple[float, float] = (-30.0, 0.0)
    device_multiplier: tuple[float, float] = (-150.0, -5.0)
    defocus: tuple[float, float] = (0.0, 10.0)
    defocus_um: Optional[tuple[float, float]] = None
    noise_var: tuple[float, float] = (0.0, 0.01)
    halo_top_intensity: tuple[float, float] = (0.8, 1.2)
    halo_bottom_intensity: tuple[float, float] = (0.8, 1.2)
    # PSF condenser parameters (optional, included only when bounds are set)
    apo_sigma: Optional[tuple[float, float]] = None
    psf_offset: Optional[tuple[float, float]] = None
    # Device edge halo parameters (optional)
    edge_halo_width: Optional[tuple[float, float]] = None
    edge_halo_intensity: Optional[tuple[float, float]] = None


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_config: "RenderConfig"
    best_distance: float
    n_evaluations: int
    convergence_history: list[float] = field(default_factory=list)
    feature_comparison: Optional[dict] = None


def optimize_render_params(
    real_image: np.ndarray,
    opl_scene: np.ndarray,
    scene_masks: np.ndarray,
    device_mask: np.ndarray,
    psf_model: "PSFModel",
    base_config: Optional["RenderConfig"] = None,
    bounds: Optional[OptimizationBounds] = None,
    supersampling: int = 1,
    camera: Optional["Camera"] = None,
    method: str = "differential_evolution",
    maxiter: int = 100,
    popsize: int = 15,
    seed: Optional[int] = None,
    feature_weights: Optional[dict[str, float]] = None,
    real_masks: Optional[np.ndarray] = None,
    real_device_mask: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """Automatically optimize rendering parameters to match a real image.

    Uses gradient-free global optimization (differential evolution or dual
    annealing) to search the rendering parameter space. At each evaluation,
    renders a synthetic image with candidate parameters and computes the
    feature distance to the real image.

    Args:
        real_image: Target real microscopy image.
        opl_scene: Pre-computed OPL scene (at supersampled resolution if
            supersampling > 1).
        scene_masks: Instance segmentation masks (same res as opl_scene).
        device_mask: Device geometry mask (same res as opl_scene).
        psf_model: PSF model for convolution.
        base_config: Starting RenderConfig. Used for fields not being optimized.
            If None, uses defaults.
        bounds: Parameter search bounds. If None, uses defaults.
        supersampling: Supersampling factor.
        camera: Optional camera noise model.
        method: Optimization method. "differential_evolution" or "dual_annealing".
        maxiter: Maximum number of iterations.
        popsize: Population size (for differential_evolution).
        seed: Random seed for reproducibility.
        feature_weights: Custom feature weights for distance computation.
        real_masks: Optional masks for the real image (for region statistics).
        real_device_mask: Optional device mask for the real image.
        callback: Optional callback(xk, convergence) called each iteration.
        verbose: Whether to print progress.

    Returns:
        OptimizationResult with the best configuration found.
    """
    from symbac.imaging.renderer import RenderConfig, render_image
    from scipy.optimize import differential_evolution, dual_annealing

    if base_config is None:
        base_config = RenderConfig()
    if bounds is None:
        bounds = OptimizationBounds()

    # Extract features from real image once
    real_norm = _normalize_image(real_image)
    real_feats = ImageFeatures.extract(
        real_norm, masks=real_masks, device_mask=real_device_mask,
    )

    convergence_history = []
    eval_count = [0]

    # Build bounds list for optimizer dynamically
    # Core rendering parameters (always included)
    use_defocus_um = bounds.defocus_um is not None
    param_names = [
        "media_multiplier", "cell_multiplier", "device_multiplier",
    ]
    if use_defocus_um:
        param_names.append("defocus_um")
    else:
        param_names.append("defocus")
    param_names.extend(["noise_var", "halo_top_intensity", "halo_bottom_intensity"])

    # Optional PSF condenser parameters
    optimize_psf = False
    if bounds.apo_sigma is not None:
        param_names.append("apo_sigma")
        optimize_psf = True
    if bounds.psf_offset is not None:
        param_names.append("psf_offset")
        optimize_psf = True

    # Optional device edge halo parameters
    if bounds.edge_halo_width is not None:
        param_names.append("edge_halo_width")
    if bounds.edge_halo_intensity is not None:
        param_names.append("edge_halo_intensity")

    scipy_bounds = [getattr(bounds, name) for name in param_names]

    def _build_config_and_psf(x):
        """Build RenderConfig and optionally a new PSFModel from parameter vector."""
        params = dict(zip(param_names, x))

        config_kwargs = dict(
            media_multiplier=params["media_multiplier"],
            cell_multiplier=params["cell_multiplier"],
            device_multiplier=params["device_multiplier"],
            noise_var=params["noise_var"],
            halo_top_intensity=params["halo_top_intensity"],
            halo_bottom_intensity=params["halo_bottom_intensity"],
            border_expansion=base_config.border_expansion,
        )
        if use_defocus_um:
            config_kwargs["defocus_um"] = params["defocus_um"]
            config_kwargs["defocus"] = 0.0
        else:
            config_kwargs["defocus"] = params["defocus"]

        if "edge_halo_width" in params:
            config_kwargs["edge_halo_width"] = params["edge_halo_width"]
        if "edge_halo_intensity" in params:
            config_kwargs["edge_halo_intensity"] = params["edge_halo_intensity"]

        config = RenderConfig(**config_kwargs)

        # Build a new PSF if condenser params are being optimized
        current_psf = psf_model
        if optimize_psf:
            from symbac.imaging.optics import PSFModel
            new_sigma = params.get("apo_sigma", psf_model.apo_sigma)
            new_offset = params.get("psf_offset", psf_model.offset)
            current_psf = PSFModel.phase_contrast(
                wavelength=psf_model.wavelength,
                NA=psf_model.NA,
                n=psf_model.n,
                condenser=psf_model.condenser,
                apo_sigma=new_sigma,
                radius=psf_model.radius,
                pixel_scale=psf_model.pixel_scale,
                offset=new_offset,
            )

        return config, current_psf

    def objective(x):
        """Evaluate rendering parameters and return feature distance."""
        config, current_psf = _build_config_and_psf(x)

        try:
            synth, _ = render_image(
                opl_scene, scene_masks, device_mask,
                current_psf, config,
                supersampling=supersampling,
                camera=camera,
                rng=np.random.default_rng(seed),
            )
        except Exception:
            return 1e6  # penalty for failed renders

        # Resize synthetic to match real
        synth_resized = resize(
            synth, real_norm.shape[:2],
            anti_aliasing=True, preserve_range=True,
        )
        synth_norm = _normalize_image(synth_resized)

        synth_feats = ImageFeatures.extract(synth_norm)
        dist = feature_distance(real_feats, synth_feats, weights=feature_weights)

        eval_count[0] += 1
        convergence_history.append(dist)

        if verbose and eval_count[0] % 10 == 0:
            print(f"  eval {eval_count[0]:4d}: distance = {dist:.6f}")

        return dist

    if verbose:
        print(f"Optimizing rendering parameters via {method}...")
        print(f"  Search space: {len(param_names)} parameters")

    if method == "differential_evolution":
        result = differential_evolution(
            objective, scipy_bounds,
            maxiter=maxiter, popsize=popsize,
            seed=seed, callback=callback,
            tol=1e-6, atol=1e-6,
            polish=True,
        )
    elif method == "dual_annealing":
        result = dual_annealing(
            objective, scipy_bounds,
            maxiter=maxiter, seed=seed,
            callback=callback,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'differential_evolution' or 'dual_annealing'.")

    best_x = result.x
    best_config, _ = _build_config_and_psf(best_x)

    if verbose:
        print(f"\nOptimization complete after {eval_count[0]} evaluations")
        print(f"  Best distance: {result.fun:.6f}")
        print(f"  Best config:")
        for name, val in zip(param_names, best_x):
            print(f"    {name}: {val:.4f}")

    return OptimizationResult(
        best_config=best_config,
        best_distance=float(result.fun),
        n_evaluations=eval_count[0],
        convergence_history=convergence_history,
    )


# ============================================================
# 3. Perceptual (Neural) Feature Matching
# ============================================================


class PerceptualMatcher:
    """CNN-based perceptual feature matching using VGG-16.

    Compares images using Gram-matrix texture statistics extracted from
    intermediate layers of a pre-trained VGG-16 network. This is the same
    approach used in neural style transfer (Gatys et al., 2015) and captures
    texture patterns that handcrafted features miss.

    Requires ``torch`` and ``torchvision``.

    Example::

        matcher = PerceptualMatcher()
        loss = matcher.perceptual_loss(synth_image, real_image)
        # loss is a scalar: lower means more similar textures

        # Or get per-layer losses for analysis
        losses = matcher.per_layer_loss(synth_image, real_image)
    """

    def __init__(
        self,
        layers: Optional[list[str]] = None,
        device: Optional[str] = None,
    ):
        """Initialize the perceptual matcher.

        Args:
            layers: VGG-16 layer names to extract features from.
                Default: ["relu1_2", "relu2_2", "relu3_3", "relu4_3"].
            device: Torch device ("cpu", "cuda"). Auto-detected if None.
        """
        try:
            import torch
            import torchvision.models as models
        except ImportError:
            raise ImportError(
                "PerceptualMatcher requires PyTorch. "
                "Install with: pip install torch torchvision"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        self.layer_names = layers

        # Load VGG-16 features
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        vgg = vgg.to(self.device)
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self._layer_map = self._build_layer_map()

        # ImageNet normalization constants
        self._mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)

    def _build_layer_map(self) -> dict[str, int]:
        """Map human-readable layer names to VGG module indices."""
        # VGG-16 layer naming convention
        layer_map = {}
        block = 1
        relu_count = 0
        for i, module in enumerate(self.vgg):
            if isinstance(module, __import__("torch").nn.ReLU):
                relu_count += 1
                name = f"relu{block}_{relu_count}"
                layer_map[name] = i
            elif isinstance(module, __import__("torch").nn.MaxPool2d):
                block += 1
                relu_count = 0
        return layer_map

    def _preprocess(self, image: np.ndarray):
        """Convert a grayscale numpy image to a VGG-ready tensor."""
        import torch

        img = _normalize_image(image)

        # Grayscale -> 3-channel
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2).transpose(2, 0, 1)

        tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        # Apply ImageNet normalization
        tensor = (tensor - self._mean) / self._std
        return tensor

    def _extract_features(self, tensor) -> dict:
        """Extract feature maps at specified layers."""
        features = {}
        x = tensor
        target_indices = {self._layer_map[name]: name for name in self.layer_names
                         if name in self._layer_map}

        for i, module in enumerate(self.vgg):
            x = module(x)
            if i in target_indices:
                features[target_indices[i]] = x
            # Early exit once all features extracted
            if len(features) == len(target_indices):
                break

        return features

    @staticmethod
    def _gram_matrix(feature_map):
        """Compute the Gram matrix (texture representation)."""
        b, c, h, w = feature_map.shape
        F = feature_map.view(b, c, h * w)
        G = F @ F.transpose(1, 2) / (c * h * w)
        return G

    def perceptual_loss(
        self,
        synthetic: np.ndarray,
        real: np.ndarray,
        layer_weights: Optional[dict[str, float]] = None,
    ) -> float:
        """Compute perceptual (Gram-matrix) loss between two images.

        The loss measures how different the texture statistics are between
        the synthetic and real images, as captured by intermediate CNN layers.

        Args:
            synthetic: Synthetic microscopy image.
            real: Real microscopy image.
            layer_weights: Optional per-layer weights. Default: equal weights.

        Returns:
            Scalar loss value (lower = more similar textures).
        """
        import torch

        if layer_weights is None:
            layer_weights = {name: 1.0 for name in self.layer_names}

        # Resize images to same shape (VGG minimum ~32x32)
        target_shape = (max(real.shape[0], 64), max(real.shape[1], 64))
        synth_resized = resize(synthetic, target_shape, anti_aliasing=True, preserve_range=True)
        real_resized = resize(real, target_shape, anti_aliasing=True, preserve_range=True)

        synth_tensor = self._preprocess(synth_resized)
        real_tensor = self._preprocess(real_resized)

        with torch.no_grad():
            synth_feats = self._extract_features(synth_tensor)
            real_feats = self._extract_features(real_tensor)

        total_loss = 0.0
        for layer_name in self.layer_names:
            if layer_name not in synth_feats or layer_name not in real_feats:
                continue
            G_synth = self._gram_matrix(synth_feats[layer_name])
            G_real = self._gram_matrix(real_feats[layer_name])
            layer_loss = float(((G_synth - G_real) ** 2).mean())
            total_loss += layer_weights.get(layer_name, 1.0) * layer_loss

        return total_loss

    def per_layer_loss(
        self,
        synthetic: np.ndarray,
        real: np.ndarray,
    ) -> dict[str, float]:
        """Compute per-layer Gram-matrix losses for analysis.

        Args:
            synthetic: Synthetic microscopy image.
            real: Real microscopy image.

        Returns:
            Dict mapping layer name to its Gram-matrix loss.
        """
        import torch

        target_shape = (max(real.shape[0], 64), max(real.shape[1], 64))
        synth_resized = resize(synthetic, target_shape, anti_aliasing=True, preserve_range=True)
        real_resized = resize(real, target_shape, anti_aliasing=True, preserve_range=True)

        synth_tensor = self._preprocess(synth_resized)
        real_tensor = self._preprocess(real_resized)

        with torch.no_grad():
            synth_feats = self._extract_features(synth_tensor)
            real_feats = self._extract_features(real_tensor)

        losses = {}
        for layer_name in self.layer_names:
            if layer_name not in synth_feats or layer_name not in real_feats:
                continue
            G_synth = self._gram_matrix(synth_feats[layer_name])
            G_real = self._gram_matrix(real_feats[layer_name])
            losses[layer_name] = float(((G_synth - G_real) ** 2).mean())

        return losses


# ============================================================
# 4. Combined Neural + Handcrafted Optimization
# ============================================================


def optimize_render_params_neural(
    real_image: np.ndarray,
    opl_scene: np.ndarray,
    scene_masks: np.ndarray,
    device_mask: np.ndarray,
    psf_model: "PSFModel",
    base_config: Optional["RenderConfig"] = None,
    bounds: Optional[OptimizationBounds] = None,
    supersampling: int = 1,
    camera: Optional["Camera"] = None,
    maxiter: int = 50,
    popsize: int = 10,
    seed: Optional[int] = None,
    perceptual_weight: float = 0.5,
    feature_weight: float = 0.5,
    verbose: bool = True,
) -> OptimizationResult:
    """Optimize rendering params using combined handcrafted + neural features.

    Combines the handcrafted feature distance with VGG perceptual loss.
    More expensive per evaluation but captures deeper texture statistics.

    Args:
        real_image: Target real microscopy image.
        opl_scene: Pre-computed OPL scene.
        scene_masks: Instance masks.
        device_mask: Device mask.
        psf_model: PSF model.
        base_config: Base rendering configuration.
        bounds: Parameter bounds.
        supersampling: Supersampling factor.
        camera: Camera noise model.
        maxiter: Max iterations for optimizer.
        popsize: Population size.
        seed: Random seed.
        perceptual_weight: Weight for VGG perceptual loss [0, 1].
        feature_weight: Weight for handcrafted feature distance [0, 1].
        verbose: Print progress.

    Returns:
        OptimizationResult with best found configuration.
    """
    from symbac.imaging.renderer import RenderConfig, render_image
    from scipy.optimize import differential_evolution

    if base_config is None:
        base_config = RenderConfig()
    if bounds is None:
        bounds = OptimizationBounds()

    # Initialize neural matcher
    matcher = PerceptualMatcher()

    # Extract handcrafted features from real image once
    real_norm = _normalize_image(real_image)
    real_feats = ImageFeatures.extract(real_norm)

    convergence_history = []
    eval_count = [0]

    # Build dynamic parameter list (same logic as optimize_render_params)
    use_defocus_um = bounds.defocus_um is not None
    param_names = ["media_multiplier", "cell_multiplier", "device_multiplier"]
    if use_defocus_um:
        param_names.append("defocus_um")
    else:
        param_names.append("defocus")
    param_names.extend(["noise_var", "halo_top_intensity", "halo_bottom_intensity"])

    optimize_psf = False
    if bounds.apo_sigma is not None:
        param_names.append("apo_sigma")
        optimize_psf = True
    if bounds.psf_offset is not None:
        param_names.append("psf_offset")
        optimize_psf = True
    if bounds.edge_halo_width is not None:
        param_names.append("edge_halo_width")
    if bounds.edge_halo_intensity is not None:
        param_names.append("edge_halo_intensity")

    scipy_bounds = [getattr(bounds, name) for name in param_names]

    def _build_config_and_psf(x):
        params = dict(zip(param_names, x))
        config_kwargs = dict(
            media_multiplier=params["media_multiplier"],
            cell_multiplier=params["cell_multiplier"],
            device_multiplier=params["device_multiplier"],
            noise_var=params["noise_var"],
            halo_top_intensity=params["halo_top_intensity"],
            halo_bottom_intensity=params["halo_bottom_intensity"],
            border_expansion=base_config.border_expansion,
        )
        if use_defocus_um:
            config_kwargs["defocus_um"] = params["defocus_um"]
            config_kwargs["defocus"] = 0.0
        else:
            config_kwargs["defocus"] = params["defocus"]
        if "edge_halo_width" in params:
            config_kwargs["edge_halo_width"] = params["edge_halo_width"]
        if "edge_halo_intensity" in params:
            config_kwargs["edge_halo_intensity"] = params["edge_halo_intensity"]

        config = RenderConfig(**config_kwargs)
        current_psf = psf_model
        if optimize_psf:
            from symbac.imaging.optics import PSFModel
            current_psf = PSFModel.phase_contrast(
                wavelength=psf_model.wavelength, NA=psf_model.NA, n=psf_model.n,
                condenser=psf_model.condenser,
                apo_sigma=params.get("apo_sigma", psf_model.apo_sigma),
                radius=psf_model.radius, pixel_scale=psf_model.pixel_scale,
                offset=params.get("psf_offset", psf_model.offset),
            )
        return config, current_psf

    def objective(x):
        config, current_psf = _build_config_and_psf(x)

        try:
            synth, _ = render_image(
                opl_scene, scene_masks, device_mask, current_psf, config,
                supersampling=supersampling, camera=camera,
                rng=np.random.default_rng(seed),
            )
        except Exception:
            return 1e6

        synth_resized = resize(
            synth, real_norm.shape[:2],
            anti_aliasing=True, preserve_range=True,
        )
        synth_norm = _normalize_image(synth_resized)

        # Handcrafted distance
        synth_feats = ImageFeatures.extract(synth_norm)
        hc_dist = feature_distance(real_feats, synth_feats)

        # Neural perceptual loss
        p_loss = matcher.perceptual_loss(synth_norm, real_norm)

        # Combine
        combined = feature_weight * hc_dist + perceptual_weight * p_loss

        eval_count[0] += 1
        convergence_history.append(combined)

        if verbose and eval_count[0] % 5 == 0:
            print(f"  eval {eval_count[0]:4d}: combined={combined:.4f} "
                  f"(features={hc_dist:.4f}, perceptual={p_loss:.6f})")

        return combined

    if verbose:
        print("Optimizing with combined handcrafted + neural features...")

    result = differential_evolution(
        objective, scipy_bounds,
        maxiter=maxiter, popsize=popsize, seed=seed,
        tol=1e-5, polish=True,
    )

    best_x = result.x
    best_config, _ = _build_config_and_psf(best_x)

    return OptimizationResult(
        best_config=best_config,
        best_distance=float(result.fun),
        n_evaluations=eval_count[0],
        convergence_history=convergence_history,
    )


# ============================================================
# 5. Diagnostic / Reporting
# ============================================================


def compare_images(
    real_image: np.ndarray,
    synthetic_image: np.ndarray,
    real_masks: Optional[np.ndarray] = None,
    real_device_mask: Optional[np.ndarray] = None,
    synth_masks: Optional[np.ndarray] = None,
    synth_device_mask: Optional[np.ndarray] = None,
) -> dict:
    """Generate a comprehensive comparison report between two images.

    Args:
        real_image: Real microscopy image.
        synthetic_image: Synthetic image to compare.
        real_masks: Optional cell masks for real image.
        real_device_mask: Optional device mask for real image.
        synth_masks: Optional cell masks for synthetic image.
        synth_device_mask: Optional device mask for synthetic image.

    Returns:
        Dict with comparison metrics and feature breakdowns.
    """
    real_norm = _normalize_image(real_image)
    synth_resized = resize(
        synthetic_image, real_norm.shape[:2],
        anti_aliasing=True, preserve_range=True,
    )
    synth_norm = _normalize_image(synth_resized)

    real_feats = ImageFeatures.extract(
        real_norm, masks=real_masks, device_mask=real_device_mask,
    )
    synth_feats = ImageFeatures.extract(
        synth_norm, masks=synth_masks, device_mask=synth_device_mask,
    )

    dist = feature_distance(real_feats, synth_feats)

    # Per-category breakdown
    intensity_names = [
        "intensity_mean", "intensity_std", "intensity_skew",
        "intensity_kurtosis", "intensity_p50",
    ]
    texture_names = [
        "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity",
    ]
    edge_names = ["edge_density", "gradient_mean", "gradient_std", "laplacian_var"]
    freq_names = ["spectral_slope", "spectral_intercept"]

    def _category_errors(names):
        errors = {}
        for name in names:
            a = getattr(real_feats, name)
            b = getattr(synth_feats, name)
            denom = max(abs(a), abs(b), 1e-8)
            errors[name] = {
                "real": float(a),
                "synthetic": float(b),
                "relative_error": float(abs(a - b) / denom),
            }
        return errors

    report = {
        "overall_distance": dist,
        "intensity": _category_errors(intensity_names),
        "texture": _category_errors(texture_names),
        "edges": _category_errors(edge_names),
        "frequency": _category_errors(freq_names),
        "real_features": real_feats,
        "synth_features": synth_feats,
    }

    # SSIM if available
    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(real_norm, synth_norm, data_range=1.0)
        report["ssim"] = float(ssim)
    except ImportError:
        pass

    return report


def plot_comparison(
    real_image: np.ndarray,
    synthetic_image: np.ndarray,
    report: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a visual comparison between real and synthetic images.

    Shows the images side-by-side with intensity histograms, radial
    power spectra, and feature distance breakdown.

    Args:
        real_image: Real microscopy image.
        synthetic_image: Synthetic image.
        report: Optional pre-computed comparison report. If None, computed here.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    if report is None:
        report = compare_images(real_image, synthetic_image)

    real_norm = _normalize_image(real_image)
    synth_resized = resize(
        synthetic_image, real_norm.shape[:2],
        anti_aliasing=True, preserve_range=True,
    )
    synth_norm = _normalize_image(synth_resized)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: Images
    axes[0, 0].imshow(real_norm, cmap="Greys_r")
    axes[0, 0].set_title("Real Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(synth_norm, cmap="Greys_r")
    axes[0, 1].set_title("Synthetic Image")
    axes[0, 1].axis("off")

    # Difference map
    diff = np.abs(real_norm - synth_norm)
    axes[0, 2].imshow(diff, cmap="hot")
    axes[0, 2].set_title(f"Absolute Difference (dist={report['overall_distance']:.4f})")
    axes[0, 2].axis("off")

    # Row 1: Analysis
    # Intensity histograms
    axes[1, 0].hist(real_norm.ravel(), bins=128, alpha=0.6, label="Real", density=True, color="blue")
    axes[1, 0].hist(synth_norm.ravel(), bins=128, alpha=0.6, label="Synthetic", density=True, color="red")
    axes[1, 0].set_title("Intensity Histograms")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Density")

    # Radial power spectra
    real_spec = report["real_features"].radial_spectrum
    synth_spec = report["synth_features"].radial_spectrum
    if len(real_spec) > 0 and len(synth_spec) > 0:
        min_len = min(len(real_spec), len(synth_spec))
        freqs = np.arange(1, min_len + 1)
        axes[1, 1].loglog(freqs, real_spec[:min_len] + 1e-10, label="Real", color="blue")
        axes[1, 1].loglog(freqs, synth_spec[:min_len] + 1e-10, label="Synthetic", color="red")
        axes[1, 1].set_title("Radial Power Spectrum")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Spatial Frequency")
        axes[1, 1].set_ylabel("Power")

    # Feature error breakdown
    categories = ["intensity", "texture", "edges", "frequency"]
    cat_errors = []
    for cat in categories:
        if cat in report:
            errors = [v["relative_error"] for v in report[cat].values()]
            cat_errors.append(np.mean(errors) if errors else 0)
        else:
            cat_errors.append(0)

    axes[1, 2].barh(categories, cat_errors, color=["steelblue", "darkorange", "green", "purple"])
    axes[1, 2].set_title("Mean Relative Error by Category")
    axes[1, 2].set_xlabel("Relative Error")
    ssim_val = report.get("ssim", None)
    if ssim_val is not None:
        axes[1, 2].text(
            0.95, 0.05, f"SSIM = {ssim_val:.4f}",
            transform=axes[1, 2].transAxes,
            ha="right", fontsize=11, weight="bold",
        )

    plt.suptitle("Real vs Synthetic Image Comparison", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Internal helpers
# ============================================================


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1] float64."""
    img = image.astype(np.float64)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        # Constant image: map to 0
        img = np.zeros_like(img)
    return img


def _skewness(arr: np.ndarray) -> float:
    """Compute skewness of a flattened array."""
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis of a flattened array."""
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def _radial_power_spectrum(image: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Compute the radially averaged power spectrum.

    Args:
        image: 2D grayscale image.
        n_bins: Number of radial frequency bins.

    Returns:
        1D array of radially averaged power values.
    """
    h, w = image.shape
    F = fft.fftshift(fft.fft2(image))
    power = np.abs(F) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    r = np.sqrt(x ** 2 + y ** 2)

    max_r = min(cy, cx)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    spectrum = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            spectrum[i] = np.mean(power[mask])

    return spectrum


def _spectral_slope(spectrum: np.ndarray) -> tuple[float, float]:
    """Fit a power-law slope to the radial spectrum.

    Natural images follow f^(-beta) power-law scaling. The slope beta
    characterizes the spatial frequency content.

    Returns:
        (slope, intercept) from log-log linear regression.
    """
    nonzero = spectrum > 0
    if np.sum(nonzero) < 2:
        return 0.0, 0.0

    freqs = np.arange(1, len(spectrum) + 1, dtype=np.float64)
    log_f = np.log10(freqs[nonzero])
    log_p = np.log10(spectrum[nonzero])

    # Simple linear regression
    n = len(log_f)
    sx = np.sum(log_f)
    sy = np.sum(log_p)
    sxx = np.sum(log_f ** 2)
    sxy = np.sum(log_f * log_p)
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-10:
        return 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    return float(slope), float(intercept)


def _compute_glcm_features(
    image: np.ndarray,
    levels: int = 64,
    distance: int = 1,
) -> dict[str, float]:
    """Compute GLCM texture features (contrast, correlation, energy, homogeneity).

    Uses a fast numpy implementation to avoid optional skimage dependency.

    Args:
        image: 2D grayscale image in [0, 1] range.
        levels: Number of gray levels for quantization.
        distance: Pixel distance for co-occurrence.

    Returns:
        Dict with GLCM property values averaged over 4 directions.
    """
    # Quantize to integer levels
    quantized = np.clip(
        (image * (levels - 1)).astype(np.int32), 0, levels - 1
    )

    # Compute GLCM for 4 directions: 0°, 45°, 90°, 135°
    offsets = [(0, distance), (distance, distance), (distance, 0), (distance, -distance)]
    results = {"contrast": 0, "correlation": 0, "energy": 0, "homogeneity": 0}

    for dy, dx in offsets:
        glcm = _glcm_single(quantized, levels, dy, dx)

        # Properties
        i_idx, j_idx = np.mgrid[0:levels, 0:levels]
        i_idx = i_idx.astype(np.float64)
        j_idx = j_idx.astype(np.float64)

        # Marginals
        px = glcm.sum(axis=1)
        py = glcm.sum(axis=0)
        mu_x = np.sum(i_idx[:, 0] * px)
        mu_y = np.sum(j_idx[0, :] * py)
        sig_x = np.sqrt(np.sum((i_idx[:, 0] - mu_x) ** 2 * px))
        sig_y = np.sqrt(np.sum((j_idx[0, :] - mu_y) ** 2 * py))

        # Contrast: sum_ij (i-j)^2 * P(i,j)
        results["contrast"] += float(np.sum((i_idx - j_idx) ** 2 * glcm))

        # Correlation: sum_ij (i-mu_x)(j-mu_y)*P(i,j) / (sig_x*sig_y)
        if sig_x > 0 and sig_y > 0:
            results["correlation"] += float(
                np.sum((i_idx - mu_x) * (j_idx - mu_y) * glcm) / (sig_x * sig_y)
            )

        # Energy: sum_ij P(i,j)^2
        results["energy"] += float(np.sum(glcm ** 2))

        # Homogeneity: sum_ij P(i,j) / (1 + |i-j|)
        results["homogeneity"] += float(np.sum(glcm / (1 + np.abs(i_idx - j_idx))))

    # Average over 4 directions
    for key in results:
        results[key] /= len(offsets)

    return results


def _glcm_single(
    quantized: np.ndarray,
    levels: int,
    dy: int,
    dx: int,
) -> np.ndarray:
    """Compute a single GLCM for a given offset direction."""
    h, w = quantized.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    # Determine valid region
    y_start = max(0, -dy)
    y_end = min(h, h - dy)
    x_start = max(0, -dx)
    x_end = min(w, w - dx)

    if y_end <= y_start or x_end <= x_start:
        return glcm / max(glcm.sum(), 1)

    ref = quantized[y_start:y_end, x_start:x_end]
    neighbor = quantized[y_start + dy:y_end + dy, x_start + dx:x_end + dx]

    # Vectorized GLCM accumulation
    np.add.at(glcm, (ref.ravel(), neighbor.ravel()), 1)

    # Normalize
    total = glcm.sum()
    if total > 0:
        glcm /= total

    return glcm


def _cell_edge_features(
    image: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float]:
    """Extract features characterizing the intensity profile at cell edges.

    Measures the mean intensity at cell boundaries, the contrast between
    edge and interior, and the characteristic width of the edge transition.

    Args:
        image: 2D grayscale image in [0, 1] range.
        masks: Instance segmentation mask (cells > 0).

    Returns:
        Dict with "edge_mean", "edge_contrast", "edge_width".
    """
    result = {"edge_mean": 0.0, "edge_contrast": 0.0, "edge_width": 0.0}

    cell_mask = masks > 0
    if not cell_mask.any():
        return result

    # Find cell boundaries using morphological gradient
    from scipy.ndimage import binary_erosion, binary_dilation
    eroded = binary_erosion(cell_mask, iterations=1)
    dilated = binary_dilation(cell_mask, iterations=1)
    boundary = dilated & ~eroded

    if not boundary.any():
        return result

    # Edge mean intensity
    result["edge_mean"] = float(np.mean(image[boundary]))

    # Interior = eroded cell region
    if eroded.any():
        interior_mean = float(np.mean(image[eroded]))
        # Edge contrast: difference between edge and interior
        result["edge_contrast"] = abs(result["edge_mean"] - interior_mean)

    # Edge width: measure the gradient magnitude at the boundary
    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx**2 + gy**2)
    if boundary.any() and grad_mag[boundary].mean() > 0:
        # Edge width inversely proportional to gradient steepness
        # Normalise by the edge contrast to get a width-like measure
        mean_grad = float(np.mean(grad_mag[boundary]))
        if result["edge_contrast"] > 0:
            result["edge_width"] = result["edge_contrast"] / mean_grad
        else:
            result["edge_width"] = 1.0 / mean_grad

    return result


def _device_halo_features(
    image: np.ndarray,
    device_mask: np.ndarray,
) -> dict[str, float]:
    """Extract features characterizing the halo at device/media boundaries.

    Measures the characteristic width and intensity of the bright fringe
    near the device edge, which is a key indicator of phase contrast quality.

    Args:
        image: 2D grayscale image in [0, 1] range.
        device_mask: Binary device mask (1 = device, 0 = media).

    Returns:
        Dict with "halo_width" and "halo_intensity".
    """
    from scipy.ndimage import distance_transform_edt

    result = {"halo_width": 0.0, "halo_intensity": 0.0}

    device_bool = device_mask.astype(bool)
    if not device_bool.any() or device_bool.all():
        return result

    # Distance from each media pixel to nearest device pixel
    media_mask = ~device_bool
    if not media_mask.any():
        return result

    dist = distance_transform_edt(media_mask)

    # Sample intensity as a function of distance from device edge
    max_dist = min(20, int(dist.max()))
    if max_dist < 2:
        return result

    # Compute mean intensity at each distance bin
    intensities = []
    for d in range(1, max_dist + 1):
        band = media_mask & (dist >= d - 0.5) & (dist < d + 0.5)
        if band.any():
            intensities.append(float(np.mean(image[band])))
        else:
            intensities.append(np.nan)

    intensities = np.array(intensities)
    valid = ~np.isnan(intensities)
    if valid.sum() < 3:
        return result

    # Background intensity = mean of far pixels (last third)
    far_start = max(1, len(intensities) * 2 // 3)
    far_intensities = intensities[far_start:]
    far_valid = ~np.isnan(far_intensities)
    if far_valid.any():
        background = float(np.mean(far_intensities[far_valid]))
    else:
        background = float(np.nanmean(intensities))

    # Halo intensity = peak deviation from background
    deviations = np.abs(intensities - background)
    deviations[~valid] = 0
    result["halo_intensity"] = float(np.max(deviations))

    # Halo width = distance at which intensity drops to 1/e of peak
    peak_dev = result["halo_intensity"]
    if peak_dev > 0:
        threshold = peak_dev / np.e
        # Find the first distance where deviation drops below threshold
        below = deviations < threshold
        crossed = np.where(below & valid)[0]
        if len(crossed) > 0:
            result["halo_width"] = float(crossed[0] + 1)
        else:
            result["halo_width"] = float(max_dist)

    return result


def _region_statistics(
    image: np.ndarray,
    masks: Optional[np.ndarray],
    device_mask: Optional[np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-region intensity statistics.

    Extracts mean and std for media, cell, and device regions.
    If masks are not provided, uses automatic thresholding.

    Returns:
        (means_dict, stds_dict) with keys "media", "cell", "device".
    """
    means = {}
    stds = {}

    if masks is not None and device_mask is not None:
        cell_region = masks > 0
        device_region = device_mask.astype(bool)
        media_region = ~cell_region & ~device_region
    elif masks is not None:
        cell_region = masks > 0
        media_region = ~cell_region
        device_region = np.zeros_like(cell_region)
    elif device_mask is not None:
        device_region = device_mask.astype(bool)
        media_region = ~device_region
        cell_region = np.zeros_like(device_region)
    else:
        # Auto-segment using multi-Otsu thresholding
        try:
            from skimage.filters import threshold_multiotsu
            thresholds = threshold_multiotsu(image, classes=3)
            regions = np.digitize(image, bins=thresholds)
            # Typically: 0=dark (device), 1=mid (cells), 2=bright (media)
            device_region = regions == 0
            cell_region = regions == 1
            media_region = regions == 2
        except Exception:
            return means, stds

    for name, region in [("media", media_region), ("cell", cell_region), ("device", device_region)]:
        if np.any(region):
            means[name] = float(np.mean(image[region]))
            stds[name] = float(np.std(image[region]))

    return means, stds
