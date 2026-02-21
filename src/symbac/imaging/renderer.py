"""Core rendering pipeline for synthetic microscopy image generation.

Takes OPL scenes and masks from the drawing module, applies region-specific
intensity multipliers, PSF convolution, noise, and optional image matching
to produce photorealistic synthetic microscopy images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms, rescale_intensity
from skimage.transform import rescale
from skimage.util import random_noise

from symbac.imaging.convolution import convolve_rescale
from symbac.imaging.optics import PSFModel, Camera, PSFMode, apply_defocus_pupil


@dataclass
class RenderConfig:
    """Configuration for the rendering pipeline.

    Controls intensity multipliers for different image regions (media, cells,
    device), defocus amount, noise parameters, and image matching options.

    Defocus can be specified in two ways:

    - ``defocus``: Legacy Gaussian blur sigma (pixels). Used when ``defocus_um``
      is None. Fast but physically inaccurate.
    - ``defocus_um``: Physical defocus distance in microns. Uses a proper
      pupil-function phase aberration model. Preferred for realistic renders.
      When set, ``defocus`` is ignored.

    Output normalisation:

    - ``normalize_output=True`` (default): Rescale output to [0, 1].
    - ``normalize_output=False``: Preserve physical intensity scale.
    - ``intensity_range``: When set, use this (min, max) as the input range
      for rescaling instead of auto-detecting from the data.

    For phase contrast microscopy, the PSF kernel sum is typically negative
    (~-1), which inverts overall intensity during convolution. This means:

    - ``device_multiplier`` should be **negative** (e.g. -50): after PSF
      inversion, device (PDMS) regions become positive → dark on Greys_r.
    - ``media_multiplier`` should be **positive** (e.g. 30): after inversion,
      media regions become negative → bright on Greys_r.
    - ``cell_multiplier`` should be **negative** (e.g. -5): this *subtracts*
      the cell OPL from the media level. After PSF inversion, cells become
      less negative than media → darker than media on Greys_r, matching the
      characteristic dark-cell appearance of positive phase contrast.
    """
    media_multiplier: float = 30.0
    cell_multiplier: float = -5.0
    device_multiplier: float = -50.0
    defocus: float = 1.0
    defocus_um: Optional[float] = None
    noise_var: float = 0.001
    border_expansion: float = 1.5
    halo_top_intensity: float = 1.0
    halo_bottom_intensity: float = 1.0
    edge_halo_width: float = 0.0
    edge_halo_intensity: float = 0.0
    intensity_range: Optional[tuple[float, float]] = None
    normalize_output: bool = True


def generate_pc_opl(
    scene_opl: np.ndarray,
    scene_masks: np.ndarray,
    device_mask: np.ndarray,
    config: RenderConfig,
    is_fluorescence: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a phase-contrast or fluorescence OPL scene with region intensities.

    For phase contrast: the scene is built by assigning device_multiplier to
    device (PDMS) regions, media_multiplier to the trench/chamber interior,
    and then adding cell_multiplier * OPL on top of whatever region the cell
    occupies (typically media).

    With the standard negative cell_multiplier, cells subtract from the media
    level, which after PSF convolution (kernel sum ~-1) produces the dark-cell
    appearance of positive phase contrast microscopy.

    For fluorescence: only cells emit light; background is zero.

    Args:
        scene_opl: Raw OPL scene from draw_scene().
        scene_masks: Instance segmentation masks.
        device_mask: Binary device geometry mask.
        config: Rendering configuration.
        is_fluorescence: If True, render as fluorescence (cells only).

    Returns:
        Tuple of (expanded_scene, expanded_background, expanded_masks).
    """
    h, w = scene_opl.shape
    device_bool = device_mask.astype(bool)

    # Normalize OPL to [0, 1] range. Raw OPL from segment chains can be
    # very large due to overlapping spheres. Normalizing ensures that
    # cell_multiplier has a consistent effect regardless of cell geometry.
    opl_max = scene_opl.max()
    if opl_max > 0:
        opl_norm = scene_opl / opl_max
    else:
        opl_norm = scene_opl

    if is_fluorescence:
        # Fluorescence: only cells emit, background is zero
        scene = opl_norm * config.cell_multiplier
        background = np.zeros_like(scene)
    else:
        # Phase contrast: build the layered intensity scene
        # 1. Start with device_multiplier everywhere
        scene = np.full_like(scene_opl, config.device_multiplier, dtype=np.float64)

        # 2. Set non-device regions (media + cells) to media_multiplier
        scene[~device_bool] = config.media_multiplier

        # 3. Save background (before adding cells)
        background = scene.copy()

        # 4. Add normalized cell OPL contribution on top
        #    cell_multiplier now controls the OPL intensity relative to
        #    the media background. E.g., media=30, cell_mult=3 means
        #    the thickest cell pixel reaches 33.
        scene = scene + opl_norm * config.cell_multiplier

    # Border expansion to prevent convolution edge artifacts
    exp = config.border_expansion
    eh = int(h * exp)
    ew = int(w * exp)

    if is_fluorescence:
        fill_val = 0.0
    else:
        fill_val = config.media_multiplier

    expanded_scene = np.full((eh, ew), fill_val, dtype=np.float64)
    expanded_bg = np.full((eh, ew), fill_val, dtype=np.float64)
    expanded_masks = np.zeros((eh, ew), dtype=np.int32)

    # Place scene at center of expanded canvas
    y_off = (eh - h) // 2
    x_off = (ew - w) // 2

    expanded_scene[y_off:y_off + h, x_off:x_off + w] = scene
    expanded_bg[y_off:y_off + h, x_off:x_off + w] = background
    expanded_masks[y_off:y_off + h, x_off:x_off + w] = scene_masks

    return expanded_scene, expanded_bg, expanded_masks


def apply_illumination_gradient(
    image: np.ndarray,
    top_intensity: float = 1.0,
    bottom_intensity: float = 1.0,
) -> np.ndarray:
    """Apply a linear intensity gradient to simulate uneven illumination.

    In phase contrast microscopy, the illumination can vary across the
    field of view, producing a smooth gradient.

    Args:
        image: Input 2D image.
        top_intensity: Multiplier at the top of the image.
        bottom_intensity: Multiplier at the bottom.

    Returns:
        Image with illumination gradient applied.
    """
    if top_intensity == bottom_intensity == 1.0:
        return image

    h = image.shape[0]
    gradient = np.linspace(top_intensity, bottom_intensity, h)[:, np.newaxis]
    return image * gradient


# Keep backward compatibility
apply_halo = apply_illumination_gradient


def compute_device_edge_halo(
    device_mask: np.ndarray,
    halo_width: float = 5.0,
    halo_intensity: float = 0.15,
) -> np.ndarray:
    """Compute a spatially varying halo at device/media boundaries.

    In phase contrast microscopy, the refractive index mismatch at the
    PDMS/media boundary produces a diffraction fringe (halo). This is
    modelled as an exponential decay from the device edge into the media.

    Args:
        device_mask: Binary device mask (1 = device, 0 = media).
        halo_width: Characteristic decay width in pixels.
        halo_intensity: Peak intensity of the halo (additive).

    Returns:
        2D halo image (same shape as device_mask). Values are additive
        intensity contributions, zero far from edges.
    """
    from scipy.ndimage import distance_transform_edt

    device_bool = device_mask.astype(bool)
    if not device_bool.any():
        return np.zeros_like(device_mask, dtype=np.float64)

    # Distance from each media pixel to nearest device pixel
    media_dist = distance_transform_edt(~device_bool)

    # Distance from each device pixel to nearest media pixel
    device_dist = distance_transform_edt(device_bool)

    # Halo decays exponentially from the boundary into both regions
    # Stronger on the media side (bright fringe), weaker on device side
    halo = np.zeros_like(device_mask, dtype=np.float64)

    if halo_width > 0:
        # Media side: bright fringe decaying away from boundary
        media_fringe = halo_intensity * np.exp(-media_dist / halo_width)
        media_fringe[device_bool] = 0.0

        # Device side: dimmer fringe
        device_fringe = 0.3 * halo_intensity * np.exp(-device_dist / halo_width)
        device_fringe[~device_bool] = 0.0

        halo = media_fringe + device_fringe

    return halo


def render_image(
    scene_opl: np.ndarray,
    scene_masks: np.ndarray,
    device_mask: np.ndarray,
    psf_model: PSFModel,
    config: RenderConfig,
    supersampling: int = 1,
    camera: Optional[Camera] = None,
    real_image: Optional[np.ndarray] = None,
    match_histogram: bool = False,
    perlin_background: Optional[np.ndarray] = None,
    perlin_attenuation: Optional[float] = None,
    perlin_blur_sigma: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a synthetic microscopy image from OPL scene and masks.

    Implements the SyMBac pipeline: when supersampling > 1, the inputs are
    expected to be at supersampled resolution. The scene is convolved with
    the PSF at high resolution, then downscaled to native resolution before
    noise is applied. This captures sub-pixel features that would be lost
    if downscaling happened before convolution.

    Pipeline:
    1. Generate region-intensity OPL scene (at input resolution)
    2. Apply defocus to PSF kernel
    3. Convolve scene with PSF (at input resolution)
    4. Crop to scene bounds (still at input resolution)
    5. Apply halo effect
    6. Downscale to native resolution (if supersampling > 1)
    6b. Add Perlin noise background (for colony/agar pad rendering)
    7. Rescale intensity to [0, 1]
    8. Optionally match histogram to a real image
    9. Add camera or ad-hoc noise

    Args:
        scene_opl: Raw OPL scene. At supersampled resolution if supersampling > 1.
        scene_masks: Instance segmentation masks (same resolution as scene_opl).
        device_mask: Binary device mask (same resolution as scene_opl).
        psf_model: PSF model to use for convolution. pixel_scale should match
            the input resolution (i.e., native_pixel_scale / supersampling).
        config: Rendering configuration.
        supersampling: If > 1, inputs are at this multiple of native resolution.
            The convolved result is downscaled by this factor before noise.
        camera: Optional Camera for noise simulation. If None, ad-hoc noise is used.
        real_image: Optional real image for histogram matching.
        match_histogram: Whether to match the histogram to the real image.
        perlin_background: Optional Perlin noise background to add to the image.
            Used for colony/agar pad rendering to simulate substrate texture.
            Should be at native (not supersampled) resolution.
        perlin_attenuation: Divide perlin noise by this factor (default: random 500-1500).
        perlin_blur_sigma: Gaussian blur sigma for perlin noise (default: random 1-3).
        rng: Optional numpy random generator.

    Returns:
        Tuple of (synthetic_image, output_masks) at native resolution.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_fluo = psf_model.mode in (PSFMode.FLUORESCENCE_2D, PSFMode.FLUORESCENCE_3D)

    # Step 1: Generate region-intensity scene (at input resolution)
    expanded_scene, expanded_bg, expanded_masks = generate_pc_opl(
        scene_opl, scene_masks, device_mask, config, is_fluorescence=is_fluo,
    )

    # Step 2: Get PSF kernel with defocus
    # For phase contrast, the kernel sum is typically negative (it inverts
    # bright objects to dark). Do NOT renormalize to sum=1, as that
    # destroys the phase contrast effect.
    kernel = psf_model.kernel_2d.copy()
    if config.defocus_um is not None and config.defocus_um > 0:
        # Pupil-function defocus: physically accurate model
        kernel = apply_defocus_pupil(
            kernel,
            defocus_um=config.defocus_um,
            wavelength=psf_model.wavelength,
            NA=psf_model.NA,
            n=psf_model.n,
            pixel_scale=psf_model.pixel_scale,
        )
    elif config.defocus > 0:
        # Legacy Gaussian blur defocus (fast but approximate)
        kernel = gaussian_filter(kernel, config.defocus, mode="constant")

    # Step 3: No additional normalization for phase contrast
    if is_fluo and kernel.sum() != 0:
        kernel = kernel / kernel.sum()  # Only normalize fluorescence PSFs

    # Step 4: Convolve with PSF (at input resolution, which may be supersampled)
    convolved = convolve_rescale(
        expanded_scene, kernel,
        rescale_factor=1.0,
        rescale_int=False,
    )

    # Step 5: Crop to original scene size BEFORE normalization
    # This ensures contrast is computed over the actual scene, not the
    # padding region which is dominated by uniform media.
    h_input, w_input = scene_opl.shape
    eh, ew = expanded_scene.shape
    y_off = (eh - h_input) // 2
    x_off = (ew - w_input) // 2
    cropped = convolved[y_off:y_off + h_input, x_off:x_off + w_input].copy()
    output_masks = expanded_masks[y_off:y_off + h_input, x_off:x_off + w_input]

    # Step 6: Apply illumination gradient (at input resolution)
    cropped = apply_illumination_gradient(
        cropped, config.halo_top_intensity, config.halo_bottom_intensity,
    )

    # Step 6b: Apply device edge halo (diffraction fringe at PDMS/media boundary)
    if config.edge_halo_width > 0 and config.edge_halo_intensity > 0:
        # Crop the device mask to match the scene region
        device_cropped = device_mask[:h_input, :w_input]
        edge_halo = compute_device_edge_halo(
            device_cropped,
            halo_width=config.edge_halo_width,
            halo_intensity=config.edge_halo_intensity,
        )
        cropped = cropped + edge_halo

    # Step 7: Downscale to native resolution if supersampled
    if supersampling > 1:
        cropped = rescale(
            cropped, 1.0 / supersampling,
            anti_aliasing=True, preserve_range=True,
        )
        output_masks = rescale(
            output_masks.astype(float), 1.0 / supersampling,
            order=0, anti_aliasing=False, preserve_range=True,
        ).astype(np.int32)

    # Step 8: Rescale intensity (configurable)
    # When normalize_output=True (default), rescale to [0, 1] as before.
    # When False, preserve the physical intensity scale from the OPL/PSF pipeline.
    # intensity_range overrides the auto-detected (min, max) for rescaling.
    if config.normalize_output:
        if config.intensity_range is not None:
            cropped = rescale_intensity(
                cropped.astype(np.float64),
                in_range=tuple(config.intensity_range),
                out_range=(0.0, 1.0),
            )
        else:
            cropped = rescale_intensity(cropped.astype(np.float64), out_range=(0.0, 1.0))

    # Step 8b: Add Perlin noise background (for colony/agar pad rendering)
    # Applied AFTER rescale_intensity so the noise is visible on the [0,1] scale.
    if perlin_background is not None:
        from symbac.imaging.noise import apply_perlin_to_phase_contrast
        cropped = apply_perlin_to_phase_contrast(
            cropped, background=perlin_background,
            attenuation=perlin_attenuation,
            blur_sigma=perlin_blur_sigma,
            rng=rng,
        )

    # Step 9: Histogram matching
    if match_histogram and real_image is not None:
        from skimage.transform import resize
        real_resized = resize(real_image, cropped.shape[:2], anti_aliasing=True, preserve_range=True)
        real_resized = real_resized / (real_resized.max() + 1e-10)
        cropped = match_histograms(cropped, real_resized).astype(np.float64)

    # Step 10: Apply noise (at native resolution)
    if camera is not None:
        noisy = camera.apply_noise(cropped, rng=rng)
    elif config.noise_var > 0:
        if config.normalize_output:
            noisy = random_noise(
                rescale_intensity(cropped.astype(np.float64), out_range=(0.0, 1.0)),
                mode="gaussian", mean=0, var=config.noise_var, clip=False, rng=rng,
            )
        else:
            # Scale noise variance relative to signal range
            signal_range = cropped.max() - cropped.min()
            if signal_range > 0:
                noise = rng.normal(0, np.sqrt(config.noise_var) * signal_range, size=cropped.shape)
            else:
                noise = rng.normal(0, np.sqrt(config.noise_var), size=cropped.shape)
            noisy = cropped + noise
    else:
        noisy = cropped

    # Final normalize (configurable)
    if config.normalize_output:
        output_image = rescale_intensity(noisy.astype(np.float64), out_range=(0.0, 1.0))
    else:
        output_image = noisy.astype(np.float64)

    return output_image, output_masks


def render_fluorescence(
    scene_opl: np.ndarray,
    scene_masks: np.ndarray,
    psf_model: PSFModel,
    config: RenderConfig,
    fl_density: float = 1.0,
    supersampling: int = 1,
    camera: Optional[Camera] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a fluorescence image with optional molecule sampling.

    For fluorescence, the OPL represents cell thickness/volume. If fl_density
    is provided, fluorescent molecules are sampled proportionally to OPL,
    creating realistic punctate fluorescence patterns.

    Args:
        scene_opl: Raw OPL scene (at supersampled resolution if supersampling > 1).
        scene_masks: Instance masks (same resolution as scene_opl).
        psf_model: Fluorescence PSF model.
        config: Render config (cell_multiplier, defocus, noise_var used).
        fl_density: Fluorescent molecule density (molecules per OPL unit).
        supersampling: If > 1, inputs are supersampled by this factor.
        camera: Optional camera noise model.
        rng: Optional random generator.

    Returns:
        Tuple of (fluorescence_image, output_masks) at native resolution.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample fluorescent molecules from OPL distribution
    fl_scene = opl_to_fluorescence(scene_opl, fl_density, rng=rng)

    # Use a zero device mask (no device in fluorescence)
    device_mask = np.zeros_like(scene_opl, dtype=np.uint8)

    return render_image(
        fl_scene, scene_masks, device_mask, psf_model, config,
        supersampling=supersampling, camera=camera, rng=rng,
    )


def opl_to_fluorescence(
    opl: np.ndarray,
    density: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convert OPL image to fluorescence by sampling molecules.

    Each pixel's OPL value is treated as a probability weight for placing
    fluorescent molecules. The result is a sparse image with shot noise
    characteristic of real fluorescence microscopy.

    Args:
        opl: OPL image (pixel values = optical path length).
        density: Number of molecules per unit OPL volume.
        rng: Random generator.

    Returns:
        Fluorescence image (sparse intensity).
    """
    if rng is None:
        rng = np.random.default_rng()

    total_opl = opl.sum()
    if total_opl == 0:
        return np.zeros_like(opl)

    n_molecules = int(density * total_opl)
    if n_molecules == 0:
        return np.zeros_like(opl)

    # Normalize to probability distribution
    prob = opl.ravel() / total_opl

    # Sample molecule positions
    positions = rng.choice(len(prob), size=n_molecules, p=prob)

    fl_image = np.zeros(opl.size, dtype=np.float64)
    np.add.at(fl_image, positions, 1.0)

    return fl_image.reshape(opl.shape)
