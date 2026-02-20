"""Training data generation for cell segmentation models.

Renders batches of synthetic microscopy images with parameter variation,
producing paired (image, mask) datasets for training segmentation networks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms, rescale_intensity
from skimage.transform import resize
from tqdm import tqdm

from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig, render_image


@dataclass
class TrainingDataConfig:
    """Configuration for training data generation.

    Controls parameter variation, image matching, and output settings.
    """
    n_samples: int = 500
    burn_in: int = 40
    sample_variation: float = 0.05
    in_series: bool = False
    match_histogram: bool = True
    match_fourier: bool = False
    randomize_histogram: bool = True
    randomize_fourier: bool = False
    mask_dtype: type = np.uint8
    seed: Optional[int] = None


def generate_training_data(
    opl_scenes: list[np.ndarray],
    mask_scenes: list[np.ndarray],
    device_masks: list[np.ndarray],
    psf_model: PSFModel,
    render_config: RenderConfig,
    training_config: TrainingDataConfig,
    supersampling: int = 1,
    camera: Optional[Camera] = None,
    real_image: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    prefix: str = "",
    n_jobs: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate a batch of synthetic training image/mask pairs.

    Takes pre-computed OPL scenes and masks (e.g. from running a simulation
    and calling draw_scene_supersampled() each frame), then renders each
    with parameter variations to produce diverse training data.

    Args:
        opl_scenes: List of OPL scenes (one per simulation frame).
        mask_scenes: List of instance mask arrays (same indexing as opl_scenes).
        device_masks: List of device mask arrays (same indexing).
        psf_model: PSF model for convolution.
        render_config: Base rendering configuration. Parameters are varied
            around these base values.
        training_config: Training data generation configuration.
        supersampling: If > 1, OPL scenes are at supersampled resolution.
        camera: Optional camera noise model.
        real_image: Optional real image for histogram matching.
        save_dir: If provided, save images and masks to this directory.
        prefix: Filename prefix for saved images.
        n_jobs: Number of parallel workers (1 = sequential).

    Returns:
        List of (synthetic_image, mask) tuples.
    """
    tc = training_config
    rng = np.random.default_rng(tc.seed)

    n_frames = len(opl_scenes)
    if n_frames == 0:
        raise ValueError("No OPL scenes provided.")

    # Determine which frames to render
    valid_start = min(tc.burn_in, n_frames - 1)
    valid_end = n_frames

    if tc.in_series:
        # Sequential frames with same parameters per series
        series_len = valid_end - valid_start
        n_series = max(1, int(np.ceil(tc.n_samples / series_len)))
        scene_indices = []
        for _ in range(n_series):
            scene_indices.extend(range(valid_start, valid_end))
        scene_indices = scene_indices[:tc.n_samples]
    else:
        # Random frame selection
        scene_indices = rng.integers(valid_start, valid_end, size=tc.n_samples)

    # Generate varied parameters
    def vary(base_val):
        v = tc.sample_variation
        return rng.uniform(1 - v, 1 + v, size=tc.n_samples) * base_val

    media_mults = vary(render_config.media_multiplier)
    cell_mults = vary(render_config.cell_multiplier)
    device_mults = vary(render_config.device_multiplier)
    noise_vars = vary(render_config.noise_var)
    defocuses = vary(render_config.defocus)

    # For in_series mode, repeat parameters per series
    if tc.in_series:
        series_len = valid_end - valid_start
        n_series = max(1, int(np.ceil(tc.n_samples / series_len)))
        media_mults = np.repeat(
            rng.uniform(1 - tc.sample_variation, 1 + tc.sample_variation, size=n_series)
            * render_config.media_multiplier,
            series_len,
        )[:tc.n_samples]
        cell_mults = np.repeat(
            rng.uniform(1 - tc.sample_variation, 1 + tc.sample_variation, size=n_series)
            * render_config.cell_multiplier,
            series_len,
        )[:tc.n_samples]
        device_mults = np.repeat(
            rng.uniform(1 - tc.sample_variation, 1 + tc.sample_variation, size=n_series)
            * render_config.device_multiplier,
            series_len,
        )[:tc.n_samples]

    # Matching toggles
    if tc.randomize_histogram:
        hist_match_flags = rng.choice([True, False], size=tc.n_samples)
    else:
        hist_match_flags = np.full(tc.n_samples, tc.match_histogram)

    if tc.randomize_fourier:
        fourier_match_flags = rng.choice([True, False], size=tc.n_samples)
    else:
        fourier_match_flags = np.full(tc.n_samples, tc.match_fourier)

    # Prepare save directories
    if save_dir is not None:
        dirs = _create_save_dirs(save_dir)

    # Render samples
    def render_single(idx):
        i = int(scene_indices[idx])
        sample_rng = np.random.default_rng(rng.integers(0, 2**31) + idx)

        # Build varied config for this sample
        cfg = RenderConfig(
            media_multiplier=float(media_mults[idx]),
            cell_multiplier=float(cell_mults[idx]),
            device_multiplier=float(device_mults[idx]),
            defocus=float(defocuses[idx]),
            noise_var=float(noise_vars[idx]),
            border_expansion=render_config.border_expansion,
            halo_top_intensity=render_config.halo_top_intensity,
            halo_bottom_intensity=render_config.halo_bottom_intensity,
        )

        do_hist_match = bool(hist_match_flags[idx]) and real_image is not None

        synth, mask_out = render_image(
            opl_scenes[i], mask_scenes[i], device_masks[i],
            psf_model, cfg,
            supersampling=supersampling,
            camera=camera,
            real_image=real_image,
            match_histogram=do_hist_match,
            rng=sample_rng,
        )

        # Fourier matching
        if bool(fourier_match_flags[idx]) and real_image is not None:
            synth = _fourier_match(synth, real_image)

        mask_out = mask_out.astype(tc.mask_dtype)

        if save_dir is not None:
            _save_sample(dirs, prefix, idx, synth, mask_out)

        return synth, mask_out

    if n_jobs == 1:
        results = [
            render_single(idx)
            for idx in tqdm(range(tc.n_samples), desc="Rendering training data")
        ]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(render_single)(idx)
            for idx in tqdm(range(tc.n_samples), desc="Rendering training data")
        )

    return results


def record_simulation_frames(
    simulator,
    draw_fn,
    n_steps: int,
    record_every: int = 1,
    desc: str = "Recording frames",
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run a simulation and record OPL/mask/device frames at intervals.

    This is a convenience function for building the input lists needed
    by generate_training_data().

    Args:
        simulator: The Simulator instance (already initialized with hooks).
        draw_fn: A callable(simulator) -> (opl, masks, device) that draws
            the current frame. Typically a lambda wrapping draw_scene_supersampled
            or draw_scene_with_geometry.
        n_steps: Number of simulation steps to run.
        record_every: Record a frame every N steps.
        desc: Description for progress bar.

    Returns:
        Tuple of (opl_scenes, mask_scenes, device_masks) lists.
    """
    opl_scenes = []
    mask_scenes = []
    device_masks = []

    for step in tqdm(range(n_steps), desc=desc):
        simulator.step()
        if step % record_every == 0:
            opl, masks, device = draw_fn(simulator)
            opl_scenes.append(opl)
            mask_scenes.append(masks)
            device_masks.append(device)

    return opl_scenes, mask_scenes, device_masks


def _create_save_dirs(save_dir: str) -> dict[str, Path]:
    """Create output directory structure."""
    base = Path(save_dir)
    dirs = {
        "convolutions": base / "convolutions",
        "masks": base / "masks",
        "opl_scenes": base / "opl_scenes",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_sample(
    dirs: dict[str, Path],
    prefix: str,
    idx: int,
    synth: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Save a single training sample to disk."""
    # Save synthetic image as 16-bit TIFF
    synth_16 = rescale_intensity(
        synth.astype(np.float64), out_range=(0, 65535)
    ).astype(np.uint16)
    Image.fromarray(synth_16).save(
        dirs["convolutions"] / f"{prefix}synth_{idx:05d}.tiff"
    )

    # Save mask as PNG
    Image.fromarray(mask).save(
        dirs["masks"] / f"{prefix}mask_{idx:05d}.png"
    )


def _fourier_match(synth: np.ndarray, real_image: np.ndarray) -> np.ndarray:
    """Apply Fourier spectrum matching to a synthetic image."""
    try:
        from symbac.imaging.spectral_matching import match_fourier_spectrum, match_luminance
    except ImportError:
        return synth

    real_resized = resize(
        real_image, synth.shape[:2],
        anti_aliasing=True, preserve_range=True,
    )
    real_norm = real_resized / (real_resized.max() + 1e-10)

    matched = match_fourier_spectrum([real_norm, synth])[1]
    matched = match_luminance(
        [real_norm, matched],
        target_lum=[np.mean(real_norm), np.std(real_norm)],
    )[1]
    return matched
