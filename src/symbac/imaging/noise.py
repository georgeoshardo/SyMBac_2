"""Procedural noise generation for synthetic microscopy backgrounds.

Generates Perlin/simplex noise textures used to simulate agar pad and
substrate backgrounds in colony microscopy images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class NoiseConfig:
    """Configuration for procedural noise generation.

    Default values are tuned to produce agar pad textures at
    typical microscopy resolutions.
    """
    scale: float = 5.0
    octaves: int = 10
    persistence: float = 1.9
    lacunarity: float = 1.8
    seed: int = 42


def generate_perlin_background(
    shape: tuple[int, int],
    config: Optional[NoiseConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a 2D Perlin/simplex noise background texture.

    Produces a smooth noise texture suitable for agar pad backgrounds in
    colony microscopy images.

    Args:
        shape: (height, width) of the output image.
        config: Noise configuration. If None, uses defaults.
        rng: Random generator (used for random config if config is None).

    Returns:
        2D array with values in [0, 1] range.
    """
    if config is None:
        config = NoiseConfig()

    try:
        return _generate_opensimplex(shape, config)
    except ImportError:
        return _generate_numpy_perlin(shape, config)


def random_perlin_background(
    shape: tuple[int, int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a Perlin background with randomized parameters.

    Varies the noise parameters to produce diverse agar pad textures
    for training data generation.

    Args:
        shape: (height, width) of the output image.
        rng: Random generator for parameter selection.

    Returns:
        2D array with values in [0, 1] range.
    """
    if rng is None:
        rng = np.random.default_rng()

    config = NoiseConfig(
        scale=rng.uniform(1.0, 7.0),
        octaves=rng.choice([10, 11, 12, 13]),
        persistence=rng.uniform(1.0, 1.9),
        lacunarity=rng.uniform(1.55, 1.9),
        seed=int(rng.integers(0, 2**31)),
    )
    return generate_perlin_background(shape, config, rng)


def apply_perlin_to_phase_contrast(
    image: np.ndarray,
    background: Optional[np.ndarray] = None,
    attenuation: Optional[float] = None,
    blur_sigma: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add a Perlin noise background to a phase contrast image.

    This simulates the texture of an agar pad substrate visible in
    colony phase contrast microscopy. The noise is rotated, attenuated,
    blurred, and added to the image (which should already be in [0,1] range).

    The old SyMBac used uint8 noise (0-255) divided by 500-1500, which gives
    an additive range of about 0.0-0.5 on a [0,1] image. Our noise is already
    in [0,1], so the attenuation defaults are scaled accordingly (2-6).

    Args:
        image: The phase contrast image in [0, 1] range (after rescale_intensity).
        background: Pre-generated Perlin background in [0, 1], or None to generate.
        attenuation: Divide noise by this factor (default: random 2-6).
            Lower values = stronger background texture.
        blur_sigma: Gaussian blur sigma for the noise (default: random 1-3).
        rng: Random generator.

    Returns:
        Image with agar pad background texture added.
    """
    if rng is None:
        rng = np.random.default_rng()

    if background is None:
        background = random_perlin_background(image.shape, rng)

    if attenuation is None:
        attenuation = rng.uniform(2.0, 6.0)

    if blur_sigma is None:
        blur_sigma = rng.uniform(1.0, 3.0)

    # Rotate background randomly
    n_rotations = rng.integers(0, 4)
    bg = np.rot90(background, k=n_rotations)

    # Resize to match image if shapes differ after rotation
    if bg.shape != image.shape:
        from skimage.transform import resize
        bg = resize(bg, image.shape, anti_aliasing=True, preserve_range=True)

    # Attenuate and blur
    bg_processed = gaussian_filter(bg / attenuation, blur_sigma, mode="reflect")

    return image + bg_processed


def _generate_opensimplex(
    shape: tuple[int, int],
    config: NoiseConfig,
) -> np.ndarray:
    """Generate noise using the opensimplex library."""
    from opensimplex import OpenSimplex

    gen = OpenSimplex(seed=config.seed)
    h, w = shape

    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    # Multi-octave simplex noise
    result = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0 / config.scale

    for _ in range(config.octaves):
        noise_grid = np.vectorize(gen.noise2)(
            xx * frequency, yy * frequency
        )
        result += amplitude * noise_grid
        amplitude *= config.persistence
        frequency *= config.lacunarity

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    return result


def _generate_numpy_perlin(
    shape: tuple[int, int],
    config: NoiseConfig,
) -> np.ndarray:
    """Fallback Perlin-like noise using pure numpy (value noise with interpolation)."""
    rng = np.random.default_rng(config.seed)
    h, w = shape

    result = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0 / config.scale

    for _ in range(config.octaves):
        # Grid size for this octave
        gh = max(2, int(h * frequency) + 2)
        gw = max(2, int(w * frequency) + 2)

        # Random grid values
        grid = rng.random((gh, gw))

        # Bicubic interpolation to target size
        from scipy.ndimage import zoom
        scale_y = h / gh
        scale_x = w / gw
        interpolated = zoom(grid, (scale_y, scale_x), order=3, mode="wrap")

        # Crop to exact size
        interpolated = interpolated[:h, :w]

        result += amplitude * interpolated
        amplitude *= config.persistence
        frequency *= config.lacunarity

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    return result
