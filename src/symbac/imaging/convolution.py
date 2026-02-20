"""Convolution and rescaling operations for image synthesis.

Provides CPU and optional GPU (CuPy) convolution with downsampling,
used to apply PSF kernels to OPL scenes.
"""

from __future__ import annotations

import warnings

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

# Detect CuPy availability
_cupy_available = False
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cuconvolve
    _cupy_available = True
except ImportError:
    pass


def convolve_rescale(
    image: np.ndarray,
    kernel: np.ndarray,
    rescale_factor: float = 1.0,
    rescale_int: bool = True,
    backend: str = "auto",
) -> np.ndarray:
    """Convolve an image with a kernel and rescale to target resolution.

    The image and kernel are assumed to be at supersampled resolution.
    After convolution, the image is downsampled by rescale_factor.

    Args:
        image: 2D input image (supersampled resolution).
        kernel: 2D convolution kernel (PSF).
        rescale_factor: Downsample factor (e.g. 1/3 for 3x supersampling).
        rescale_int: If True, rescale output intensities to [0, 1].
        backend: "auto" (detect GPU), "gpu" (force CuPy), or "cpu" (force scipy).

    Returns:
        Convolved and downsampled 2D image.
    """
    if backend == "auto":
        backend = "gpu" if _cupy_available else "cpu"

    if backend == "gpu":
        if not _cupy_available:
            warnings.warn("CuPy not available, falling back to CPU convolution.")
            backend = "cpu"
        else:
            output = cuconvolve(cp.array(image), cp.array(kernel), mode="nearest")
            output = output.get()

    if backend == "cpu":
        from scipy.signal import fftconvolve
        output = fftconvolve(image, kernel, mode="same")

    if rescale_factor != 1.0:
        output = rescale(output, rescale_factor, anti_aliasing=False, preserve_range=True)

    if rescale_int:
        output = rescale_intensity(output.astype(np.float32), out_range=(0, 1))

    return output
