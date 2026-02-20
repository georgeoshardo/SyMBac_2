"""Spectral and luminance matching for synthetic microscopy images.

Ported from the SHINE toolbox implementation (pySHINE.py).

References:
    Willenbockel, V., Sadr, J., Fiset, D. et al. Controlling low-level
    image properties: The SHINE toolbox. Behavior Research Methods 42,
    671-684 (2010). https://doi.org/10.3758/BRM.42.3.671
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from numpy import fft


def _cart2pol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian to polar coordinates."""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return phi, rho


def _pol2cart(phi: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar to Cartesian coordinates."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def match_fourier_spectrum(
    images: list[np.ndarray],
    target_magnitude: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Match the rotational Fourier spectrum of a stack of images.

    If no target magnitude is provided, the average spectrum of all images
    is used as the target.

    Args:
        images: List of 2D numpy arrays (must all have the same shape).
        target_magnitude: Optional target Fourier magnitude spectrum.

    Returns:
        List of spectrally matched images.
    """
    if not isinstance(images, list):
        raise TypeError("Input must be a list of images.")

    numin = len(images)
    xs, ys = images[0].shape[:2]
    angs = np.zeros((xs, ys, numin))
    mags = np.zeros((xs, ys, numin))

    for i in range(numin):
        im = images[i]
        if len(im.shape) == 3:
            im = np.mean(im, axis=2)  # Convert to grayscale
        im = im / 255.0 if im.max() > 1 else im
        fftim = fft.fftshift(fft.fft2(im))
        angs[:, :, i], mags[:, :, i] = _cart2pol(np.real(fftim), np.imag(fftim))

    if target_magnitude is None:
        target_magnitude = np.mean(mags, axis=2)

    xt, yt = target_magnitude.shape
    assert (xs == xt) and (ys == yt), "Target spectrum must match image size."

    f1 = np.linspace(-ys / 2, ys / 2 - 1, ys)
    f2 = np.linspace(-xs / 2, xs / 2 - 1, xs)
    XX, YY = np.meshgrid(f1, f2)
    _, r = _cart2pol(XX, YY)
    if xs % 2 == 1 or ys % 2 == 1:
        r = np.round(r) - 1
    else:
        r = np.round(r)

    output_images = []
    for x in range(numin):
        fftim = mags[:, :, x]
        a = fftim.T.ravel()
        accmap = r.T.ravel() + 1
        a2 = target_magnitude.T.ravel()
        unique_acc = np.unique(accmap)

        en_old = np.array([np.sum(a[accmap == z]) for z in unique_acc])
        en_new = np.array([np.sum(a2[accmap == z]) for z in unique_acc])

        with np.errstate(divide='ignore', invalid='ignore'):
            coefficient = np.where(en_old != 0, en_new / en_old, 0)

        cmat = coefficient[r.astype(int)]
        cmat[r > np.floor(np.max((xs, ys)) / 2)] = 0
        newmag = fftim * cmat
        XX, YY = _pol2cart(angs[:, :, x], newmag)
        new = XX + YY * 1j
        output = np.real(fft.ifft2(fft.ifftshift(new)))
        output_images.append(output * 255)

    return output_images


def match_luminance(
    images: list[np.ndarray],
    mask: Optional[list[np.ndarray]] = None,
    target_lum: Optional[tuple[float, float]] = None,
) -> list[np.ndarray]:
    """Match the luminosity (mean and std) of a stack of images.

    Args:
        images: List of 2D numpy arrays.
        mask: Optional list of binary masks (1 = region to match).
        target_lum: Optional tuple (mean, std) to match to. If None,
            uses the average across all images.

    Returns:
        List of luminance-matched images.
    """
    if not isinstance(images, list):
        raise TypeError("Input must be a list of images.")

    numin = len(images)

    if target_lum is not None:
        M, S = target_lum
    else:
        M, S = 0.0, 0.0
        for im in images:
            if len(im.shape) == 3:
                im = np.mean(im, axis=2)
            if mask is None:
                M += np.mean(im)
                S += np.std(im)
            else:
                m = mask[0] if len(mask) == 1 else mask[images.index(im)]
                M += np.mean(im[m == 1])
                S += np.std(im[m == 1])
        M /= numin
        S /= numin

    output_images = []
    for i in range(numin):
        im = copy.deepcopy(images[i])
        if len(im.shape) == 3:
            im = np.mean(im, axis=2)

        if mask is None or len(mask) == 0:
            if np.std(im) != 0:
                im = (im - np.mean(im)) / np.std(im) * S + M
            else:
                im[:, :] = M
        else:
            m = mask[i] if len(mask) > 1 else mask[0]
            if np.std(im[m == 1]) != 0:
                im[m == 1] = (im[m == 1] - np.mean(im[m == 1])) / np.std(im[m == 1]) * S + M
            else:
                im[m == 1] = M

        output_images.append(im)
    return output_images
