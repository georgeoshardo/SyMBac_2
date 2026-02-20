"""Point Spread Function (PSF) models and Camera noise simulation.

Provides physically-based PSF generation for phase contrast and fluorescence
microscopy, as well as a camera noise model for realistic image synthesis.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jv


class PSFMode(Enum):
    PHASE_CONTRAST = "phase_contrast"
    FLUORESCENCE_2D = "fluorescence_2d"
    FLUORESCENCE_3D = "fluorescence_3d"


# Standard phase contrast condenser parameters: (W, R, diameter) in mm
CONDENSERS = {
    "Ph1": (0.45, 3.75, 24),
    "Ph2": (0.8, 5.0, 24),
    "Ph3": (1.0, 9.5, 24),
    "Ph4": (1.5, 14.0, 24),
    "PhF": (1.5, 19.0, 25),
}


class PSFModel:
    """Generates point spread function kernels for microscopy simulation.

    Supports phase contrast, 2D fluorescence (Airy disk), and 3D fluorescence
    (via psfmodels library) modes.

    The kernel is lazily computed on first access via the `kernel` property.

    Examples:
        >>> psf = PSFModel.phase_contrast(wavelength=0.75, NA=1.2, n=1.3,
        ...     condenser="Ph3", apo_sigma=10, radius=50, pixel_scale=0.02)
        >>> psf.kernel.shape
        (101, 101)

        >>> psf = PSFModel.fluorescence_2d(wavelength=0.5, NA=1.4, n=1.5,
        ...     radius=50, pixel_scale=0.02)
        >>> psf.kernel.shape
        (101, 101)
    """

    def __init__(
        self,
        mode: PSFMode,
        wavelength: float,
        NA: float,
        n: float,
        radius: int,
        pixel_scale: float,
        apo_sigma: float = 10.0,
        condenser: Optional[str] = None,
        z_height: Optional[int] = None,
        pz: float = 0.0,
        working_distance: Optional[float] = None,
        offset: float = 0.0,
    ):
        """Initialize PSF model. Prefer using factory methods instead.

        Args:
            mode: The PSF mode (phase contrast, fluorescence 2D/3D).
            wavelength: Wavelength of imaging light in microns.
            NA: Numerical aperture of the objective lens.
            n: Refractive index of the imaging medium.
            radius: Radius of the PSF kernel in pixels.
            pixel_scale: Microns per pixel at the rendering resolution.
            apo_sigma: Gaussian apodisation sigma for phase contrast.
            condenser: Condenser type (Ph1-Ph4, PhF) for phase contrast.
            z_height: Z-size for 3D fluorescence PSF.
            pz: Particle z-position for 3D PSF.
            working_distance: Working distance for 3D PSF.
            offset: Constant offset added to the PSF kernel.
        """
        self.mode = mode
        self.wavelength = wavelength
        self.NA = NA
        self.n = n
        self.radius = radius
        self.pixel_scale = pixel_scale
        self.apo_sigma = apo_sigma
        self.condenser = condenser
        self.z_height = z_height
        self.pz = pz
        self.working_distance = working_distance
        self.offset = offset

        if condenser is not None:
            if condenser not in CONDENSERS:
                raise ValueError(f"Unknown condenser '{condenser}'. Must be one of: {list(CONDENSERS.keys())}")
            self.W, self.R, self.diameter = CONDENSERS[condenser]

        self._kernel = None

    @classmethod
    def phase_contrast(
        cls,
        wavelength: float = 0.75,
        NA: float = 1.2,
        n: float = 1.3,
        condenser: str = "Ph3",
        apo_sigma: float = 10.0,
        radius: int = 50,
        pixel_scale: float = 0.02,
        offset: float = 0.0,
    ) -> PSFModel:
        """Create a phase contrast PSF model.

        Args:
            wavelength: Wavelength of imaging light in microns.
            NA: Numerical aperture.
            n: Refractive index.
            condenser: Condenser ring type (Ph1, Ph2, Ph3, Ph4, PhF).
            apo_sigma: Gaussian apodisation sigma in pixels.
            radius: PSF kernel radius in pixels.
            pixel_scale: Microns per pixel at the rendering resolution.
            offset: Constant offset for long-range effects.
        """
        return cls(
            mode=PSFMode.PHASE_CONTRAST,
            wavelength=wavelength,
            NA=NA,
            n=n,
            radius=radius,
            pixel_scale=pixel_scale,
            apo_sigma=apo_sigma,
            condenser=condenser,
            offset=offset,
        )

    @classmethod
    def fluorescence_2d(
        cls,
        wavelength: float = 0.5,
        NA: float = 1.4,
        n: float = 1.5,
        radius: int = 50,
        pixel_scale: float = 0.02,
        offset: float = 0.0,
    ) -> PSFModel:
        """Create a 2D fluorescence PSF model (Airy disk).

        Args:
            wavelength: Emission wavelength in microns.
            NA: Numerical aperture.
            n: Refractive index.
            radius: PSF kernel radius in pixels.
            pixel_scale: Microns per pixel at the rendering resolution.
            offset: Constant offset.
        """
        return cls(
            mode=PSFMode.FLUORESCENCE_2D,
            wavelength=wavelength,
            NA=NA,
            n=n,
            radius=radius,
            pixel_scale=pixel_scale,
            offset=offset,
        )

    @classmethod
    def fluorescence_3d(
        cls,
        wavelength: float = 0.5,
        NA: float = 1.4,
        n: float = 1.5,
        radius: int = 50,
        pixel_scale: float = 0.02,
        z_height: int = 21,
        pz: float = 0.0,
        working_distance: Optional[float] = None,
        offset: float = 0.0,
    ) -> PSFModel:
        """Create a 3D fluorescence PSF model.

        Requires the `psfmodels` package.

        Args:
            wavelength: Emission wavelength in microns.
            NA: Numerical aperture.
            n: Refractive index.
            radius: PSF kernel radius in pixels (lateral).
            pixel_scale: Microns per pixel at the rendering resolution.
            z_height: Number of z-slices.
            pz: Particle z-position in microns.
            working_distance: Working distance of the objective.
            offset: Constant offset.
        """
        return cls(
            mode=PSFMode.FLUORESCENCE_3D,
            wavelength=wavelength,
            NA=NA,
            n=n,
            radius=radius,
            pixel_scale=pixel_scale,
            z_height=z_height,
            pz=pz,
            working_distance=working_distance,
            offset=offset,
        )

    @property
    def kernel(self) -> np.ndarray:
        """The PSF kernel array. Computed lazily on first access."""
        if self._kernel is None:
            self._compute_kernel()
        return self._kernel

    def _compute_kernel(self) -> None:
        """Compute the PSF kernel based on mode and parameters."""
        if self.mode == PSFMode.PHASE_CONTRAST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kernel = _phase_contrast_kernel(
                    R=self.R,
                    W=self.W,
                    radius=self.radius,
                    scale=self.pixel_scale,
                    NA=self.NA,
                    n=self.n,
                    sigma=self.apo_sigma,
                    wavelength=self.wavelength,
                    offset=self.offset,
                )

        elif self.mode == PSFMode.FLUORESCENCE_2D:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kernel = _fluorescence_kernel(
                    radius=self.radius,
                    scale=self.pixel_scale,
                    NA=self.NA,
                    n=self.n,
                    wavelength=self.wavelength,
                    offset=self.offset,
                )

        elif self.mode == PSFMode.FLUORESCENCE_3D:
            try:
                import psfmodels as psfm
            except ImportError:
                raise ImportError("psfmodels is required for 3D fluorescence PSF. Install with: pip install psfmodels")

            if self.z_height is None:
                raise ValueError("z_height must be specified for 3D fluorescence PSF")

            kwargs = dict(
                z=self.z_height,
                nx=self.radius * 2 + 1,
                dxy=self.pixel_scale,
                dz=self.pixel_scale,
                pz=self.pz,
                ni=self.n,
                ni0=self.n,
                wvl=self.wavelength,
                NA=self.NA,
                model="scalar",
            )
            if self.working_distance is not None:
                kwargs["ti0"] = self.working_distance

            self._kernel = psfm.make_psf(**kwargs) + self.offset

        self._kernel = self._kernel / self._kernel.max()

    @property
    def kernel_2d(self) -> np.ndarray:
        """Return a 2D kernel, averaging over z for 3D PSFs."""
        if self.mode == PSFMode.FLUORESCENCE_3D:
            return self.kernel.mean(axis=0)
        return self.kernel

    def plot(self, ax=None) -> None:
        """Plot the PSF kernel."""
        if ax is None:
            fig, ax = plt.subplots()

        if self.mode == PSFMode.FLUORESCENCE_3D:
            # Show middle z-slice
            mid = self.kernel.shape[0] // 2
            ax.imshow(self.kernel[mid], cmap="Greys_r")
            ax.set_title(f"3D Fluorescence PSF (z={mid})")
        else:
            ax.imshow(self.kernel, cmap="Greys_r")
            mode_name = "Phase Contrast" if self.mode == PSFMode.PHASE_CONTRAST else "Fluorescence"
            ax.set_title(f"{mode_name} PSF")
        ax.axis("off")


class Camera:
    """Camera noise model for synthetic microscopy image generation.

    Models the camera's baseline intensity, sensitivity (gain), and
    read noise (dark noise) to produce realistic noise patterns.

    Examples:
        >>> cam = Camera(baseline=100, sensitivity=2.9, dark_noise=8)
        >>> dark = cam.render_dark_image((512, 512), plot=False)
        >>> dark.shape
        (512, 512)
    """

    def __init__(self, baseline: float, sensitivity: float, dark_noise: float):
        """
        Args:
            baseline: The baseline intensity offset of the camera.
            sensitivity: The camera sensitivity (gain).
            dark_noise: Standard deviation of the read noise.
        """
        self.baseline = baseline
        self.sensitivity = sensitivity
        self.dark_noise = dark_noise

    def render_dark_image(
        self,
        size: tuple[int, int],
        plot: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Render a synthetic dark image.

        Args:
            size: (height, width) of the output image.
            plot: If True, display the image.
            rng: Optional numpy random generator for reproducibility.

        Returns:
            Dark image as a 2D numpy array.
        """
        if rng is None:
            rng = np.random.default_rng()
        dark_img = rng.normal(loc=self.baseline, scale=self.dark_noise, size=size)
        dark_img = rng.poisson(np.maximum(dark_img, 0))

        if plot:
            plt.imshow(dark_img, cmap="Greys_r")
            plt.colorbar()
            plt.axis("off")
            plt.title("Dark Image")
            plt.show()
        return dark_img

    def apply_noise(
        self,
        image: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply camera noise model to an image.

        Scales image by sensitivity, adds Gaussian read noise with baseline.

        Args:
            image: Input image (should be in physical intensity units).
            rng: Optional numpy random generator.

        Returns:
            Noisy image.
        """
        if rng is None:
            rng = np.random.default_rng()

        noisy = image / self.sensitivity
        noisy = noisy + rng.normal(loc=self.baseline, scale=self.dark_noise, size=image.shape)
        return noisy


# ---- Static kernel computation functions ----

def _gaussian_2d(size: int, sigma: float) -> np.ndarray:
    """2D Gaussian kernel."""
    x = np.linspace(0, size, size)
    mu = np.mean(x)
    A = 1 / (sigma * np.sqrt(2 * np.pi))
    B = np.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))
    g1d = A * B
    return np.outer(g1d, g1d)


def _fluorescence_kernel(
    wavelength: float,
    NA: float,
    n: float,
    radius: int,
    scale: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Airy disk approximation of fluorescence PSF.

    Args:
        wavelength: Wavelength in microns.
        NA: Numerical aperture.
        n: Refractive index.
        radius: Kernel radius in pixels.
        scale: Pixel size in microns.
        offset: Constant offset.
    """
    r = np.arange(-radius, radius + 1)
    kaw = 2 * NA / n * np.pi / wavelength
    xx, yy = np.meshgrid(r, r)
    xx, yy = xx * scale, yy * scale
    rr = np.sqrt(xx ** 2 + yy ** 2) * kaw
    PSF = (2 * jv(1, rr) / rr) ** 2
    PSF[radius, radius] = 1
    PSF += offset
    return PSF


def _phase_contrast_kernel(
    R: float,
    W: float,
    radius: int,
    scale: float,
    NA: float,
    n: float,
    sigma: float,
    wavelength: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Phase contrast PSF kernel.

    Models the PSF as the difference between an Airy disk and an obscured
    annular aperture, with Gaussian apodisation.

    Args:
        R: Condenser ring radius in mm.
        W: Condenser ring width in mm.
        radius: Kernel radius in pixels.
        scale: Pixel size in microns.
        NA: Numerical aperture.
        n: Refractive index.
        sigma: Apodisation Gaussian sigma in pixels.
        wavelength: Wavelength in microns.
        offset: Constant offset.
    """
    gaussian = _gaussian_2d(radius * 2 + 1, sigma)

    scale1 = 1000  # micron per millimeter
    Lambda = wavelength
    R = R * scale1
    W = W * scale1

    r = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(r, r)
    xx, yy = xx * scale, yy * scale
    kaw = 2 * NA / n * np.pi / Lambda
    rr = np.sqrt(xx ** 2 + yy ** 2) * kaw

    kernel1 = 2 * jv(1, rr) / rr
    kernel1[radius, radius] = 1

    kernel2 = 2 * (R - W) ** 2 / R ** 2 * jv(1, (R - W) ** 2 / R ** 2 * rr) / rr
    kernel2[radius, radius] = np.nanmax(kernel2)

    kernel1 *= gaussian
    kernel2 *= gaussian

    kernel = kernel1 - kernel2
    kernel = kernel / np.max(kernel)
    kernel[radius, radius] = 1

    if np.sum(kernel1) > np.sum(kernel2):
        kernel = -kernel / np.sum(kernel)
    else:
        kernel = kernel / np.sum(kernel)

    kernel += offset
    return kernel
