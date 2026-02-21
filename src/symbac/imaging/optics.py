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


from dataclasses import dataclass


class PSFMode(Enum):
    PHASE_CONTRAST = "phase_contrast"
    FLUORESCENCE_2D = "fluorescence_2d"
    FLUORESCENCE_3D = "fluorescence_3d"


@dataclass
class CellOpticsConfig:
    """Refractive index configuration for two-layer cell OPL model.

    Models a bacterium as a peptidoglycan wall (higher RI) surrounding
    cytoplasm (lower RI), both immersed in growth medium. The OPL at
    each pixel is:

        OPL(x,y) = (n_wall - n_medium) * wall_thickness(x,y)
                  + (n_cytoplasm - n_medium) * cytoplasm_thickness(x,y)

    This produces a bright rim and dimmer interior characteristic of
    phase contrast images of bacteria.

    Args:
        n_medium: Refractive index of growth medium (default: 1.33, water).
        n_wall: Refractive index of peptidoglycan wall (default: 1.45).
        n_cytoplasm: Refractive index of cytoplasm (default: 1.39).
        wall_fraction: Fraction of cell radius occupied by the wall
            (default: 0.1, i.e. wall is 10% of the radius thick).
    """
    n_medium: float = 1.33
    n_wall: float = 1.45
    n_cytoplasm: float = 1.39
    wall_fraction: float = 0.1


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

        The image is expected to be in [0, 1] range. It is:
        1. Scaled up to physical detector counts (baseline * sensitivity)
        2. Poisson shot noise is applied
        3. Gaussian read noise is added

        Args:
            image: Input image in [0, 1] range.
            rng: Optional numpy random generator.

        Returns:
            Noisy image in physical intensity units.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Scale to detector count range
        signal = image * self.baseline * self.sensitivity
        signal = np.maximum(signal, 0)
        # Poisson shot noise
        noisy = rng.poisson(signal).astype(np.float64)
        # Gaussian read noise
        noisy = noisy + rng.normal(loc=0, scale=self.dark_noise, size=image.shape)
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


def apply_defocus_pupil(
    kernel: np.ndarray,
    defocus_um: float,
    wavelength: float,
    NA: float,
    n: float,
    pixel_scale: float,
) -> np.ndarray:
    """Apply defocus via a pupil-function phase term in Fourier space.

    Instead of blurring the PSF with a Gaussian (which has no physical basis),
    this applies the optical defocus aberration:

        W(rho) = defocus_um * (1 - sqrt(1 - (rho * NA / n)^2))

    where rho is the normalised pupil coordinate. The PSF is Fourier-transformed,
    multiplied by exp(i * 2pi/lambda * W(rho)), and transformed back.

    Args:
        kernel: 2D PSF kernel in spatial domain.
        defocus_um: Defocus distance in microns. 0 = in-focus.
        wavelength: Imaging wavelength in microns.
        NA: Numerical aperture.
        n: Refractive index of the immersion medium.
        pixel_scale: Microns per pixel at the kernel's resolution.

    Returns:
        Defocused PSF kernel (same shape as input).
    """
    if defocus_um == 0:
        return kernel.copy()

    h, w = kernel.shape
    # Spatial frequency coordinates (cycles per micron)
    fy = np.fft.fftfreq(h, d=pixel_scale)
    fx = np.fft.fftfreq(w, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fy)
    rho_freq = np.sqrt(FX**2 + FY**2)

    # Normalise to pupil coordinate: rho_pupil in [0, 1] maps to [0, NA/wavelength]
    # The cutoff spatial frequency is NA / wavelength
    cutoff = NA / wavelength
    rho_norm = rho_freq / cutoff  # normalised pupil radius

    # Compute defocus wavefront error (only inside the pupil)
    # W(rho) = defocus_um * (1 - sqrt(1 - (rho * NA / n)^2))
    # Using the exact Debye model for defocus
    sin_alpha = rho_norm * NA / n
    # Clip to valid range (inside pupil: sin_alpha <= 1)
    valid = sin_alpha < 1.0
    W = np.zeros_like(rho_norm)
    W[valid] = defocus_um * (1.0 - np.sqrt(1.0 - sin_alpha[valid]**2))

    # Phase transfer function
    phase = np.exp(1j * 2 * np.pi / wavelength * W)

    # Apply in Fourier domain: zero outside pupil
    pupil_mask = rho_norm <= 1.0
    H_defocus = np.zeros_like(phase)
    H_defocus[pupil_mask] = phase[pupil_mask]

    # Transform PSF to OTF, apply defocus, transform back
    OTF = np.fft.fft2(kernel)
    OTF_defocused = OTF * H_defocus
    kernel_defocused = np.real(np.fft.ifft2(OTF_defocused))

    return kernel_defocused
