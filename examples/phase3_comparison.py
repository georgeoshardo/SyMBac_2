"""Phase 3: Side-by-side comparison with real 100x phase contrast image.

Uses the correct SyMBac pipeline: convolve at supersampled resolution,
then downscale to native resolution.

Matches the real reference image in:
- Dimensions: 256 x 46 pixels
- Dynamic range: 16-bit (uint16, range ~591-7507)
- Features: single mother machine trench with E. coli
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymunk import Vec2d
from skimage.exposure import match_histograms, rescale_intensity
from skimage.transform import resize

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator
from symbac.imaging.drawing import draw_scene_supersampled
from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig, render_image

np.random.seed(42)

# ============================================================
# Load real reference image
# ============================================================
real_img = np.array(Image.open("/home/user/SyMBac_2/old/src/sample_images/sample_100x.tiff"))
print(f"Real image: shape={real_img.shape}, dtype={real_img.dtype}, "
      f"range={real_img.min()}-{real_img.max()}")
TARGET_H, TARGET_W = real_img.shape  # 256 x 46

# ============================================================
# Simulate a mother machine trench matching the real image
# ============================================================
pix_mic_conv = 0.065  # microns per pixel at 100x

physics_config = PhysicsConfig(ITERATIONS=100, DAMPING=0.3)
cell_config = CellConfig(
    GRANULARITY=4,
    SEGMENT_RADIUS=10,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=5,
    BASE_MAX_LENGTH=130,
    MAX_LENGTH_VARIATION=0.15,
    MIN_LENGTH_AFTER_DIVISION=4,
    NOISE_STRENGTH=0.05,
    SEED_CELL_SEGMENTS=20,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.005,
    STIFFNESS=300_000,
    PIVOT_JOINT_STIFFNESS=5000,
    START_POS=Vec2d(0, 50),
    START_ANGLE=np.pi / 2,
    SIMPLE_LENGTH=True,
)


def growth_hook(cell: SimCell):
    compression = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression ** 4


sim = Simulator(physics_config, cell_config)
sim.add_and_run_post_init_hook(lambda s: trench_creator(30, 500, (0, 0), s.space))
sim.add_pre_cell_grow_hook(growth_hook)

def cell_remover(s):
    for c in s.cells[:]:
        pos = c.physics_representation.segments[0].body.position.y
        if pos > 500 or pos < -50:
            s.colony.delete_cell(c)

sim.add_post_step_hook(cell_remover)

for _ in range(1500):
    sim.step()
print(f"Simulation: {sim.num_cells} cells")

# ============================================================
# Render using convolve-at-high-res-then-downscale pipeline
# ============================================================
pixel_scale = 0.65
supersampling = 3
ss_pixel_scale = pixel_scale / supersampling

# Get OPL, masks, device at supersampled resolution
opl_ss, masks_ss, device_ss, native_size = draw_scene_supersampled(
    sim, pixel_scale=pixel_scale, supersampling=supersampling,
)
print(f"Supersampled scene: {opl_ss.shape}, native target: {native_size}")

# Phase contrast PSF at supersampled pixel scale
psf_pc = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=40, pixel_scale=ss_pixel_scale,
)

config = RenderConfig(
    media_multiplier=30.0,
    cell_multiplier=1.7,
    device_multiplier=29.0,
    defocus=0.3,
    noise_var=0.0003,
    border_expansion=2.0,
)

# Convolve at supersampled resolution, then downscale
pc_image, pc_masks = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, config,
    supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)
print(f"PC image: {pc_image.shape}, range {pc_image.min():.3f}-{pc_image.max():.3f}")

# ============================================================
# Resize to match real image dimensions exactly
# ============================================================
pc_resized = resize(pc_image, (TARGET_H, TARGET_W), anti_aliasing=True, preserve_range=True)
masks_resized = resize(masks_ss.astype(float), (TARGET_H, TARGET_W), order=0,
                       anti_aliasing=False, preserve_range=True).astype(np.int32)

# ============================================================
# Match histogram to real image
# ============================================================
real_norm = real_img.astype(np.float64) / real_img.max()
pc_matched = match_histograms(pc_resized, real_norm)

# Convert to 16-bit matching real image range
pc_16bit = rescale_intensity(
    pc_matched,
    out_range=(float(real_img.min()), float(real_img.max()))
).astype(np.uint16)
print(f"16-bit output: shape={pc_16bit.shape}, dtype={pc_16bit.dtype}, "
      f"range={pc_16bit.min()}-{pc_16bit.max()}")

# ============================================================
# Figures
# ============================================================

# 1. Main comparison figure
fig, axes = plt.subplots(1, 4, figsize=(20, 10))

axes[0].imshow(real_img, cmap="Greys_r")
axes[0].set_title(f"Real (100x)\n{real_img.shape[0]}x{real_img.shape[1]}, uint16", fontsize=11)
axes[0].axis("off")

axes[1].imshow(pc_16bit, cmap="Greys_r")
axes[1].set_title(f"Synthetic (matched)\n{pc_16bit.shape[0]}x{pc_16bit.shape[1]}, uint16", fontsize=11)
axes[1].axis("off")

axes[2].imshow(pc_image, cmap="Greys_r")
axes[2].set_title(f"Synthetic (raw)\n{pc_image.shape[0]}x{pc_image.shape[1]}, float", fontsize=11)
axes[2].axis("off")

axes[3].imshow(opl_ss, cmap="inferno")
axes[3].set_title(f"Raw OPL (supersampled)\n{opl_ss.shape[0]}x{opl_ss.shape[1]}", fontsize=11)
axes[3].axis("off")

plt.suptitle("Phase 3: Real vs Synthetic Phase Contrast (convolve-then-downscale)", fontsize=14)
plt.tight_layout()
plt.savefig("/home/user/SyMBac_2/examples/phase3_real_vs_synthetic.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Intensity histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(real_img.ravel(), bins=100, alpha=0.7, label="Real", color="blue", density=True)
axes[0].hist(pc_16bit.ravel(), bins=100, alpha=0.7, label="Synthetic", color="red", density=True)
axes[0].set_title("Intensity Histograms (16-bit)")
axes[0].set_xlabel("Pixel Value")
axes[0].legend()

axes[1].hist(real_norm.ravel(), bins=100, alpha=0.7, label="Real (norm)", color="blue", density=True)
axes[1].hist(pc_resized.ravel(), bins=100, alpha=0.7, label="Synthetic (norm)", color="red", density=True)
axes[1].set_title("Normalized Intensity Histograms")
axes[1].set_xlabel("Pixel Value")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/user/SyMBac_2/examples/phase3_histograms.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Save 16-bit TIFF
Image.fromarray(pc_16bit).save("/home/user/SyMBac_2/examples/phase3_synthetic_100x.tiff")
print(f"\nSaved:")
print(f"  phase3_real_vs_synthetic.png (comparison)")
print(f"  phase3_histograms.png (intensity distributions)")
print(f"  phase3_synthetic_100x.tiff (16-bit output)")
