"""Render mother machine synthetic images using parameters from the old SyMBac
optimiser widget.

Parameters (from napari widget screenshot):
  media_multiplier = -28.21
  cell_multiplier  = 50.00
  device_multiplier = 16.22
  defocus          = 20.00
  noise_var        = 0.00
  halo_top/bottom  = 1.00
  match_histogram  = off
  match_fourier    = off

Updated:
  - PSF: NA=1.49, wavelength=0.70 µm (700nm) for sharper cells
  - Trench width tightened so cells touch walls (ratio ~1.15:1)
  - Simulation length: 4000 steps (fill the trench)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymunk import Vec2d
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from tqdm import tqdm

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator
from symbac.imaging.drawing import draw_scene_supersampled
from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig, render_image

np.random.seed(42)

# ============================================================
# Load real reference image for comparison
# ============================================================
real_img = np.array(Image.open("/home/user/SyMBac_2/old/src/sample_images/sample_100x.tiff"))
print(f"Real image: shape={real_img.shape}, dtype={real_img.dtype}, "
      f"range={real_img.min()}-{real_img.max()}")

# ============================================================
# Mother machine simulation — 4000 steps, tight trench
# ============================================================
# Geometry tuned to match real mother machine appearance:
# - Trench +20% wider again (26→31)
# - Trench 20% shorter (475→380)
# - Cells 20% wider (SEGMENT_RADIUS 12→14)
# - Division length ~2x longer (BASE_MAX_LENGTH 130→260)
TRENCH_WIDTH = 31
TRENCH_LENGTH = 380
SIM_STEPS = 4000

physics_config = PhysicsConfig(ITERATIONS=100, DAMPING=0.3)
cell_config = CellConfig(
    GRANULARITY=4,
    SEGMENT_RADIUS=14,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=5,
    BASE_MAX_LENGTH=173,
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
sim.add_and_run_post_init_hook(
    lambda s: trench_creator(TRENCH_WIDTH, TRENCH_LENGTH, (0, 0), s.space)
)
sim.add_pre_cell_grow_hook(growth_hook)


def cell_remover(s):
    for c in s.cells[:]:
        pos = c.physics_representation.segments[0].body.position.y
        if pos > TRENCH_LENGTH or pos < -50:
            s.colony.delete_cell(c)


sim.add_post_step_hook(cell_remover)

print(f"Running simulation ({SIM_STEPS} steps)...")
for _ in tqdm(range(SIM_STEPS), desc="Simulating"):
    sim.step()
print(f"Simulation complete: {sim.num_cells} cells")

# ============================================================
# Render setup — sharper PSF: NA=1.49, 700nm
# ============================================================
pixel_scale = 0.65
supersampling = 3
ss_pixel_scale = pixel_scale / supersampling

# Sharp phase contrast PSF at NA=1.49, 700nm
psf_pc = PSFModel.phase_contrast(
    wavelength=0.70, NA=1.49, n=1.515,
    condenser="Ph3", apo_sigma=20,
    radius=50, pixel_scale=ss_pixel_scale,
)

# Camera from old notebook
camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)

# ============================================================
# Draw the scene
# ============================================================
opl_ss, masks_ss, device_ss, native_size = draw_scene_supersampled(
    sim, pixel_scale=pixel_scale, supersampling=supersampling,
)
print(f"Supersampled scene: {opl_ss.shape}, native target: {native_size}")

# ============================================================
# Parameters from the user's screenshot
# ============================================================
config_matched = RenderConfig(
    media_multiplier=-28.21,
    cell_multiplier=50.00,
    device_multiplier=16.22,
    defocus=10.0,
    noise_var=0.0,
    border_expansion=1.5,
    halo_top_intensity=1.0,
    halo_bottom_intensity=1.0,
)

# Also render with default params for comparison
config_default = RenderConfig(
    media_multiplier=30.0,
    cell_multiplier=-5.0,
    device_multiplier=-50.0,
    defocus=1.0,
    noise_var=0.001,
    border_expansion=2.0,
    halo_top_intensity=1.0,
    halo_bottom_intensity=0.97,
)

# ---- Render with user's matched parameters ----
print("Rendering with matched parameters...")
pc_matched, masks_matched = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, config_matched,
    supersampling=supersampling,
    camera=None,
    rng=np.random.default_rng(42),
)
print(f"  Matched: {pc_matched.shape}, range [{pc_matched.min():.3f}, {pc_matched.max():.3f}]")

# ---- Render with camera noise ----
print("Rendering with matched parameters + camera noise...")
pc_matched_cam, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, config_matched,
    supersampling=supersampling,
    camera=camera,
    rng=np.random.default_rng(42),
)

# ---- Render with default parameters ----
print("Rendering with default parameters...")
pc_default, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, config_default,
    supersampling=supersampling,
    camera=None,
    rng=np.random.default_rng(42),
)

# ============================================================
# Figures
# ============================================================
TARGET_H, TARGET_W = real_img.shape
real_norm = real_img.astype(np.float64) / real_img.max()

# Figure 1: Full comparison
fig, axes = plt.subplots(1, 5, figsize=(25, 10))

axes[0].imshow(real_norm, cmap="Greys_r")
axes[0].set_title(f"Real (100x)\n{real_img.shape[0]}x{real_img.shape[1]}", fontsize=11)
axes[0].axis("off")

axes[1].imshow(pc_matched, cmap="Greys_r")
axes[1].set_title(f"Synthetic (matched params)\nmedia={config_matched.media_multiplier}\n"
                  f"cell={config_matched.cell_multiplier}\n"
                  f"device={config_matched.device_multiplier}\n"
                  f"defocus={config_matched.defocus}",
                  fontsize=9)
axes[1].axis("off")

axes[2].imshow(pc_matched_cam, cmap="Greys_r")
axes[2].set_title("Synthetic (matched + camera)", fontsize=11)
axes[2].axis("off")

axes[3].imshow(pc_default, cmap="Greys_r")
axes[3].set_title(f"Synthetic (default params)\nmedia={config_default.media_multiplier}\n"
                  f"cell={config_default.cell_multiplier}\n"
                  f"device={config_default.device_multiplier}\n"
                  f"defocus={config_default.defocus}",
                  fontsize=9)
axes[3].axis("off")

axes[4].imshow(opl_ss, cmap="inferno")
axes[4].set_title(f"Raw OPL (supersampled)\n{opl_ss.shape[0]}x{opl_ss.shape[1]}", fontsize=11)
axes[4].axis("off")

plt.suptitle(f"Mother Machine Renders (NA=1.49, 700nm, trench_w={TRENCH_WIDTH}, sim={SIM_STEPS})",
             fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/render_matched_params_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved render_matched_params_comparison.png")

# Figure 2: Intensity histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

matched_resized = resize(pc_matched, (TARGET_H, TARGET_W),
                         anti_aliasing=True, preserve_range=True)

axes[0].hist(real_norm.ravel(), bins=100, alpha=0.6, label="Real", color="blue", density=True)
axes[0].hist(matched_resized.ravel(), bins=100, alpha=0.6, label="Synthetic (matched)", color="red", density=True)
axes[0].set_title("Intensity Histograms — Real vs Matched")
axes[0].set_xlabel("Pixel Value (normalized)")
axes[0].set_ylabel("Density")
axes[0].legend()

default_resized = resize(pc_default, (TARGET_H, TARGET_W),
                         anti_aliasing=True, preserve_range=True)
axes[1].hist(real_norm.ravel(), bins=100, alpha=0.6, label="Real", color="blue", density=True)
axes[1].hist(default_resized.ravel(), bins=100, alpha=0.6, label="Synthetic (default)", color="orange", density=True)
axes[1].set_title("Intensity Histograms — Real vs Default")
axes[1].set_xlabel("Pixel Value (normalized)")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/user/SyMBac_2/examples/render_matched_params_histograms.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved render_matched_params_histograms.png")

# Figure 3: Feature matching diagnostic
from symbac.imaging.feature_matching import compare_images, plot_comparison

report_matched = compare_images(real_norm, matched_resized)
report_default = compare_images(real_norm, default_resized)

print(f"\nFeature distance (matched params):  {report_matched['overall_distance']:.4f}")
print(f"Feature distance (default params):  {report_default['overall_distance']:.4f}")
if "ssim" in report_matched:
    print(f"SSIM (matched params):  {report_matched['ssim']:.4f}")
    print(f"SSIM (default params):  {report_default['ssim']:.4f}")

plot_comparison(
    real_norm, matched_resized, report_matched,
    save_path="/home/user/SyMBac_2/examples/render_matched_params_features.png",
)
print("Saved render_matched_params_features.png")

# Figure 4: Side-by-side
fig, axes = plt.subplots(1, 3, figsize=(12, 8))

axes[0].imshow(real_norm, cmap="Greys_r")
axes[0].set_title("Real Image", fontsize=12)
axes[0].axis("off")

axes[1].imshow(matched_resized, cmap="Greys_r")
axes[1].set_title("Synthetic (matched params)", fontsize=12)
axes[1].axis("off")

diff = np.abs(real_norm - matched_resized)
axes[2].imshow(diff, cmap="hot")
axes[2].set_title(f"Absolute Difference\n(dist={report_matched['overall_distance']:.3f})", fontsize=12)
axes[2].axis("off")

plt.suptitle("Real vs Synthetic — Detailed Comparison", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/render_matched_params_sidebyside.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved render_matched_params_sidebyside.png")

# Print config
print("\n=== Configuration ===")
print(f"  PSF: NA=1.49, wavelength=0.70µm, n=1.515, condenser=Ph3, apo_sigma=20")
print(f"  Trench: width={TRENCH_WIDTH}, length={TRENCH_LENGTH}")
print(f"  Simulation: {SIM_STEPS} steps, {sim.num_cells} cells")
print(f"  Render parameters:")
for field in ["media_multiplier", "cell_multiplier", "device_multiplier",
              "defocus", "noise_var", "halo_top_intensity", "halo_bottom_intensity"]:
    print(f"    {field}: {getattr(config_matched, field)}")

print("\nDone!")
