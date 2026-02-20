"""Phase 2 Demo: PSF models and Camera noise visualization.

Shows phase contrast and fluorescence PSFs, and previews what they look like
applied (conceptually via simple convolution) to the 3 simulation scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pymunk import Vec2d
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator, box_creator
from symbac.imaging.drawing import draw_scene, draw_scene_with_geometry
from symbac.imaging.optics import PSFModel, Camera

np.random.seed(42)

# --- Shared configs ---
physics_config = PhysicsConfig(ITERATIONS=100, DAMPING=0.3)


def make_cell_config(**overrides):
    defaults = dict(
        GRANULARITY=4, SEGMENT_RADIUS=10, SEGMENT_MASS=1.0,
        GROWTH_RATE=5, BASE_MAX_LENGTH=130, MAX_LENGTH_VARIATION=0.15,
        MIN_LENGTH_AFTER_DIVISION=4, NOISE_STRENGTH=0.05, SEED_CELL_SEGMENTS=20,
        ROTARY_LIMIT_JOINT=True, MAX_BEND_ANGLE=0.005, STIFFNESS=300_000,
        PIVOT_JOINT_STIFFNESS=5000, SIMPLE_LENGTH=True,
    )
    defaults.update(overrides)
    return CellConfig(**defaults)


def growth_hook(cell: SimCell) -> None:
    compression = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression ** 4


def run_sim(simulator, n_steps, desc):
    for _ in tqdm(range(n_steps), desc=desc):
        simulator.step()


# ============================================================
# Run 3 scenarios (same as Phase 1)
# ============================================================

# 1. Mother Machine
sim_mm = Simulator(physics_config, make_cell_config(START_POS=Vec2d(0, 50), START_ANGLE=np.pi / 2))
sim_mm.add_and_run_post_init_hook(lambda s: trench_creator(30, 500, (0, 0), s.space))
sim_mm.add_post_step_hook(lambda s: [s.colony.delete_cell(c) for c in s.cells[:] if c.physics_representation.segments[0].body.position.y > 500])
sim_mm.add_pre_cell_grow_hook(growth_hook)
run_sim(sim_mm, 1500, "Mother Machine")

# 2. Colony
sim_colony = Simulator(physics_config, make_cell_config(START_POS=Vec2d(0, 0), START_ANGLE=0.3, BASE_MAX_LENGTH=90, GROWTH_RATE=8))
sim_colony.add_and_run_post_init_hook(lambda s: box_creator(600, 600, (0, 0), s.space, barrier_thickness=10, fillet_radius=50))
sim_colony.add_post_step_hook(lambda s: [s.colony.delete_cell(c) for c in s.cells[:] if abs(c.physics_representation.segments[0].body.position.x) > 400 or abs(c.physics_representation.segments[0].body.position.y) > 400])
sim_colony.add_pre_cell_grow_hook(growth_hook)
run_sim(sim_colony, 1500, "Colony")

# 3. Microfluidic Chip
sim_chip = Simulator(physics_config, make_cell_config(START_POS=Vec2d(0, 100), START_ANGLE=np.pi / 2, BASE_MAX_LENGTH=100, GROWTH_RATE=6))
sim_chip.add_and_run_post_init_hook(lambda s: box_creator(80, 1200, (0, 0), s.space, barrier_thickness=8, fillet_radius=30))
sim_chip.add_post_step_hook(lambda s: [s.colony.delete_cell(c) for c in s.cells[:] if abs(c.physics_representation.segments[0].body.position.y) > 700])
sim_chip.add_pre_cell_grow_hook(growth_hook)
run_sim(sim_chip, 2000, "Microfluidic Chip")

# ============================================================
# Create PSF models
# ============================================================
print("\n=== Creating PSF models ===")

pix_mic_conv = 0.065  # microns per pixel (native camera)
supersampling = 3
render_scale = pix_mic_conv / supersampling  # pixel scale at rendering resolution

# Phase contrast PSFs for different condensers
psf_pc_ph3 = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=50, pixel_scale=render_scale,
)

psf_pc_ph1 = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph1", apo_sigma=10,
    radius=50, pixel_scale=render_scale,
)

# Fluorescence PSF
psf_fluo = PSFModel.fluorescence_2d(
    wavelength=0.5, NA=1.4, n=1.5,
    radius=50, pixel_scale=render_scale,
)

# Camera
camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)

# ============================================================
# Figure 1: PSF Gallery
# ============================================================
print("Generating PSF gallery...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: PSF kernels
for ax, (name, psf) in zip(axes[0, :3], [
    ("Phase Contrast (Ph3)", psf_pc_ph3),
    ("Phase Contrast (Ph1)", psf_pc_ph1),
    ("Fluorescence 2D", psf_fluo),
]):
    k = psf.kernel
    ax.imshow(k, cmap="RdBu_r", interpolation="nearest")
    ax.set_title(name, fontsize=11)
    ax.axis("off")

# Camera dark image
dark = camera.render_dark_image((101, 101), plot=False)
axes[0, 3].imshow(dark, cmap="Greys_r")
axes[0, 3].set_title("Camera Dark Image", fontsize=11)
axes[0, 3].axis("off")

# Row 2: PSF cross-sections
for ax, (name, psf) in zip(axes[1, :3], [
    ("Ph3 Cross-Section", psf_pc_ph3),
    ("Ph1 Cross-Section", psf_pc_ph1),
    ("Fluorescence Cross-Section", psf_fluo),
]):
    k = psf.kernel
    mid = k.shape[0] // 2
    ax.plot(k[mid, :], 'b-', linewidth=1.5)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)

# Camera noise histogram
axes[1, 3].hist(dark.ravel(), bins=50, color='gray', edgecolor='black', alpha=0.7)
axes[1, 3].set_title("Dark Image Histogram", fontsize=11)
axes[1, 3].set_xlabel("Intensity")
axes[1, 3].set_ylabel("Count")

plt.suptitle("Phase 2: PSF Models and Camera", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/home/user/SyMBac_2/examples/phase2_psf_gallery.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved phase2_psf_gallery.png")

# ============================================================
# Figure 2: Quick convolution preview (Phase Contrast + Fluorescence)
# ============================================================
print("Generating convolution previews...")

pixel_scale = 0.5  # For the OPL rendering
scenarios = [
    ("Mother Machine", sim_mm),
    ("Colony", sim_colony),
    ("Microfluidic Chip", sim_chip),
]

# Build PSFs at the OPL rendering scale
psf_pc_render = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=30, pixel_scale=pixel_scale / supersampling,
)
psf_fl_render = PSFModel.fluorescence_2d(
    wavelength=0.5, NA=1.4, n=1.5,
    radius=30, pixel_scale=pixel_scale / supersampling,
)


def quick_convolve(opl, kernel, defocus=2.0):
    """Quick convolution + rescale for preview."""
    if defocus > 0:
        kernel = gaussian_filter(kernel, defocus, mode="constant")
    result = fftconvolve(opl, kernel, mode="same")
    result = rescale_intensity(result.astype(np.float32), out_range=(0, 1))
    return result


fig, axes = plt.subplots(3, 3, figsize=(18, 18))

for row, (name, sim) in enumerate(scenarios):
    print(f"  Rendering {name}...")
    opl, masks, device = draw_scene_with_geometry(sim, pixel_scale=pixel_scale, supersampling=supersampling)

    # OPL
    axes[row, 0].imshow(opl, cmap="inferno")
    axes[row, 0].set_title(f"{name} - Raw OPL", fontsize=12)
    axes[row, 0].axis("off")

    # Phase contrast preview
    pc_img = quick_convolve(opl, psf_pc_render.kernel, defocus=3.0)
    axes[row, 1].imshow(pc_img, cmap="Greys_r")
    axes[row, 1].set_title(f"{name} - Phase Contrast (preview)", fontsize=12)
    axes[row, 1].axis("off")

    # Fluorescence preview
    fl_img = quick_convolve(opl, psf_fl_render.kernel, defocus=1.0)
    axes[row, 2].imshow(fl_img, cmap="Greens")
    axes[row, 2].set_title(f"{name} - Fluorescence (preview)", fontsize=12)
    axes[row, 2].axis("off")

plt.suptitle("Phase 2: OPL + PSF Convolution Preview", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/phase2_convolution_preview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved phase2_convolution_preview.png")

print("\nDone! All Phase 2 images saved.")
