"""Phase 3 Demo: Full rendering pipeline with phase contrast and fluorescence.

Uses the correct SyMBac pipeline: render OPL at supersampled resolution,
convolve with PSF at high resolution, then downscale to native resolution.

Generates images for all 3 simulation scenarios:
  - Mother machine trench
  - Colony in a box
  - Microfluidic chip channel
"""

import numpy as np
import matplotlib.pyplot as plt
from pymunk import Vec2d
from tqdm import tqdm

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator, box_creator
from symbac.imaging.drawing import draw_scene_supersampled
from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig, render_image, render_fluorescence

np.random.seed(42)

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


def growth_hook(cell: SimCell):
    compression = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression ** 4


def run_sim(simulator, n_steps, desc):
    for _ in tqdm(range(n_steps), desc=desc):
        simulator.step()


# ============================================================
# Run 3 scenarios
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
# PSF and Camera setup
# ============================================================
pixel_scale = 0.5       # native microns/pixel
supersampling = 3       # render and convolve at 3x, then downscale
ss_pixel_scale = pixel_scale / supersampling  # PSF must match supersampled resolution

# Phase contrast PSF (at supersampled pixel scale)
psf_pc = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=40, pixel_scale=ss_pixel_scale,
)

# Fluorescence PSF (at supersampled pixel scale)
psf_fl = PSFModel.fluorescence_2d(
    wavelength=0.5, NA=1.4, n=1.5,
    radius=40, pixel_scale=ss_pixel_scale,
)

# Camera noise model
camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)

# ============================================================
# Render all scenarios
# ============================================================
print("\n=== Phase 3: Full Rendering Pipeline (convolve-then-downscale) ===")

scenarios = [
    ("Mother Machine", sim_mm),
    ("Colony", sim_colony),
    ("Microfluidic Chip", sim_chip),
]

# --- Phase Contrast rendering config ---
pc_config = RenderConfig(
    media_multiplier=30.0,
    cell_multiplier=-5.0,
    device_multiplier=-50.0,
    defocus=1.0,
    noise_var=0.001,
    border_expansion=2.0,
    halo_top_intensity=1.0,
    halo_bottom_intensity=0.97,
)

# --- Fluorescence rendering config ---
fl_config = RenderConfig(
    cell_multiplier=3.0,
    defocus=1.0,
    noise_var=0.0005,
    border_expansion=1.5,
)

# Big comparison figure
fig, axes = plt.subplots(3, 4, figsize=(24, 18))

for row, (name, sim) in enumerate(scenarios):
    print(f"  Rendering {name}...")

    # Get OPL, masks, device at supersampled resolution
    opl_ss, masks_ss, device_ss, native_size = draw_scene_supersampled(
        sim, pixel_scale=pixel_scale, supersampling=supersampling,
    )
    print(f"    Supersampled scene: {opl_ss.shape}, native target: {native_size}")

    # Phase contrast: convolve at high-res, downscale, then add noise
    pc_image, pc_masks = render_image(
        opl_ss, masks_ss, device_ss, psf_pc, pc_config,
        supersampling=supersampling,
        camera=None, rng=np.random.default_rng(42),
    )

    # Phase contrast with camera noise model
    pc_camera, _ = render_image(
        opl_ss, masks_ss, device_ss, psf_pc, pc_config,
        supersampling=supersampling,
        camera=camera, rng=np.random.default_rng(42),
    )

    # Fluorescence (smooth OPL-based)
    fl_image, fl_masks = render_image(
        opl_ss, masks_ss, np.zeros_like(device_ss), psf_fl, fl_config,
        supersampling=supersampling,
        camera=None, rng=np.random.default_rng(42),
    )

    # Fluorescence with molecule sampling
    fl_sampled, _ = render_fluorescence(
        opl_ss, masks_ss, psf_fl, fl_config,
        fl_density=0.05, supersampling=supersampling,
        camera=None, rng=np.random.default_rng(42),
    )

    print(f"    Output: PC={pc_image.shape}, FL={fl_image.shape}")

    # Col 0: Phase contrast (clean)
    axes[row, 0].imshow(pc_image, cmap='Greys_r')
    axes[row, 0].set_title(f"{name} - Phase Contrast", fontsize=12)
    axes[row, 0].axis('off')

    # Col 1: Phase contrast with camera
    axes[row, 1].imshow(pc_camera, cmap='Greys_r')
    axes[row, 1].set_title(f"{name} - PC + Camera Noise", fontsize=12)
    axes[row, 1].axis('off')

    # Col 2: Fluorescence (smooth)
    axes[row, 2].imshow(fl_image, cmap='Greens')
    axes[row, 2].set_title(f"{name} - Fluorescence", fontsize=12)
    axes[row, 2].axis('off')

    # Col 3: Fluorescence (molecule sampling)
    axes[row, 3].imshow(fl_sampled, cmap='Greens')
    axes[row, 3].set_title(f"{name} - FL (sampled molecules)", fontsize=12)
    axes[row, 3].axis('off')

plt.suptitle("Phase 3: Convolve-then-Downscale Pipeline", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/phase3_output.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved phase3_output.png")

# ============================================================
# Also save individual high-res images
# ============================================================
for name, sim in scenarios:
    safe_name = name.lower().replace(" ", "_")
    opl_ss, masks_ss, device_ss, native_size = draw_scene_supersampled(
        sim, pixel_scale=pixel_scale, supersampling=supersampling,
    )

    # Phase contrast
    pc_image, _ = render_image(
        opl_ss, masks_ss, device_ss, psf_pc, pc_config,
        supersampling=supersampling,
        camera=None, rng=np.random.default_rng(42),
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pc_image, cmap='Greys_r')
    ax.set_title(f"{name} - Phase Contrast")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/home/user/SyMBac_2/examples/phase3_{safe_name}_pc.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fluorescence
    fl_image, _ = render_image(
        opl_ss, masks_ss, np.zeros_like(device_ss), psf_fl, fl_config,
        supersampling=supersampling,
        camera=None, rng=np.random.default_rng(42),
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(fl_image, cmap='Greens')
    ax.set_title(f"{name} - Fluorescence")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/home/user/SyMBac_2/examples/phase3_{safe_name}_fl.png", dpi=150, bbox_inches='tight')
    plt.close()

print("Done! All Phase 3 images saved.")
