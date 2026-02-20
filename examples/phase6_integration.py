"""Phase 6: Full integration example - simulation to training data.

Demonstrates the complete SyMBac_2 pipeline:
1. Set up physics simulation with cell growth
2. Draw OPL scenes at supersampled resolution
3. Render phase contrast and fluorescence images
4. Add agar pad backgrounds for colony rendering
5. Generate diverse training data with parameter variation

This is the main reference example for using the SyMBac_2 imaging pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymunk import Vec2d
from functools import partial

# --- Simulation imports ---
from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator, box_creator

# --- Imaging imports (all from the clean public API) ---
from symbac.imaging import (
    PSFModel, Camera, RenderConfig,
    draw_scene_supersampled, render_image, render_fluorescence,
    random_perlin_background, generate_perlin_background, NoiseConfig,
    TrainingDataConfig, generate_training_data, record_simulation_frames,
)

np.random.seed(42)


# ============================================================
# 1. SIMULATION SETUP
# ============================================================
print("=" * 60)
print("SyMBac_2: Full Integration Example")
print("=" * 60)

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
    """Reduce growth rate when cells are compressed."""
    compression = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression ** 4


# Mother machine simulator
sim = Simulator(physics_config, cell_config)
sim.add_and_run_post_init_hook(lambda s: trench_creator(30, 500, (0, 0), s.space))
sim.add_post_step_hook(
    lambda s: [
        s.colony.delete_cell(c)
        for c in s.cells[:]
        if c.physics_representation.segments[0].body.position.y > 500
        or c.physics_representation.segments[0].body.position.y < -50
    ]
)
sim.add_pre_cell_grow_hook(growth_hook)

print("\n1. Simulation setup complete.")

# ============================================================
# 2. OPTICS SETUP
# ============================================================
pixel_scale = 0.5       # microns/pixel at native resolution
supersampling = 3       # render at 3x, convolve, then downscale
ss_pixel_scale = pixel_scale / supersampling

# Phase contrast PSF (at supersampled pixel scale)
psf_pc = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=40, pixel_scale=ss_pixel_scale,
)

# Fluorescence PSF
psf_fl = PSFModel.fluorescence_2d(
    wavelength=0.5, NA=1.4, n=1.5,
    radius=40, pixel_scale=ss_pixel_scale,
)

# Camera noise model
camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)

print("2. Optics setup complete.")
print(f"   PSF kernel sum: {psf_pc.kernel_2d.sum():.4f} (negative = inverts intensity)")

# ============================================================
# 3. RENDERING CONFIGURATION
# ============================================================
pc_config = RenderConfig(
    media_multiplier=30.0,       # Trench interior (bright after inversion)
    cell_multiplier=-5.0,        # Negative: cells darker than media
    device_multiplier=-50.0,     # PDMS (dark after inversion)
    defocus=1.0,
    noise_var=0.001,
    border_expansion=1.5,
)

fl_config = RenderConfig(
    cell_multiplier=3.0,
    defocus=1.0,
    noise_var=0.0005,
    border_expansion=1.5,
)

print("3. Rendering config ready.")

# ============================================================
# 4. SINGLE-FRAME RENDERING
# ============================================================
print("\n4. Running simulation and rendering single frame...")

for _ in range(1500):
    sim.step()
print(f"   Cells: {sim.num_cells}")

# Draw OPL at supersampled resolution
opl_ss, masks_ss, device_ss, native_size = draw_scene_supersampled(
    sim, pixel_scale=pixel_scale, supersampling=supersampling,
)
print(f"   Supersampled scene: {opl_ss.shape}")
print(f"   Native target: {native_size}")

# Phase contrast
pc_image, pc_masks = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, pc_config,
    supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)

# Phase contrast with camera noise
pc_camera, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, pc_config,
    supersampling=supersampling,
    camera=camera, rng=np.random.default_rng(42),
)

# Fluorescence (smooth OPL)
fl_image, _ = render_image(
    opl_ss, masks_ss, np.zeros_like(device_ss), psf_fl, fl_config,
    supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)

# Fluorescence (sampled molecules)
fl_sampled, _ = render_fluorescence(
    opl_ss, masks_ss, psf_fl, fl_config,
    fl_density=0.05, supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)

print(f"   Output: PC={pc_image.shape}, FL={fl_image.shape}")

# ============================================================
# 5. PERLIN NOISE BACKGROUND
# ============================================================
print("\n5. Adding agar pad background texture...")

native_h = opl_ss.shape[0] // supersampling
native_w = opl_ss.shape[1] // supersampling

# Phase contrast with Perlin background
bg = generate_perlin_background(
    (native_h, native_w),
    NoiseConfig(scale=4, octaves=12, seed=99),
)
pc_with_bg, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, pc_config,
    supersampling=supersampling,
    camera=None,
    perlin_background=bg,
    perlin_attenuation=3.0,
    perlin_blur_sigma=2.0,
    rng=np.random.default_rng(42),
)

print(f"   PC with background: {pc_with_bg.shape}")

# ============================================================
# 6. TRAINING DATA GENERATION
# ============================================================
print("\n6. Generating training data batch...")

# Record simulation frames
draw_fn = partial(
    draw_scene_supersampled,
    pixel_scale=pixel_scale,
    supersampling=supersampling,
)


def draw_3tuple(sim):
    opl, masks, device, _native = draw_fn(sim)
    return opl, masks, device


opl_scenes, mask_scenes, device_masks = record_simulation_frames(
    sim, draw_fn=draw_3tuple,
    n_steps=100, record_every=5,
    desc="Recording frames",
)
print(f"   Recorded {len(opl_scenes)} frames")

# Generate training data
training_config = TrainingDataConfig(
    n_samples=6,
    burn_in=3,
    sample_variation=0.15,
    in_series=False,
    randomize_histogram=False,
    match_histogram=False,
    seed=42,
)

results = generate_training_data(
    opl_scenes=opl_scenes,
    mask_scenes=mask_scenes,
    device_masks=device_masks,
    psf_model=psf_pc,
    render_config=pc_config,
    training_config=training_config,
    supersampling=supersampling,
    camera=None,
)
print(f"   Generated {len(results)} training pairs")

# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n7. Creating visualization...")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 0: Single-frame rendering pipeline
axes[0, 0].imshow(opl_ss, cmap="inferno")
axes[0, 0].set_title("OPL (supersampled)", fontsize=10)
axes[0, 0].axis("off")

axes[0, 1].imshow(pc_image, cmap="Greys_r")
axes[0, 1].set_title("Phase Contrast", fontsize=10)
axes[0, 1].axis("off")

axes[0, 2].imshow(pc_camera, cmap="Greys_r")
axes[0, 2].set_title("PC + Camera Noise", fontsize=10)
axes[0, 2].axis("off")

axes[0, 3].imshow(pc_with_bg, cmap="Greys_r")
axes[0, 3].set_title("PC + Perlin BG", fontsize=10)
axes[0, 3].axis("off")

# Row 1: Fluorescence + masks
axes[1, 0].imshow(fl_image, cmap="Greens")
axes[1, 0].set_title("Fluorescence (smooth)", fontsize=10)
axes[1, 0].axis("off")

axes[1, 1].imshow(fl_sampled, cmap="Greens")
axes[1, 1].set_title("Fluorescence (sampled)", fontsize=10)
axes[1, 1].axis("off")

axes[1, 2].imshow(masks_ss, cmap="nipy_spectral")
axes[1, 2].set_title("Instance Masks (SS)", fontsize=10)
axes[1, 2].axis("off")

axes[1, 3].imshow(pc_masks, cmap="nipy_spectral")
axes[1, 3].set_title("Output Masks (native)", fontsize=10)
axes[1, 3].axis("off")

# Row 2: Training data samples
n_show = min(4, len(results))
for i in range(n_show):
    synth, mask = results[i]
    # Overlay: show synth with mask outline
    axes[2, i].imshow(synth, cmap="Greys_r")
    axes[2, i].set_title(f"Training Sample {i}", fontsize=10)
    axes[2, i].axis("off")

plt.suptitle("SyMBac_2: Full Imaging Pipeline Integration", fontsize=14)
plt.tight_layout()
plt.savefig(
    "/home/user/SyMBac_2/examples/phase6_integration.png",
    dpi=150, bbox_inches="tight",
)
plt.close()

print("   Saved phase6_integration.png")
print("\n" + "=" * 60)
print("Integration example complete!")
print("=" * 60)
