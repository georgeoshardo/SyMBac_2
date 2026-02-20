"""Phase 4 Demo: Training data generation pipeline.

Demonstrates:
1. Recording simulation frames (OPL scenes + masks at supersampled resolution)
2. Generating varied training data with parameter variation
3. Saving image/mask pairs to disk
"""

import numpy as np
import matplotlib.pyplot as plt
from pymunk import Vec2d
from functools import partial

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator, box_creator
from symbac.imaging.drawing import draw_scene_supersampled
from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig
from symbac.imaging.training_data import (
    TrainingDataConfig, generate_training_data, record_simulation_frames,
)

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


# ============================================================
# Setup: Mother Machine simulation
# ============================================================
pixel_scale = 0.5
supersampling = 3

sim = Simulator(
    physics_config,
    make_cell_config(START_POS=Vec2d(0, 50), START_ANGLE=np.pi / 2),
)
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

# ============================================================
# Step 1: Record simulation frames
# ============================================================
print("=== Phase 4: Training Data Generation ===")
print("Step 1: Recording simulation frames...")

draw_fn = partial(
    draw_scene_supersampled,
    pixel_scale=pixel_scale,
    supersampling=supersampling,
)

# Wrap draw_fn to strip the native_size return value
def draw_3tuple(sim):
    opl, masks, device, _native_size = draw_fn(sim)
    return opl, masks, device

opl_scenes, mask_scenes, device_masks = record_simulation_frames(
    sim,
    draw_fn=draw_3tuple,
    n_steps=300,
    record_every=5,
    desc="Mother Machine",
)
print(f"  Recorded {len(opl_scenes)} frames")
print(f"  Frame shape: {opl_scenes[0].shape} (supersampled at {supersampling}x)")

# ============================================================
# Step 2: Generate training data with parameter variation
# ============================================================
print("\nStep 2: Generating training data with parameter variation...")

ss_pixel_scale = pixel_scale / supersampling

psf_pc = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=40, pixel_scale=ss_pixel_scale,
)

render_config = RenderConfig(
    media_multiplier=30.0,
    cell_multiplier=1.7,
    device_multiplier=29.0,
    defocus=0.3,
    noise_var=0.0003,
    border_expansion=2.0,
)

training_config = TrainingDataConfig(
    n_samples=12,           # Small batch for demo
    burn_in=5,              # Skip first 5 recorded frames
    sample_variation=0.15,  # +-15% parameter variation
    in_series=False,        # Random frame selection
    randomize_histogram=False,
    match_histogram=False,
    seed=42,
)

save_dir = "/home/user/SyMBac_2/examples/phase4_training_data"

results = generate_training_data(
    opl_scenes=opl_scenes,
    mask_scenes=mask_scenes,
    device_masks=device_masks,
    psf_model=psf_pc,
    render_config=render_config,
    training_config=training_config,
    supersampling=supersampling,
    camera=None,
    save_dir=save_dir,
    prefix="mm_",
)

print(f"  Generated {len(results)} training pairs")
print(f"  Saved to {save_dir}/")

# ============================================================
# Step 3: Visualize training data samples
# ============================================================
print("\nStep 3: Visualizing training data samples...")

# Grid of first 12 samples
n_show = min(12, len(results))
fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))

for i in range(n_show):
    synth, mask = results[i]
    axes[0, i].imshow(synth, cmap="Greys_r")
    axes[0, i].set_title(f"Sample {i}", fontsize=9)
    axes[0, i].axis("off")

    axes[1, i].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Synthetic", fontsize=11)
axes[1, 0].set_ylabel("Mask", fontsize=11)

plt.suptitle("Phase 4: Training Data Samples (with parameter variation)", fontsize=14)
plt.tight_layout()
plt.savefig(
    "/home/user/SyMBac_2/examples/phase4_training_samples.png",
    dpi=150, bbox_inches="tight",
)
plt.close()
print("  Saved phase4_training_samples.png")

# ============================================================
# Step 4: Show parameter variation effect
# ============================================================
print("\nStep 4: Showing effect of parameter variation...")

# Render the same frame with different variation levels
fixed_frame = len(opl_scenes) // 2  # Middle frame
variations = [0.0, 0.05, 0.10, 0.20, 0.40]

fig, axes = plt.subplots(1, len(variations), figsize=(4 * len(variations), 6))

for col, var in enumerate(variations):
    tc = TrainingDataConfig(
        n_samples=1, burn_in=0, sample_variation=var,
        randomize_histogram=False, match_histogram=False, seed=99,
    )
    result = generate_training_data(
        [opl_scenes[fixed_frame]],
        [mask_scenes[fixed_frame]],
        [device_masks[fixed_frame]],
        psf_pc, render_config, tc,
        supersampling=supersampling,
    )
    synth, _ = result[0]
    axes[col].imshow(synth, cmap="Greys_r")
    axes[col].set_title(f"Variation: {var:.0%}", fontsize=11)
    axes[col].axis("off")

plt.suptitle("Effect of Parameter Variation on Rendered Image", fontsize=14)
plt.tight_layout()
plt.savefig(
    "/home/user/SyMBac_2/examples/phase4_variation_effect.png",
    dpi=150, bbox_inches="tight",
)
plt.close()
print("  Saved phase4_variation_effect.png")

print("\nDone! Phase 4 training data generation complete.")
