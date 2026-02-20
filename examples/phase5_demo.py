"""Phase 5 Demo: Colony rendering with Perlin noise backgrounds.

Demonstrates:
1. Perlin noise texture generation for agar pad backgrounds
2. Colony phase contrast rendering with substrate texture
3. Comparison of different noise parameters
4. Full colony and trench rendering with backgrounds
"""

import numpy as np
import matplotlib.pyplot as plt
from pymunk import Vec2d

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator, box_creator
from symbac.imaging.drawing import draw_scene_supersampled
from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig, render_image, render_fluorescence
from symbac.imaging.noise import (
    generate_perlin_background,
    random_perlin_background,
    NoiseConfig,
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
# Step 1: Demonstrate Perlin noise textures
# ============================================================
print("=== Phase 5: Colony Rendering with Perlin Noise ===")
print("Step 1: Generating Perlin noise textures...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

configs = [
    NoiseConfig(scale=5, octaves=10, persistence=1.9, lacunarity=1.8, seed=42),
    NoiseConfig(scale=2, octaves=12, persistence=1.5, lacunarity=1.6, seed=42),
    NoiseConfig(scale=7, octaves=8, persistence=1.0, lacunarity=2.0, seed=42),
    NoiseConfig(scale=3, octaves=13, persistence=1.8, lacunarity=1.55, seed=42),
]
labels = ["Default (scale=5)", "Fine (scale=2, oct=12)", "Coarse (scale=7)", "High detail (oct=13)"]

for col, (cfg, label) in enumerate(zip(configs, labels)):
    bg = generate_perlin_background((300, 300), cfg)
    axes[0, col].imshow(bg, cmap="Greys_r")
    axes[0, col].set_title(label, fontsize=10)
    axes[0, col].axis("off")

# Random backgrounds
rng = np.random.default_rng(42)
for col in range(4):
    bg = random_perlin_background((300, 300), rng=rng)
    axes[1, col].imshow(bg, cmap="Greys_r")
    axes[1, col].set_title(f"Random #{col+1}", fontsize=10)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Fixed params", fontsize=12)
axes[1, 0].set_ylabel("Randomized", fontsize=12)

plt.suptitle("Phase 5: Perlin Noise Textures (Agar Pad Simulation)", fontsize=14)
plt.tight_layout()
plt.savefig("/home/user/SyMBac_2/examples/phase5_noise_gallery.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved phase5_noise_gallery.png")

# ============================================================
# Step 2: Simulate colony and mother machine
# ============================================================
print("\nStep 2: Running simulations...")

# Colony simulation (smaller box for faster rendering)
sim_colony = Simulator(
    physics_config,
    make_cell_config(START_POS=Vec2d(0, 0), START_ANGLE=0.3,
                     BASE_MAX_LENGTH=90, GROWTH_RATE=8),
)
sim_colony.add_and_run_post_init_hook(
    lambda s: box_creator(300, 300, (0, 0), s.space, barrier_thickness=10, fillet_radius=30)
)
sim_colony.add_post_step_hook(
    lambda s: [s.colony.delete_cell(c) for c in s.cells[:]
               if abs(c.physics_representation.segments[0].body.position.x) > 200
               or abs(c.physics_representation.segments[0].body.position.y) > 200]
)
sim_colony.add_pre_cell_grow_hook(growth_hook)
for _ in range(1000):
    sim_colony.step()
print(f"  Colony: {sim_colony.num_cells} cells")

# Mother machine simulation
sim_mm = Simulator(
    physics_config,
    make_cell_config(START_POS=Vec2d(0, 50), START_ANGLE=np.pi / 2),
)
sim_mm.add_and_run_post_init_hook(lambda s: trench_creator(30, 500, (0, 0), s.space))
sim_mm.add_post_step_hook(
    lambda s: [s.colony.delete_cell(c) for c in s.cells[:]
               if c.physics_representation.segments[0].body.position.y > 500
               or c.physics_representation.segments[0].body.position.y < -50]
)
sim_mm.add_pre_cell_grow_hook(growth_hook)
for _ in range(1500):
    sim_mm.step()
print(f"  Mother machine: {sim_mm.num_cells} cells")

# ============================================================
# Step 3: Render with and without Perlin backgrounds
# ============================================================
print("\nStep 3: Rendering with Perlin backgrounds...")

pixel_scale = 0.5
supersampling = 3
ss_pixel_scale = pixel_scale / supersampling

psf_pc = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=40, pixel_scale=ss_pixel_scale,
)
psf_fl = PSFModel.fluorescence_2d(
    wavelength=0.5, NA=1.4, n=1.5,
    radius=40, pixel_scale=ss_pixel_scale,
)

pc_config = RenderConfig(
    media_multiplier=30.0,
    cell_multiplier=-5.0,
    device_multiplier=-50.0,
    defocus=1.0,
    noise_var=0.001,
    border_expansion=1.3,
)

# --- Colony rendering ---
opl_col, masks_col, device_col, _ = draw_scene_supersampled(
    sim_colony, pixel_scale=pixel_scale, supersampling=supersampling,
)
print(f"  Colony scene: {opl_col.shape}")

# Native resolution for Perlin background
native_h = opl_col.shape[0] // supersampling
native_w = opl_col.shape[1] // supersampling

# Render without background
col_no_bg, col_masks = render_image(
    opl_col, masks_col, device_col, psf_pc, pc_config,
    supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)

# Render with different Perlin backgrounds
rng = np.random.default_rng(42)
perlin_bgs = [random_perlin_background((native_h, native_w), rng=rng) for _ in range(3)]

col_with_bgs = []
for bg in perlin_bgs:
    img, _ = render_image(
        opl_col, masks_col, device_col, psf_pc, pc_config,
        supersampling=supersampling,
        camera=None,
        perlin_background=bg,
        rng=np.random.default_rng(42),
    )
    col_with_bgs.append(img)

# --- Mother machine rendering ---
opl_mm, masks_mm, device_mm, _ = draw_scene_supersampled(
    sim_mm, pixel_scale=pixel_scale, supersampling=supersampling,
)
native_h_mm = opl_mm.shape[0] // supersampling
native_w_mm = opl_mm.shape[1] // supersampling

# Without background
mm_no_bg, mm_masks = render_image(
    opl_mm, masks_mm, device_mm, psf_pc, pc_config,
    supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)

# With Perlin background
mm_bg = random_perlin_background((native_h_mm, native_w_mm), rng=np.random.default_rng(99))
mm_with_bg, _ = render_image(
    opl_mm, masks_mm, device_mm, psf_pc, pc_config,
    supersampling=supersampling,
    camera=None,
    perlin_background=mm_bg,
    rng=np.random.default_rng(42),
)

# ============================================================
# Step 4: Comparison figure
# ============================================================
print("\nStep 4: Creating comparison figures...")

# Colony comparison
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

axes[0, 0].imshow(col_no_bg, cmap="Greys_r")
axes[0, 0].set_title("Colony - No Background", fontsize=11)
axes[0, 0].axis("off")

for i, img in enumerate(col_with_bgs):
    axes[0, 1+i].imshow(img, cmap="Greys_r")
    axes[0, 1+i].set_title(f"Colony - Perlin BG #{i+1}", fontsize=11)
    axes[0, 1+i].axis("off")

# Mother machine comparison
axes[1, 0].imshow(mm_no_bg, cmap="Greys_r")
axes[1, 0].set_title("Mother Machine - No BG", fontsize=11)
axes[1, 0].axis("off")

axes[1, 1].imshow(mm_with_bg, cmap="Greys_r")
axes[1, 1].set_title("Mother Machine - Perlin BG", fontsize=11)
axes[1, 1].axis("off")

# Fluorescence (colony)
fl_config = RenderConfig(
    cell_multiplier=3.0,
    defocus=1.0,
    noise_var=0.0005,
    border_expansion=1.5,
)

fl_image, _ = render_image(
    opl_col, masks_col, np.zeros_like(device_col), psf_fl, fl_config,
    supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)
axes[1, 2].imshow(fl_image, cmap="Greens")
axes[1, 2].set_title("Colony - Fluorescence", fontsize=11)
axes[1, 2].axis("off")

# Fluorescence with molecule sampling
fl_sampled, _ = render_fluorescence(
    opl_col, masks_col, psf_fl, fl_config,
    fl_density=0.05, supersampling=supersampling,
    camera=None, rng=np.random.default_rng(42),
)
axes[1, 3].imshow(fl_sampled, cmap="Greens")
axes[1, 3].set_title("Colony - FL (Sampled)", fontsize=11)
axes[1, 3].axis("off")

plt.suptitle("Phase 5: Colony + Trench Rendering with Perlin Backgrounds", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/phase5_output.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved phase5_output.png")

# ============================================================
# Step 5: Attenuation effect
# ============================================================
print("\nStep 5: Showing effect of Perlin noise attenuation...")

attenuations = [1.5, 2.0, 3.0, 5.0, 10.0]
bg_fixed = generate_perlin_background(
    (native_h, native_w),
    NoiseConfig(scale=4, octaves=12, persistence=1.5, lacunarity=1.7, seed=123),
)

fig, axes = plt.subplots(1, len(attenuations) + 1, figsize=(4 * (len(attenuations) + 1), 6))

axes[0].imshow(col_no_bg, cmap="Greys_r")
axes[0].set_title("No Background", fontsize=11)
axes[0].axis("off")

for col, att in enumerate(attenuations, 1):
    img, _ = render_image(
        opl_col, masks_col, device_col, psf_pc, pc_config,
        supersampling=supersampling,
        camera=None,
        perlin_background=bg_fixed,
        perlin_attenuation=att,
        perlin_blur_sigma=2.0,
        rng=np.random.default_rng(42),
    )
    axes[col].imshow(img, cmap="Greys_r")
    axes[col].set_title(f"Attenuation: {att}", fontsize=11)
    axes[col].axis("off")

plt.suptitle("Effect of Perlin Noise Attenuation on Colony Image", fontsize=14)
plt.tight_layout()
plt.savefig("/home/user/SyMBac_2/examples/phase5_attenuation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved phase5_attenuation.png")

print("\nDone! Phase 5 colony rendering complete.")
