"""Interactive Napari viewer demo for rendering parameter optimisation.

Runs a short mother machine simulation, records frames, then opens the
Napari viewer so you can tune rendering parameters with sliders and
visually compare the synthetic image against a real image (if provided).

Usage:
    python examples/napari_viewer_demo.py
"""

import numpy as np
from pymunk import Vec2d
from functools import partial

# --- Simulation ---
from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator

# --- Imaging ---
from symbac.imaging import (
    PSFModel, Camera, RenderConfig,
    draw_scene_supersampled, NoiseConfig,
    generate_perlin_background,
    record_simulation_frames,
    launch_viewer,
)

np.random.seed(42)

# ============================================================
# 1.  Simulation
# ============================================================
print("Setting up simulation...")
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
# 2.  Record frames
# ============================================================
pixel_scale = 0.5
supersampling = 3
ss_pixel_scale = pixel_scale / supersampling

draw_fn = partial(
    draw_scene_supersampled,
    pixel_scale=pixel_scale,
    supersampling=supersampling,
)


def draw_3tuple(sim):
    opl, masks, device, _ = draw_fn(sim)
    return opl, masks, device


print("Recording simulation frames...")
opl_scenes, mask_scenes, device_masks = record_simulation_frames(
    sim,
    draw_fn=draw_3tuple,
    n_steps=500,
    record_every=10,
    desc="Recording",
)
print(f"Recorded {len(opl_scenes)} frames  (shape: {opl_scenes[0].shape})")

# ============================================================
# 3.  Optics
# ============================================================
psf_pc = PSFModel.phase_contrast(
    wavelength=0.75, NA=1.2, n=1.3,
    condenser="Ph3", apo_sigma=10,
    radius=40, pixel_scale=ss_pixel_scale,
)

camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)

# Starting render config — tweak these in the viewer!
config = RenderConfig(
    media_multiplier=30.0,
    cell_multiplier=-5.0,
    device_multiplier=-50.0,
    defocus=1.0,
    noise_var=0.001,
    border_expansion=1.5,
)

# ============================================================
# 4.  Launch Napari viewer
# ============================================================
print("Launching Napari viewer — adjust sliders, then close the window.")
print("The final RenderConfig will be printed below.\n")

final_config = launch_viewer(
    opl_scenes=opl_scenes,
    mask_scenes=mask_scenes,
    device_masks=device_masks,
    psf_model=psf_pc,
    render_config=config,
    supersampling=supersampling,
    real_image=None,       # pass a real image here for comparison
    camera=None,           # or pass camera for camera noise
)

print("\n=== Final RenderConfig ===")
print(f"  media_multiplier     = {final_config.media_multiplier}")
print(f"  cell_multiplier      = {final_config.cell_multiplier}")
print(f"  device_multiplier    = {final_config.device_multiplier}")
print(f"  defocus              = {final_config.defocus}")
print(f"  noise_var            = {final_config.noise_var}")
print(f"  border_expansion     = {final_config.border_expansion}")
print(f"  halo_top_intensity   = {final_config.halo_top_intensity}")
print(f"  halo_bottom_intensity= {final_config.halo_bottom_intensity}")
