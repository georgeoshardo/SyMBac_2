"""Phase 1 Demo: OPL rasterization from 3 simulation scenarios.

Generates OPL and mask images for:
1. Bacteria in a mother machine trench
2. Bacteria in a colony (open box)
3. Bacteria in a large microfluidic chip
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pymunk import Vec2d
from tqdm import tqdm

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator, box_creator
from symbac.imaging.drawing import draw_scene, draw_scene_with_geometry

np.random.seed(42)

# --- Shared physics config ---
physics_config = PhysicsConfig(ITERATIONS=100, DAMPING=0.3)


def make_cell_config(**overrides):
    defaults = dict(
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
        SIMPLE_LENGTH=True,
    )
    defaults.update(overrides)
    return CellConfig(**defaults)


def growth_hook(cell: SimCell) -> None:
    compression = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression ** 4
    variation = cell.config.BASE_MAX_LENGTH * cell.config.MAX_LENGTH_VARIATION
    random_max_len = np.random.uniform(
        cell.config.BASE_MAX_LENGTH - variation,
        cell.config.BASE_MAX_LENGTH + variation,
    ) * np.sqrt(compression)
    cell.max_length = max(cell.length, int(random_max_len))


def run_simulation(simulator, n_steps, desc="Simulating"):
    for _ in tqdm(range(n_steps), desc=desc):
        simulator.step()


# ============================================================
# Scenario 1: Mother Machine
# ============================================================
print("\n=== Scenario 1: Mother Machine ===")
cell_config_mm = make_cell_config(
    START_POS=Vec2d(0, 50),
    START_ANGLE=np.pi / 2,
)

sim_mm = Simulator(physics_config, cell_config_mm)

def trench_adder(sim):
    trench_creator(30, 500, (0, 0), sim.space)
sim_mm.add_and_run_post_init_hook(trench_adder)

def cell_remover_mm(sim):
    for cell in sim.cells[:]:
        pos_y = cell.physics_representation.segments[0].body.position.y
        if pos_y > 500 or pos_y < -50:
            sim.colony.delete_cell(cell)
sim_mm.add_post_step_hook(cell_remover_mm)
sim_mm.add_pre_cell_grow_hook(growth_hook)

run_simulation(sim_mm, 1500, "Mother Machine")
print(f"  Cells: {sim_mm.num_cells}")

# ============================================================
# Scenario 2: Colony (open box)
# ============================================================
print("\n=== Scenario 2: Colony ===")
cell_config_colony = make_cell_config(
    START_POS=Vec2d(0, 0),
    START_ANGLE=0.3,
    BASE_MAX_LENGTH=90,
    GROWTH_RATE=8,
)

sim_colony = Simulator(physics_config, cell_config_colony)

def colony_box_adder(sim):
    box_creator(600, 600, (0, 0), sim.space, barrier_thickness=10, fillet_radius=50)
sim_colony.add_and_run_post_init_hook(colony_box_adder)

def cell_remover_colony(sim):
    for cell in sim.cells[:]:
        pos = cell.physics_representation.segments[0].body.position
        if abs(pos.x) > 400 or abs(pos.y) > 400:
            sim.colony.delete_cell(cell)
sim_colony.add_post_step_hook(cell_remover_colony)
sim_colony.add_pre_cell_grow_hook(growth_hook)

run_simulation(sim_colony, 1500, "Colony")
print(f"  Cells: {sim_colony.num_cells}")

# ============================================================
# Scenario 3: Large microfluidic chip (wider trench, more cells)
# ============================================================
print("\n=== Scenario 3: Large Microfluidic Chip ===")
cell_config_chip = make_cell_config(
    START_POS=Vec2d(0, 100),
    START_ANGLE=np.pi / 2,
    BASE_MAX_LENGTH=100,
    GROWTH_RATE=6,
)

sim_chip = Simulator(physics_config, cell_config_chip)

def chip_geometry_adder(sim):
    # A wide, long channel
    box_creator(80, 1200, (0, 0), sim.space, barrier_thickness=8, fillet_radius=30)
sim_chip.add_and_run_post_init_hook(chip_geometry_adder)

def cell_remover_chip(sim):
    for cell in sim.cells[:]:
        pos = cell.physics_representation.segments[0].body.position
        if abs(pos.y) > 700 or abs(pos.x) > 200:
            sim.colony.delete_cell(cell)
sim_chip.add_post_step_hook(cell_remover_chip)
sim_chip.add_pre_cell_grow_hook(growth_hook)

run_simulation(sim_chip, 2000, "Microfluidic Chip")
print(f"  Cells: {sim_chip.num_cells}")


# ============================================================
# Render OPL scenes and masks
# ============================================================
print("\n=== Rendering OPL scenes ===")

pixel_scale = 0.5  # microns per pixel

scenarios = [
    ("Mother Machine", sim_mm),
    ("Colony", sim_colony),
    ("Microfluidic Chip", sim_chip),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

for row, (name, sim) in enumerate(scenarios):
    print(f"  Drawing {name}...")
    opl, masks, device = draw_scene_with_geometry(sim, pixel_scale=pixel_scale, supersampling=3)

    # OPL image
    ax = axes[row, 0]
    ax.imshow(opl, cmap='inferno', interpolation='nearest')
    ax.set_title(f"{name} - OPL", fontsize=12)
    ax.axis('off')

    # Labelled mask
    ax = axes[row, 1]
    # Create a random colormap for instance labels
    unique_labels = np.unique(masks)
    n_labels = len(unique_labels)
    rng = np.random.RandomState(42)
    colors = np.zeros((n_labels, 4))
    colors[0] = [0, 0, 0, 1]  # background is black
    for i in range(1, n_labels):
        colors[i] = [*rng.rand(3), 1.0]
    # Map labels to sequential integers for colormap
    label_map = np.zeros(masks.max() + 1, dtype=int)
    for i, lbl in enumerate(unique_labels):
        label_map[lbl] = i
    mapped = label_map[masks]
    cmap = ListedColormap(colors)
    ax.imshow(mapped, cmap=cmap, interpolation='nearest')
    ax.set_title(f"{name} - Instance Masks", fontsize=12)
    ax.axis('off')

    # Device mask overlaid on OPL
    ax = axes[row, 2]
    composite = np.zeros((*opl.shape, 3))
    opl_norm = opl / (opl.max() + 1e-10)
    composite[:, :, 0] = opl_norm  # OPL in red channel
    composite[:, :, 1] = opl_norm * 0.7
    composite[:, :, 2] = device.astype(float) * 0.5  # Device in blue
    composite = np.clip(composite, 0, 1)
    ax.imshow(composite, interpolation='nearest')
    ax.set_title(f"{name} - OPL + Device", fontsize=12)
    ax.axis('off')

plt.suptitle("Phase 1: OPL Rasterization from Segment Chains", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/phase1_output.png", dpi=150, bbox_inches='tight')
print(f"\nSaved to examples/phase1_output.png")
plt.close()

# Also save individual high-res images
for name, sim in scenarios:
    safe_name = name.lower().replace(" ", "_")
    opl, masks, device = draw_scene_with_geometry(sim, pixel_scale=pixel_scale, supersampling=3)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(opl, cmap='inferno')
    ax.set_title(f"{name} - OPL")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/home/user/SyMBac_2/examples/phase1_{safe_name}_opl.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    unique_labels = np.unique(masks)
    n_labels = len(unique_labels)
    rng = np.random.RandomState(42)
    colors = np.zeros((n_labels, 4))
    colors[0] = [0, 0, 0, 1]
    for i in range(1, n_labels):
        colors[i] = [*rng.rand(3), 1.0]
    label_map = np.zeros(masks.max() + 1, dtype=int)
    for i, lbl in enumerate(unique_labels):
        label_map[lbl] = i
    mapped = label_map[masks]
    cmap = ListedColormap(colors)
    ax.imshow(mapped, cmap=cmap)
    ax.set_title(f"{name} - Instance Masks")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/home/user/SyMBac_2/examples/phase1_{safe_name}_masks.png", dpi=150, bbox_inches='tight')
    plt.close()

print("Done! All Phase 1 images saved.")
