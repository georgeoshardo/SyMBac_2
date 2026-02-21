"""Run all 3 tiers of feature matching optimization and compare results.

Tier 1: Handcrafted features + differential_evolution
Tier 2: Handcrafted features + dual_annealing
Tier 3: Neural perceptual + handcrafted combined (if torch available)

Uses the current simulation geometry (w=31, L=380, r=14, max_len=173)
and optimizes the 7 rendering parameters to match the real 100x image.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymunk import Vec2d
from skimage.transform import resize
from tqdm import tqdm

from symbac.simulation import Simulator
from symbac.simulation.config import CellConfig, PhysicsConfig
from symbac.simulation.simcell import SimCell
from symbac.simulation.microfluidic_geometry import trench_creator
from symbac.imaging.drawing import draw_scene_supersampled
from symbac.imaging.optics import PSFModel, Camera
from symbac.imaging.renderer import RenderConfig, render_image
from symbac.imaging.feature_matching import (
    ImageFeatures, feature_distance, compare_images, plot_comparison,
    optimize_render_params, OptimizationBounds,
)

np.random.seed(42)

# ============================================================
# Load real image
# ============================================================
real_img = np.array(Image.open("/home/user/SyMBac_2/old/src/sample_images/sample_100x.tiff"))
real_norm = real_img.astype(np.float64) / real_img.max()
print(f"Real image: {real_img.shape}, range {real_img.min()}-{real_img.max()}")

# ============================================================
# Simulation
# ============================================================
TRENCH_WIDTH = 31
TRENCH_LENGTH = 380
SIM_STEPS = 4000

physics_config = PhysicsConfig(ITERATIONS=100, DAMPING=0.3)
cell_config = CellConfig(
    GRANULARITY=4, SEGMENT_RADIUS=14, SEGMENT_MASS=1.0,
    GROWTH_RATE=5, BASE_MAX_LENGTH=173, MAX_LENGTH_VARIATION=0.15,
    MIN_LENGTH_AFTER_DIVISION=4, NOISE_STRENGTH=0.05, SEED_CELL_SEGMENTS=20,
    ROTARY_LIMIT_JOINT=True, MAX_BEND_ANGLE=0.005, STIFFNESS=300_000,
    PIVOT_JOINT_STIFFNESS=5000, START_POS=Vec2d(0, 50),
    START_ANGLE=np.pi / 2, SIMPLE_LENGTH=True,
)


def growth_hook(cell: SimCell):
    compression = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression ** 4


sim = Simulator(physics_config, cell_config)
sim.add_and_run_post_init_hook(lambda s: trench_creator(TRENCH_WIDTH, TRENCH_LENGTH, (0, 0), s.space))
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
print(f"Simulation: {sim.num_cells} cells")

# ============================================================
# PSF & scene
# ============================================================
pixel_scale = 0.65
supersampling = 3
ss_pixel_scale = pixel_scale / supersampling

psf_pc = PSFModel.phase_contrast(
    wavelength=0.70, NA=1.49, n=1.515,
    condenser="Ph3", apo_sigma=20,
    radius=50, pixel_scale=ss_pixel_scale,
)

opl_ss, masks_ss, device_ss, native_size = draw_scene_supersampled(
    sim, pixel_scale=pixel_scale, supersampling=supersampling,
)
print(f"Scene: {opl_ss.shape}, native: {native_size}")

# ============================================================
# Baseline (user's manual params)
# ============================================================
config_manual = RenderConfig(
    media_multiplier=-28.21, cell_multiplier=50.00,
    device_multiplier=16.22, defocus=10.0,
    noise_var=0.0, border_expansion=1.5,
)

pc_manual, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, config_manual,
    supersampling=supersampling, rng=np.random.default_rng(42),
)
manual_resized = resize(pc_manual, real_norm.shape[:2], anti_aliasing=True, preserve_range=True)
manual_report = compare_images(real_norm, manual_resized)
print(f"\nBaseline (manual): distance={manual_report['overall_distance']:.4f}, "
      f"SSIM={manual_report.get('ssim', 'N/A')}")

# ============================================================
# Optimization bounds — search both sign conventions
# ============================================================
bounds = OptimizationBounds(
    media_multiplier=(-100.0, 100.0),
    cell_multiplier=(-100.0, 100.0),
    device_multiplier=(-100.0, 100.0),
    defocus=(0.0, 20.0),
    noise_var=(0.0, 0.005),
    halo_top_intensity=(0.8, 1.2),
    halo_bottom_intensity=(0.8, 1.2),
)

# ============================================================
# Tier 1: Handcrafted features + differential_evolution
# ============================================================
print("\n" + "=" * 60)
print("TIER 1: Handcrafted features + differential_evolution")
print("=" * 60)

result_t1 = optimize_render_params(
    real_norm, opl_ss, masks_ss, device_ss, psf_pc,
    bounds=bounds, supersampling=supersampling,
    method="differential_evolution",
    maxiter=30, popsize=10, seed=42,
    verbose=True,
)

pc_t1, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, result_t1.best_config,
    supersampling=supersampling, rng=np.random.default_rng(42),
)
t1_resized = resize(pc_t1, real_norm.shape[:2], anti_aliasing=True, preserve_range=True)
t1_report = compare_images(real_norm, t1_resized)
print(f"Tier 1 result: distance={t1_report['overall_distance']:.4f}, "
      f"SSIM={t1_report.get('ssim', 'N/A')}")

# ============================================================
# Tier 2: Handcrafted features + dual_annealing
# ============================================================
print("\n" + "=" * 60)
print("TIER 2: Handcrafted features + dual_annealing")
print("=" * 60)

result_t2 = optimize_render_params(
    real_norm, opl_ss, masks_ss, device_ss, psf_pc,
    bounds=bounds, supersampling=supersampling,
    method="dual_annealing",
    maxiter=200, seed=42,
    verbose=True,
)

pc_t2, _ = render_image(
    opl_ss, masks_ss, device_ss, psf_pc, result_t2.best_config,
    supersampling=supersampling, rng=np.random.default_rng(42),
)
t2_resized = resize(pc_t2, real_norm.shape[:2], anti_aliasing=True, preserve_range=True)
t2_report = compare_images(real_norm, t2_resized)
print(f"Tier 2 result: distance={t2_report['overall_distance']:.4f}, "
      f"SSIM={t2_report.get('ssim', 'N/A')}")

# ============================================================
# Tier 3: Neural + handcrafted (if torch available)
# ============================================================
t3_available = False
try:
    import torch
    from symbac.imaging.feature_matching import optimize_render_params_neural

    print("\n" + "=" * 60)
    print("TIER 3: Neural perceptual + handcrafted combined")
    print("=" * 60)

    result_t3 = optimize_render_params_neural(
        real_norm, opl_ss, masks_ss, device_ss, psf_pc,
        bounds=bounds, supersampling=supersampling,
        maxiter=20, popsize=8, seed=42,
        perceptual_weight=0.5, feature_weight=0.5,
        verbose=True,
    )

    pc_t3, _ = render_image(
        opl_ss, masks_ss, device_ss, psf_pc, result_t3.best_config,
        supersampling=supersampling, rng=np.random.default_rng(42),
    )
    t3_resized = resize(pc_t3, real_norm.shape[:2], anti_aliasing=True, preserve_range=True)
    t3_report = compare_images(real_norm, t3_resized)
    print(f"Tier 3 result: distance={t3_report['overall_distance']:.4f}, "
          f"SSIM={t3_report.get('ssim', 'N/A')}")
    t3_available = True
except ImportError:
    print("\nTier 3 skipped (torch not installed)")

# ============================================================
# Comparison figure
# ============================================================
n_cols = 5 if t3_available else 4
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 10))

axes[0].imshow(real_norm, cmap="Greys_r")
axes[0].set_title("Real (100x)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(manual_resized, cmap="Greys_r")
axes[1].set_title(f"Manual params\ndist={manual_report['overall_distance']:.3f}\n"
                  f"SSIM={manual_report.get('ssim', 0):.3f}", fontsize=10)
axes[1].axis("off")

axes[2].imshow(t1_resized, cmap="Greys_r")
axes[2].set_title(f"Tier 1: DE\ndist={t1_report['overall_distance']:.3f}\n"
                  f"SSIM={t1_report.get('ssim', 0):.3f}\n"
                  f"({result_t1.n_evaluations} evals)", fontsize=10)
axes[2].axis("off")

axes[3].imshow(t2_resized, cmap="Greys_r")
axes[3].set_title(f"Tier 2: DA\ndist={t2_report['overall_distance']:.3f}\n"
                  f"SSIM={t2_report.get('ssim', 0):.3f}\n"
                  f"({result_t2.n_evaluations} evals)", fontsize=10)
axes[3].axis("off")

if t3_available:
    axes[4].imshow(t3_resized, cmap="Greys_r")
    axes[4].set_title(f"Tier 3: Neural+HC\ndist={t3_report['overall_distance']:.3f}\n"
                      f"SSIM={t3_report.get('ssim', 0):.3f}\n"
                      f"({result_t3.n_evaluations} evals)", fontsize=10)
    axes[4].axis("off")

plt.suptitle("Feature Matching Optimization — All Tiers", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/user/SyMBac_2/examples/optimization_all_tiers.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved optimization_all_tiers.png")

# ============================================================
# Print best configs
# ============================================================
param_names = ["media_multiplier", "cell_multiplier", "device_multiplier",
               "defocus", "noise_var", "halo_top_intensity", "halo_bottom_intensity"]

print("\n=== Best Configurations ===")
for name, result in [("Tier 1 (DE)", result_t1), ("Tier 2 (DA)", result_t2)]:
    print(f"\n{name} (distance={result.best_distance:.4f}):")
    for p in param_names:
        print(f"  {p}: {getattr(result.best_config, p):.4f}")

if t3_available:
    print(f"\nTier 3 (Neural+HC) (distance={result_t3.best_distance:.4f}):")
    for p in param_names:
        print(f"  {p}: {getattr(result_t3.best_config, p):.4f}")

# Convergence plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(result_t1.convergence_history, label=f"Tier 1: DE (best={result_t1.best_distance:.3f})", alpha=0.7)
ax.plot(result_t2.convergence_history, label=f"Tier 2: DA (best={result_t2.best_distance:.3f})", alpha=0.7)
if t3_available:
    ax.plot(result_t3.convergence_history, label=f"Tier 3: Neural (best={result_t3.best_distance:.3f})", alpha=0.7)
ax.set_xlabel("Evaluation")
ax.set_ylabel("Feature Distance")
ax.set_title("Optimization Convergence")
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("/home/user/SyMBac_2/examples/optimization_convergence.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved optimization_convergence.png")

print("\nDone!")
