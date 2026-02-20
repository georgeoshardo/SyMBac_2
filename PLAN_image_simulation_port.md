# SyMBac_2 Image Simulation Port: Implementation Plan

## Context

SyMBac_2 currently contains only the **cell physics simulation** (bendy cells with a segment-based model, hook-driven API, pymunk physics). The original SyMBac had a complete **image simulation pipeline** that took cell simulation output and produced photorealistic synthetic microscopy images (phase contrast, fluorescence) with accompanying segmentation masks. This plan describes how to port that image simulation capability into the new SyMBac_2 architecture.

### What exists in SyMBac_2 today (new code)
- `Simulator` with hook-based step loop
- `SimCell` with segment-based `PhysicsRepresentation` (chains of circles + joints)
- `Colony`, `GrowthManager`, `DivisionManager`
- `CellConfig`, `PhysicsConfig` (frozen dataclasses)
- `LiveVisualisation` (pygame real-time viewer)
- Microfluidic geometry builders (`trench_creator`, `box_creator`)
- `SimulationLogger` extension (example)

### What exists in old SyMBac (to be ported)
1. **Cell rasterization** (`drawing.py`) - `raster_cell()`, `draw_scene()`, OPL generation, cell bending transforms, mask generation
2. **PSF generation** (`PSF.py`) - `PSF_generator` class (phase contrast, simple fluorescence, 3D fluorescence), `Camera` class
3. **Rendering pipeline** (`renderer.py`) - `Renderer` class: OPL-to-image convolution, intensity tuning, noise models, histogram/Fourier matching, training data export
4. **Colony rendering** (`colony_renderer.py`) - `ColonyRenderer` for large-scale colony image rendering with Perlin noise backgrounds
5. **Spectral matching** (`pySHINE.py`) - `sfMatch()`, `lumMatch()` for matching synthetic image statistics to real images
6. **Fluorescence conversion** (`drawing.py`) - `OPL_to_FL()` for sparse fluorescent molecule sampling

### Key architectural difference
The old code used **polygon-based cells** with properties (length, width, angle, centroid, pinching_sep) extracted per-frame, then rasterized straight capsule shapes with optional sinusoidal bending transforms applied during drawing. The new code uses **segment-chain cells** that are *already* physically bendy - the curvature is intrinsic to the physics, not a post-hoc drawing transform.

---

## Porting Strategy Overview

The port is organized into **6 phases**, each producing a usable increment. The design principles are:

1. **Fit the new API** - use hooks, configs, and the existing module structure; don't bolt on old patterns
2. **Leverage intrinsic curvature** - new cells are already bendy, so the old sinusoidal bending transform is unnecessary; rasterize the actual segment chain geometry
3. **Decouple** - keep PSF, camera, rendering, and drawing as independent modules (not monolithic classes)
4. **No napari dependency in core** - napari was used in the old `Renderer` for interactive intensity tuning; replace with a programmatic API (napari can be an optional visualization)
5. **GPU optional** - support CPU-only with optional CuPy acceleration (same as old code but cleaner)
6. **No deep learning code in core** - the DeLTA/TensorFlow code from old SyMBac is out of scope

---

## Phase 1: Scene Rasterization (OPL + Masks from Segment Chains)

**Goal:** Given a `Simulator` state at any frame, produce a 2D OPL (Optical Path Length) image and a corresponding labelled mask.

### 1.1 New module: `src/symbac/imaging/drawing.py`

Port `raster_cell()` and create a new `rasterize_segment_chain()` function.

**What to port directly (with cleanup):**
- `raster_cell(length, width, separation, pinching, FL)` - the core capsule rasterization that produces an OPL image of a single straight cell. This is a pure function with no dependencies on the old simulation. Port it as-is.
- `OPL_to_FL(cell, density)` - fluorescence conversion via molecule sampling. Port as-is (it's a pure numba function).
- `get_crop_bounds_2D()`, `crop_image()` - utility functions. Port as-is.

**What to create new:**
- `rasterize_segment_chain(cell: SimCell, pixel_scale: float, supersampling: int = 3) -> tuple[np.ndarray, np.ndarray]`
  - Instead of rasterizing a single straight capsule and then applying a sinusoidal bend, we rasterize the actual segment chain from the physics simulation.
  - For each segment in `cell.physics_representation.segments`, draw a filled circle at its position with radius = `segment.radius`, with intensity proportional to the OPL (cross-sectional thickness of a sphere at that point).
  - Use supersampling (render at Nx resolution, then downsample) for anti-aliasing, matching the old `resize_amount` concept.
  - Returns (opl_image, binary_mask) for a single cell.
  - The key insight: because the new cells are chains of overlapping spheres, the OPL at each pixel is the sum of sphere cross-sections intersecting that pixel column. For a sphere of radius R at distance d from center: `thickness = 2 * sqrt(R^2 - d^2)` when `d < R`.

- `draw_scene(simulator: Simulator, pixel_scale: float, supersampling: int = 3, label_masks: bool = True) -> tuple[np.ndarray, np.ndarray]`
  - Iterates over all cells in `simulator.colony.cells`
  - Calls `rasterize_segment_chain()` for each cell
  - Composites them into a scene-sized canvas
  - Produces (scene_opl, scene_masks)
  - The scene size is determined from the bounding box of all cell positions + padding

- `draw_scene_with_geometry(simulator: Simulator, pixel_scale: float, ...) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Same as above but also draws the microfluidic device (trench walls) into the OPL image, using the static bodies from `simulator.space`
  - Returns (scene_opl, scene_masks, device_mask)
  - The device mask is needed later for intensity tuning (media vs device vs cell regions)

**What NOT to port:**
- `generate_curve_props()`, `gen_cell_props_for_draw()`, `transform_func()` - these were the sinusoidal bending system. The new cells are already bendy; no need.
- `place_cell()` - old utility, replaced by composite drawing.
- `get_space_size()` - replace with a simpler bounding-box calculation from segment positions.
- `draw_simulation_OPL()` from `simulation.py` - this was the old orchestrator; replace with the new `draw_scene()`.

### 1.2 New module: `src/symbac/imaging/__init__.py`

Package init for the imaging subpackage.

### 1.3 Integration point: Logging hook

Create an `OPLLogger` that can be registered as a `post_step_hook` to capture OPL scenes each frame:

```python
class OPLLogger:
    def __init__(self, pixel_scale, supersampling=3):
        self.pixel_scale = pixel_scale
        self.supersampling = supersampling
        self.opl_scenes = []
        self.masks = []

    def log_frame(self, simulator):
        opl, mask = draw_scene(simulator, self.pixel_scale, self.supersampling)
        self.opl_scenes.append(opl)
        self.masks.append(mask)
```

### 1.4 Perlin noise texture

Port the cell-internal Perlin noise texture from the old `draw_scene()`. In the old code, each cell had a `texture_y_coordinate` that created continuous Perlin noise along the cell body. For the new segment-chain rasterizer, apply Perlin noise modulation to the OPL values based on the cell's position in the scene, preserving the visual texture.

**Dependencies to add:** `noise` (Perlin noise library, already used by old code)

---

## Phase 2: PSF and Camera

**Goal:** Port the PSF generator and camera model as standalone, clean modules.

### 2.1 New module: `src/symbac/imaging/optics.py`

Port the `PSF_generator` class and `Camera` class from `old/src/PSF.py`.

**What to port (with refactoring):**
- `PSF_generator` class -> rename to `PSFModel`
  - `__init__()` - clean up parameter handling. Instead of `mode` string dispatch, use an enum or separate factory methods:
    - `PSFModel.phase_contrast(wavelength, NA, n, condenser, apo_sigma, radius, pixel_scale, offset=0)`
    - `PSFModel.fluorescence_2d(wavelength, NA, n, radius, pixel_scale, offset=0)`
    - `PSFModel.fluorescence_3d(wavelength, NA, n, radius, pixel_scale, z_height, pz=0, working_distance=None, offset=0)`
  - `calculate_PSF()` -> `kernel` property (lazy-computed on first access)
  - `get_phase_contrast_kernel()` - port as-is (static method)
  - `get_fluorescence_kernel()` - port as-is (static method)
  - `get_condensers()` - port as-is (static method)
  - `gaussian_2D()` - port as-is
  - `somb()` - port as-is
  - `plot_PSF()` - port as-is (optional visualization)

- `Camera` class - port as-is, it's already clean:
  - `__init__(baseline, sensitivity, dark_noise)`
  - `render_dark_image(size, plot=True)`

**What to change:**
- Remove the confusing `resize_amount` / `pix_mic_conv` / `scale` triple parameter. The new API should just take `pixel_scale` (microns per pixel at the supersampled resolution, i.e. `pix_mic_conv / supersampling`).
- The old `condenser` parameter validation was a simple string lookup - keep that pattern but validate more explicitly.

### 2.2 New config: `src/symbac/imaging/config.py`

```python
@dataclass(frozen=True)
class ImagingConfig:
    pixel_scale: float          # microns per pixel at native camera resolution
    supersampling: int = 3      # internal rendering upscale factor
    wavelength: float = 0.75    # imaging wavelength in microns
    NA: float = 1.2             # numerical aperture
    n: float = 1.3              # refractive index
    psf_radius: int = 50        # PSF kernel radius in supersampled pixels

@dataclass(frozen=True)
class PhaseContrastConfig(ImagingConfig):
    condenser: str = "Ph3"
    apo_sigma: float = 10.0
    psf_offset: float = 0.0

@dataclass(frozen=True)
class FluorescenceConfig(ImagingConfig):
    fl_density: float = 1.0     # fluorescent molecules per volume element
    z_height: int | None = None # for 3D PSF mode
```

**Dependencies:** `psfmodels` (for 3D fluorescence PSF), `scipy` (for Bessel functions)

---

## Phase 3: Convolution and Core Rendering

**Goal:** Port the convolution pipeline that takes OPL scenes + PSF and produces raw synthetic images.

### 3.1 New module: `src/symbac/imaging/convolution.py`

Port `convolve_rescale()` from `old/src/renderer.py`.

**What to port:**
- `convolve_rescale(image, kernel, rescale_factor, rescale_int)` - the dual CPU/GPU implementation
  - Clean up: single function with `backend="auto"` parameter that detects CuPy availability
  - Keep the same logic: convolve at high-res, then rescale down

**Pattern for GPU optional:**
```python
def convolve_rescale(image, kernel, rescale_factor, rescale_int=True, backend="auto"):
    if backend == "auto":
        backend = "gpu" if _cupy_available else "cpu"
    if backend == "gpu":
        # CuPy path
    else:
        # scipy.signal.convolve2d path
```

### 3.2 New module: `src/symbac/imaging/renderer.py`

This is the core rendering pipeline. Port the key methods from `old/src/renderer.py`'s `Renderer` class, but decomposed into functions rather than one monolithic class.

**Functions to create:**

- `generate_pc_opl(scene_opl, device_mask, media_multiplier, cell_multiplier, device_multiplier, border_expansion) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Port from `Renderer.generate_PC_OPL()`.
  - Instead of relying on `simulation.main_segments` and `simulation.offset`, take the OPL scene and device mask directly.
  - Multiplies different image regions (media, cells, device) by their respective intensity factors.
  - Expands borders for convolution edge effects.
  - Returns (expanded_scene, expanded_scene_no_cells, expanded_mask).

- `render_image(opl_scene, mask, psf_model, config, camera=None, noise_params=None) -> tuple[np.ndarray, np.ndarray]`
  - High-level function that orchestrates the full pipeline:
    1. Apply region intensity multipliers via `generate_pc_opl()`
    2. Optionally apply halo effect
    3. Apply defocus (gaussian blur on PSF kernel)
    4. Convolve with PSF via `convolve_rescale()`
    5. Downsample to native resolution
    6. Apply noise (camera model or ad-hoc)
    7. Return (synthetic_image, downsampled_mask)

- `apply_noise(image, camera=None, noise_var=0.001) -> np.ndarray`
  - Port the noise application logic from `generate_test_comparison()`
  - Camera-based noise: Poisson + Gaussian from Camera model
  - Ad-hoc noise: Gaussian + Poisson via `skimage.util.random_noise`

- `apply_halo(image, top_intensity, bottom_intensity, start_frac, end_frac) -> np.ndarray`
  - Port the halo line profile from `generate_test_comparison()`
  - Creates a linear intensity ramp to simulate microfluidic device optical effects

### 3.3 New module: `src/symbac/imaging/spectral_matching.py`

Port `pySHINE.py` functions.

**What to port:**
- `sfMatch(images, tarmag=None)` -> `match_fourier_spectrum(images, target_magnitude=None)`
- `lumMatch(images, mask=None, lum=None)` -> `match_luminance(images, mask=None, target_lum=None)`
- `cart2pol()`, `pol2cart()` - port as-is (utility functions)
- `rescale_shine()` -> `rescale_images()`

These are pure functions with numpy/scipy dependencies only. Straightforward port with naming cleanup.

---

## Phase 4: Training Data Generation

**Goal:** Port the batch rendering and training data export pipeline.

### 4.1 New module: `src/symbac/imaging/training_data.py`

Port `Renderer.generate_training_data()` and related methods.

**What to create:**

- `RenderConfig` dataclass:
  ```python
  @dataclass
  class RenderConfig:
      media_multiplier: float = 30.0
      cell_multiplier: float = 1.7
      device_multiplier: float = 29.0
      defocus: float = 3.0
      noise_var: float = 0.001
      apo_sigma: float = 8.85
      match_histogram: bool = True
      match_fourier: bool = False
      match_noise: bool = False
      halo_top_intensity: float = 1.0
      halo_bottom_intensity: float = 1.0
      halo_start: float = 0.0
      halo_end: float = 1.0
  ```

- `generate_training_data(opl_scenes, masks, psf_model, render_config, real_image, camera=None, n_samples=500, burn_in=40, sample_variation=0.2, save_dir=None, randomize_histogram=True, randomize_noise=True, n_jobs=1) -> list[tuple[np.ndarray, np.ndarray]]`
  - Port from `Renderer.generate_training_data()`
  - Takes pre-computed OPL scenes and masks (from Phase 1)
  - Renders each with parameter variations
  - Optionally saves to disk
  - Uses joblib for parallelization (keep, but drop the Ray dependency from `colony_renderer.py`)

- `match_image_statistics(synthetic, real_image, match_histogram=True, match_fourier=False) -> np.ndarray`
  - Combines histogram matching and Fourier spectrum matching
  - Port from the matching logic in `generate_test_comparison()`

### 4.2 Interactive tuning (optional, lower priority)

The old code used `ipywidgets.interactive` + napari for parameter tuning. This can be ported as an optional Jupyter notebook utility but should NOT be in the core API.

- `src/symbac/imaging/interactive.py` (optional)
  - Provides a function to create an interactive widget for tuning render parameters
  - Not required for the core pipeline

---

## Phase 5: Colony Rendering

**Goal:** Port the `ColonyRenderer` for large-scale colony image synthesis.

### 5.1 New module: `src/symbac/imaging/colony_renderer.py`

Port from `old/src/colony_renderer.py`, but significantly simplified.

**What to port:**
- `perlin_generator(shape, scale, octaves, persistence, lacunarity)` -> standalone function
- `random_perlin_generator(shape)` -> standalone function
- `render_scene(opl, psf_model, config)` -> function (not method)
  - Handles phase contrast background noise via Perlin
  - Handles 3D PSF slice-by-slice convolution
  - Gaussian filter for PSF defocus

**What NOT to port:**
- `generate_random_samples_ray()` - Ray dependency is heavyweight. Use joblib instead.
- Direct CuPy dependency in the class - use the `convolve_rescale()` abstraction from Phase 3.

### 5.2 Integration

The colony renderer should work with the same `draw_scene()` output from Phase 1. The only difference from the trench renderer is:
- No device geometry (or different geometry)
- Perlin noise backgrounds
- Potentially much larger scene sizes

---

## Phase 6: Integration, Testing, and Examples

### 6.1 End-to-end integration

Create a high-level convenience class or function that ties everything together:

```python
# src/symbac/imaging/pipeline.py

class ImageSimulator:
    """High-level API for generating synthetic microscopy images from a simulation."""

    def __init__(self, imaging_config, psf_model, camera=None, real_image=None):
        ...

    def render_frame(self, simulator, render_config=None) -> tuple[np.ndarray, np.ndarray]:
        """Render a single frame from current simulator state."""
        opl, mask = draw_scene(simulator, self.imaging_config.pixel_scale, self.imaging_config.supersampling)
        return render_image(opl, mask, self.psf_model, self.imaging_config, self.camera)

    def render_simulation(self, opl_scenes, masks, render_config, n_samples=None) -> list:
        """Batch render from pre-recorded OPL scenes."""
        ...

    def generate_training_data(self, opl_scenes, masks, render_config, save_dir, ...) -> None:
        """Generate and save training data pairs."""
        ...
```

### 6.2 Example scripts

- `examples/image_simulation_basic.py` - Simple phase contrast image from a trench simulation
- `examples/image_simulation_fluorescence.py` - Fluorescence imaging example
- `examples/training_data_generation.py` - Full pipeline: simulate -> render -> export training data

### 6.3 Tests

- `tests/test_drawing.py` - Test rasterization of single cells and scenes
- `tests/test_optics.py` - Test PSF generation (phase contrast, fluorescence)
- `tests/test_convolution.py` - Test convolve_rescale with known inputs
- `tests/test_renderer.py` - Test full rendering pipeline
- `tests/test_spectral_matching.py` - Test Fourier/luminance matching

---

## New File Structure

```
src/symbac/
    imaging/
        __init__.py
        config.py               # ImagingConfig, PhaseContrastConfig, FluorescenceConfig, RenderConfig
        drawing.py              # raster_cell, rasterize_segment_chain, draw_scene, OPL_to_FL
        optics.py               # PSFModel, Camera
        convolution.py          # convolve_rescale (CPU/GPU)
        renderer.py             # generate_pc_opl, render_image, apply_noise, apply_halo
        spectral_matching.py    # match_fourier_spectrum, match_luminance
        colony_renderer.py      # perlin_generator, render_colony_scene
        training_data.py        # generate_training_data, match_image_statistics
        pipeline.py             # ImageSimulator (high-level API)
        interactive.py          # (optional) Jupyter widget for parameter tuning
```

---

## Dependency Changes

**New required dependencies:**
- `scipy` (for `convolve2d`, `gaussian_filter`, `jv` Bessel functions) - likely already a dependency
- `scikit-image` (for `rescale`, `rescale_intensity`, `match_histograms`, `random_noise`, `rotate`)
- `psfmodels` (for 3D fluorescence PSF generation)
- `noise` (for Perlin noise generation)
- `Pillow` (for image I/O)
- `numba` (for `@njit` on `OPL_to_FL`)

**Optional dependencies:**
- `cupy` (GPU acceleration for convolution)
- `napari` (visualization only)
- `ipywidgets` (interactive tuning only)

**Removed dependencies (not porting):**
- `ray` (was used in `colony_renderer.py` for multi-GPU; replace with joblib)
- `tensorflow` / deep learning code (out of scope)
- `pyglet` (old visualization; new code uses pygame)
- `CellModeller` (old simulation dependency)

---

## Key Design Decisions

### 1. Segment chain rasterization vs. capsule rasterization

The biggest architectural question. Two approaches:

**Option A: Rasterize the segment chain directly (recommended)**
- For each segment (circle), compute the sphere OPL contribution at each pixel
- Sum overlapping contributions (the segments overlap because JOINT_DISTANCE < 2*SEGMENT_RADIUS)
- Advantages: captures the actual cell shape including bends, no post-hoc transform needed
- Disadvantages: more expensive than rasterizing a single capsule

**Option B: Extract capsule parameters and use the old `raster_cell()`**
- Fit a capsule (length, width, angle, centroid) to each cell's segment chain
- Optionally apply the old sinusoidal bend transform
- Advantages: reuses proven code
- Disadvantages: loses the physics-accurate curvature; the old bending was an approximation

**Decision: Option A** (rasterize segment chains directly). This is the whole point of the new bendy cell model. We still port `raster_cell()` as a utility (it's useful for quick prototyping and for cases where someone wants simple capsule shapes), but the primary rasterization path uses the actual segment positions.

### 2. Scene coordinate system

The old code had a confusing coordinate system with `offset`, `pix_mic_conv`, `resize_amount` all interacting. The new code should use a single clear coordinate system:

- **Physics coordinates**: the pymunk space coordinates (arbitrary units defined by CellConfig)
- **Pixel coordinates**: physics coordinates * `pixel_scale * supersampling`
- **Output coordinates**: pixel coordinates / `supersampling`

The `pixel_scale` (microns per pixel at native camera resolution) is the single conversion factor. `supersampling` is for anti-aliasing during rendering.

### 3. State extraction vs. hook-based logging

Two approaches for getting simulation state into the imaging pipeline:

**Option A: Extract on demand** - Call `draw_scene(simulator)` whenever you want an image. No state storage.

**Option B: Hook-based logging** - Register a post_step_hook that captures cell states each frame, then batch-render later.

**Decision: Support both.** The `draw_scene()` function works on-demand from current simulator state. The `OPLLogger` hook captures time-series for batch rendering. Users choose based on their workflow.

### 4. Where does the trench geometry come from?

The old code extracted trench geometry from the pymunk space using `get_trench_segments()` which looked for static body shapes. The new code should:

- Extract static shapes from `simulator.space` to identify the device geometry
- OR accept an explicit device mask / geometry specification

Both approaches should be supported, with the device geometry extraction being the default for convenience.

---

## Implementation Order and Priorities

| Priority | Phase | Description | Depends on |
|----------|-------|-------------|------------|
| P0 | 1.1 | `raster_cell()` port + `rasterize_segment_chain()` | Nothing |
| P0 | 1.2 | `draw_scene()` | 1.1 |
| P0 | 2.1 | PSF model port | Nothing |
| P0 | 3.1 | `convolve_rescale()` | Nothing |
| P0 | 3.2 | Core renderer (`render_image()`) | 1.2, 2.1, 3.1 |
| P1 | 3.3 | Spectral matching | Nothing |
| P1 | 4.1 | Training data generation | 3.2, 3.3 |
| P1 | 2.2 | Imaging config dataclasses | 2.1 |
| P2 | 5.1 | Colony renderer | 3.1, 3.2 |
| P2 | 6.1 | `ImageSimulator` high-level API | All above |
| P2 | 6.2 | Example scripts | 6.1 |
| P3 | 4.2 | Interactive tuning (Jupyter) | 4.1 |
| P3 | 1.4 | Perlin noise textures on cells | 1.1 |

P0 = must have for basic functionality
P1 = needed for training data generation (the main use case)
P2 = polish and convenience
P3 = nice to have

---

## Risk and Open Questions

1. **Segment chain OPL accuracy** - The overlapping spheres in the segment chain will produce a different OPL profile than the smooth capsule of the old code. Need to validate that the resulting images are visually reasonable and comparable. May need to tune segment overlap or use a different OPL model (e.g., cylinder body + hemispherical caps fit to the chain).

2. **Performance of segment chain rasterization** - Rasterizing N circles per cell is more expensive than one capsule. For large colonies this could be slow. Mitigation: use vectorized numpy operations, or numba JIT compilation.

3. **Coordinate system mapping** - The new `PhysicsRepresentation` uses pymunk coordinates directly. The old code had a `pix_mic_conv * resize_amount` scaling. Need to carefully define and document the coordinate transform between physics space and image space.

4. **Division pinching visualization** - The old code used the `separation` parameter in `raster_cell()` to show division constriction. The new code has `septum_progress` which constricts segment radii. The segment chain rasterizer naturally shows this (segments get smaller), but it may look different from the old visualization.

5. **Device geometry extraction** - The old code relied on `get_trench_segments()` to find the trench walls in the pymunk space. Need to verify this works with the new `trench_creator()` / `box_creator()` geometry, or adapt the extraction logic.

6. **Backward compatibility** - Some users may have workflows built around the old API. Consider providing a compatibility shim or migration guide, but don't let this constrain the new design.
