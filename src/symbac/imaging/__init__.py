"""SyMBac imaging package: synthetic microscopy image generation from cell simulations.

This package provides a complete pipeline for generating synthetic microscopy
images from physics-based cell simulations. It supports phase contrast and
fluorescence modalities with physically-based PSF models, camera noise, and
optional Perlin noise backgrounds for colony/agar pad rendering.

Typical usage:

    from symbac.imaging.drawing import draw_scene_supersampled
    from symbac.imaging.optics import PSFModel, Camera
    from symbac.imaging.renderer import RenderConfig, render_image
    from symbac.imaging.noise import random_perlin_background
"""

from symbac.imaging.optics import PSFModel, Camera, PSFMode
from symbac.imaging.renderer import RenderConfig, render_image, render_fluorescence
from symbac.imaging.drawing import draw_scene_supersampled, draw_scene
from symbac.imaging.noise import (
    NoiseConfig,
    generate_perlin_background,
    random_perlin_background,
)
from symbac.imaging.training_data import (
    TrainingDataConfig,
    generate_training_data,
    record_simulation_frames,
)
from symbac.imaging.feature_matching import (
    ImageFeatures,
    feature_distance,
    OptimizationBounds,
    OptimizationResult,
    optimize_render_params,
    compare_images,
    plot_comparison,
)
from symbac.imaging.napari_viewer import launch_viewer
