"""Interactive Napari viewer for rendering parameter optimisation.

Replaces the old Jupyter ipywidgets slider interface with a Napari-based
GUI using magicgui widgets. Allows real-time parameter tuning during
image rendering so users can visually match synthetic images to real data.

Usage:

    from symbac.imaging.napari_viewer import launch_viewer

    config = launch_viewer(
        opl_scenes=opl_scenes,
        mask_scenes=mask_scenes,
        device_masks=device_masks,
        psf_model=psf_pc,
        render_config=RenderConfig(),
        supersampling=3,
        real_image=real_img,  # optional
    )
    # config is the RenderConfig the user settled on before closing napari

Requires ``napari`` and ``magicgui`` (install with ``pip install napari[all]``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from symbac.imaging.optics import Camera, PSFModel
from symbac.imaging.renderer import RenderConfig, render_image


def launch_viewer(
    opl_scenes: Sequence[np.ndarray],
    mask_scenes: Sequence[np.ndarray],
    device_masks: Sequence[np.ndarray],
    psf_model: PSFModel,
    render_config: Optional[RenderConfig] = None,
    supersampling: int = 1,
    real_image: Optional[np.ndarray] = None,
    camera: Optional[Camera] = None,
    perlin_backgrounds: Optional[Sequence[np.ndarray]] = None,
) -> RenderConfig:
    """Launch a Napari viewer for interactive rendering parameter tuning.

    Opens a Napari window with the rendered synthetic image, segmentation
    masks, and (optionally) a real image for side-by-side comparison.  A
    docked widget with sliders for every ``RenderConfig`` parameter lets
    the user adjust the rendering in real-time.

    When the viewer window is closed the function returns the
    ``RenderConfig`` that was active at that time, so it can be fed
    straight into ``generate_training_data``.

    Args:
        opl_scenes: List of OPL scenes (at supersampled resolution if
            supersampling > 1).
        mask_scenes: Corresponding instance segmentation masks.
        device_masks: Corresponding binary device masks.
        psf_model: The PSF model to use for convolution.
        render_config: Starting ``RenderConfig``.  If *None*, uses defaults.
        supersampling: Supersampling factor matching the input resolution.
        real_image: Optional real microscopy image for visual comparison
            and histogram matching.
        camera: Optional ``Camera`` noise model.
        perlin_backgrounds: Optional list of Perlin noise backgrounds
            (one per scene, at native resolution).  If provided, enables
            a *Perlin background* toggle in the widget.

    Returns:
        The ``RenderConfig`` the user selected before closing the viewer.
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napari is required for the interactive viewer. "
            "Install with: pip install 'napari[all]'"
        )
    try:
        from magicgui import magicgui
    except ImportError:
        raise ImportError(
            "magicgui is required for the interactive viewer. "
            "Install with: pip install magicgui"
        )

    config = render_config or RenderConfig()
    n_scenes = len(opl_scenes)
    has_perlin = perlin_backgrounds is not None and len(perlin_backgrounds) > 0

    # Mutable container so the closure can update the "return value"
    final_config = [config]

    # ---- Initial render ------------------------------------------------
    rng = np.random.default_rng(0)
    init_image, init_masks = render_image(
        opl_scenes[0], mask_scenes[0], device_masks[0],
        psf_model, config,
        supersampling=supersampling,
        camera=camera,
        rng=rng,
    )

    # ---- Build viewer --------------------------------------------------
    viewer = napari.Viewer(title="SyMBac - Rendering Parameter Optimisation")

    if real_image is not None:
        viewer.add_image(
            real_image, name="Real Image", colormap="gray",
            blending="additive", visible=True,
        )

    synth_layer = viewer.add_image(
        init_image, name="Synthetic", colormap="gray",
    )
    mask_layer = viewer.add_labels(
        init_masks.astype(np.int32), name="Masks",
        visible=False,
    )

    # ---- Parameter widget ----------------------------------------------
    @magicgui(
        call_button="Render",
        scene_no={
            "widget_type": "Slider",
            "min": 0, "max": max(0, n_scenes - 1),
            "label": "Scene #",
        },
        media_multiplier={
            "widget_type": "FloatSlider",
            "min": -100.0, "max": 100.0, "step": 0.5,
            "label": "Media multiplier",
        },
        cell_multiplier={
            "widget_type": "FloatSlider",
            "min": -50.0, "max": 50.0, "step": 0.1,
            "label": "Cell multiplier",
        },
        device_multiplier={
            "widget_type": "FloatSlider",
            "min": -200.0, "max": 200.0, "step": 1.0,
            "label": "Device multiplier",
        },
        defocus={
            "widget_type": "FloatSlider",
            "min": 0.0, "max": 20.0, "step": 0.1,
            "label": "Defocus",
        },
        noise_var={
            "widget_type": "FloatSlider",
            "min": 0.0, "max": 0.05, "step": 0.0001,
            "label": "Noise variance",
        },
        border_expansion={
            "widget_type": "FloatSlider",
            "min": 1.0, "max": 4.0, "step": 0.1,
            "label": "Border expansion",
        },
        halo_top={
            "widget_type": "FloatSlider",
            "min": 0.0, "max": 2.0, "step": 0.01,
            "label": "Halo top",
        },
        halo_bottom={
            "widget_type": "FloatSlider",
            "min": 0.0, "max": 2.0, "step": 0.01,
            "label": "Halo bottom",
        },
        match_histogram={"label": "Match histogram"},
        use_perlin={"label": "Perlin background", "visible": has_perlin},
        perlin_attenuation={
            "widget_type": "FloatSlider",
            "min": 1.0, "max": 10.0, "step": 0.1,
            "label": "Perlin attenuation",
            "visible": has_perlin,
        },
        perlin_blur_sigma={
            "widget_type": "FloatSlider",
            "min": 0.0, "max": 10.0, "step": 0.1,
            "label": "Perlin blur sigma",
            "visible": has_perlin,
        },
    )
    def render_widget(
        scene_no: int = 0,
        media_multiplier: float = config.media_multiplier,
        cell_multiplier: float = config.cell_multiplier,
        device_multiplier: float = config.device_multiplier,
        defocus: float = config.defocus,
        noise_var: float = config.noise_var,
        border_expansion: float = config.border_expansion,
        halo_top: float = config.halo_top_intensity,
        halo_bottom: float = config.halo_bottom_intensity,
        match_histogram: bool = False,
        use_perlin: bool = False,
        perlin_attenuation: float = 3.0,
        perlin_blur_sigma: float = 2.0,
    ):
        new_config = RenderConfig(
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=device_multiplier,
            defocus=defocus,
            noise_var=noise_var,
            border_expansion=border_expansion,
            halo_top_intensity=halo_top,
            halo_bottom_intensity=halo_bottom,
        )

        perlin_bg = None
        p_att = None
        p_blur = None
        if use_perlin and has_perlin:
            idx = min(scene_no, len(perlin_backgrounds) - 1)
            perlin_bg = perlin_backgrounds[idx]
            p_att = perlin_attenuation
            p_blur = perlin_blur_sigma

        new_image, new_masks = render_image(
            opl_scenes[scene_no],
            mask_scenes[scene_no],
            device_masks[scene_no],
            psf_model,
            new_config,
            supersampling=supersampling,
            camera=camera,
            real_image=real_image if match_histogram else None,
            match_histogram=match_histogram and real_image is not None,
            perlin_background=perlin_bg,
            perlin_attenuation=p_att,
            perlin_blur_sigma=p_blur,
            rng=np.random.default_rng(0),
        )

        synth_layer.data = new_image
        mask_layer.data = new_masks.astype(np.int32)

        # Store the current config for return
        final_config[0] = new_config

    viewer.window.add_dock_widget(
        render_widget, name="Render Parameters", area="right",
    )

    napari.run()
    return final_config[0]
