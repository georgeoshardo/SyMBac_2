"""Cell rasterization and scene drawing for optical path length (OPL) images.

This module provides functions to convert the physics simulation state into
2D images where pixel intensity represents the optical path length (thickness)
of cells, along with corresponding segmentation masks.
"""

from __future__ import annotations

import typing

import numpy as np
from scipy.ndimage import binary_fill_holes, label, map_coordinates
from skimage.transform import rescale

if typing.TYPE_CHECKING:
    from symbac.simulation.simulator import Simulator
    from symbac.simulation.simcell import SimCell
    from symbac.imaging.optics import CellOpticsConfig


def compute_two_layer_opl(
    dist_sq: np.ndarray,
    r_sq: float,
    radius: float,
    cell_optics: "CellOpticsConfig",
) -> np.ndarray:
    """Compute OPL for a sphere cross-section using a two-layer cell model.

    Models the cell as a peptidoglycan wall (outer shell) surrounding
    cytoplasm (inner core). The OPL at each pixel accounts for the
    different refractive indices of each layer relative to the medium.

    Args:
        dist_sq: Array of squared distances from pixel centers to sphere center.
        r_sq: Squared outer radius of the sphere.
        radius: Outer radius of the sphere.
        cell_optics: Refractive index configuration.

    Returns:
        OPL array (same shape as dist_sq). Units are in the same length
        units as radius (typically supersampled pixels).
    """
    opl = np.zeros_like(dist_sq)
    inside = dist_sq < r_sq
    if not inside.any():
        return opl

    # Inner (cytoplasm) radius
    r_inner = radius * (1.0 - cell_optics.wall_fraction)
    r_inner_sq = r_inner * r_inner

    # Total thickness at each point (sphere cross-section)
    total_thickness = np.zeros_like(dist_sq)
    total_thickness[inside] = 2.0 * np.sqrt(r_sq - dist_sq[inside])

    # Inner core thickness (cytoplasm only, where inside inner sphere)
    inner_inside = dist_sq < r_inner_sq
    inner_thickness = np.zeros_like(dist_sq)
    inner_thickness[inner_inside] = 2.0 * np.sqrt(r_inner_sq - dist_sq[inner_inside])

    # Wall thickness = total - inner (can be zero at center if wall_fraction < 1)
    wall_thickness = total_thickness - inner_thickness

    # OPL = delta_n * thickness for each layer
    dn_wall = cell_optics.n_wall - cell_optics.n_medium
    dn_cyto = cell_optics.n_cytoplasm - cell_optics.n_medium

    opl = dn_wall * wall_thickness + dn_cyto * inner_thickness

    return opl


def raster_cell(length: int, width: int, separation: int = 0, pinching: bool = True) -> np.ndarray:
    """Produce a rasterised OPL image of a straight spherocylinder cell.

    Each pixel intensity corresponds to the optical path length (thickness) of
    the cell at that point, modelling the cell as a cylinder with hemispherical
    caps.

    Ported from old SyMBac drawing.raster_cell().

    Args:
        length: Cell length in pixels.
        width: Cell width in pixels.
        separation: Controls pinching during division (0 = no pinching).
        pinching: Whether to apply division pinching.

    Returns:
        2D numpy array with OPL values.
    """
    L = int(np.rint(length))
    W = int(np.rint(width))
    if L < 2 or W < 2:
        return np.zeros((max(L, 2), max(W, 2)))

    new_cell = np.zeros((L, W))
    R = (W - 1) / 2

    # Cylindrical body cross-section
    x_cyl = np.arange(0, 2 * R + 1, 1)
    I_cyl = np.sqrt(np.maximum(R ** 2 - (x_cyl - R) ** 2, 0))
    L_cyl = L - W
    half_W = int(W / 2)
    if half_W > 0 and L > W:
        new_cell[half_W:-half_W, :I_cyl.shape[0]] = I_cyl

    # Hemispherical caps
    x_sphere = np.arange(0, half_W, 1)
    if len(x_sphere) > 0:
        sphere_Rs = np.sqrt(np.maximum(R ** 2 - (x_sphere - R) ** 2, 0))
        sphere_Rs = np.rint(sphere_Rs).astype(int)

        for c in range(len(sphere_Rs)):
            R_ = sphere_Rs[c]
            if R_ <= 0:
                continue
            x_cyl_cap = np.arange(0, R_, 1)
            I_cyl_cap = np.sqrt(np.maximum(R_ ** 2 - (x_cyl_cap - R_) ** 2, 0))
            col_start = half_W - sphere_Rs[c]
            col_end = half_W + sphere_Rs[c]
            cap_profile = np.concatenate((I_cyl_cap, I_cyl_cap[::-1]))
            if col_end - col_start == len(cap_profile):
                new_cell[c, col_start:col_end] = cap_profile
                new_cell[L - c - 1, col_start:col_end] = cap_profile

    # Division pinching
    if separation > 2 and pinching:
        S = int(np.rint(separation))
        half_LS = int((L - S) / 2)
        if half_LS > 0:
            new_cell[half_LS + 1:-half_LS - 1, :] = 0
            for c in range(int((S + 1) / 2)):
                if c < len(x_sphere) and len(x_sphere) > 0:
                    idx = min(-c - 1, -1)
                    if abs(idx) <= len(sphere_Rs):
                        R__ = sphere_Rs[idx]
                        if R__ > 0:
                            x_cyl_ = np.arange(0, R__, 1)
                            I_cyl_ = np.sqrt(np.maximum(R__ ** 2 - (x_cyl_ - R__) ** 2, 0))
                            col_s = half_W - R__
                            col_e = half_W + R__
                            pinch_profile = np.concatenate((I_cyl_, I_cyl_[::-1]))
                            if col_e - col_s == len(pinch_profile):
                                new_cell[half_LS + c + 1, col_s:col_e] = pinch_profile
                                new_cell[-half_LS - c - 1, col_s:col_e] = pinch_profile

    return new_cell.astype(float)


def rasterize_segment_chain(
    cell: SimCell,
    pixel_scale: float,
    supersampling: int = 3,
    canvas_origin: tuple[float, float] | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a segment-chain cell into OPL and mask images.

    For each segment (sphere) in the cell's physics representation, computes
    the sphere cross-section OPL at each pixel. The result captures the actual
    physical curvature of the cell.

    Args:
        cell: The SimCell to rasterize.
        pixel_scale: Microns per pixel at native resolution.
        supersampling: Render at this multiple of native resolution, then downsample.
        canvas_origin: (x, y) in physics coords of the top-left corner. If None, auto-computed.
        canvas_size: (height, width) of the output canvas in native pixels. If None, auto-computed.

    Returns:
        Tuple of (opl_image, mask_image) at native resolution.
        opl_image: float array with optical path length values.
        mask_image: int array with cell group_id where cell exists, 0 elsewhere.
    """
    segments = cell.physics_representation.segments
    if not segments:
        if canvas_size is not None:
            return np.zeros(canvas_size), np.zeros(canvas_size, dtype=np.int32)
        return np.zeros((1, 1)), np.zeros((1, 1), dtype=np.int32)

    # Effective pixel scale at supersampled resolution
    ss_scale = pixel_scale / supersampling

    # Gather segment positions and radii
    positions = np.array([(seg.position.x, seg.position.y) for seg in segments])
    radii = np.array([seg.radius for seg in segments])

    if canvas_origin is None or canvas_size is None:
        # Auto-compute bounding box
        max_r = radii.max()
        padding = max_r + 2 * pixel_scale  # extra padding
        x_min = positions[:, 0].min() - padding
        y_min = positions[:, 1].min() - padding
        x_max = positions[:, 0].max() + padding
        y_max = positions[:, 1].max() + padding

        if canvas_origin is None:
            canvas_origin = (x_min, y_min)
        if canvas_size is None:
            w_native = int(np.ceil((x_max - x_min) / pixel_scale))
            h_native = int(np.ceil((y_max - y_min) / pixel_scale))
            canvas_size = (h_native, w_native)

    ox, oy = canvas_origin
    h_native, w_native = canvas_size
    h_ss = h_native * supersampling
    w_ss = w_native * supersampling

    opl_ss = np.zeros((h_ss, w_ss), dtype=np.float64)
    mask_ss = np.zeros((h_ss, w_ss), dtype=np.int32)

    # For each segment, compute OPL contribution
    for seg_pos, seg_r in zip(positions, radii):
        # Segment center in supersampled pixel coords
        cx_ss = (seg_pos[0] - ox) / ss_scale
        cy_ss = (seg_pos[1] - oy) / ss_scale
        r_ss = seg_r / ss_scale

        # Bounding box in supersampled pixels
        ix_min = max(0, int(np.floor(cx_ss - r_ss)))
        ix_max = min(w_ss, int(np.ceil(cx_ss + r_ss)) + 1)
        iy_min = max(0, int(np.floor(cy_ss - r_ss)))
        iy_max = min(h_ss, int(np.ceil(cy_ss + r_ss)) + 1)

        if ix_min >= ix_max or iy_min >= iy_max:
            continue

        # Create local coordinate grid
        yy, xx = np.mgrid[iy_min:iy_max, ix_min:ix_max]
        dx = xx.astype(np.float64) + 0.5 - cx_ss
        dy = yy.astype(np.float64) + 0.5 - cy_ss
        dist_sq = dx * dx + dy * dy
        r_sq = r_ss * r_ss

        inside = dist_sq < r_sq
        if not inside.any():
            continue

        # OPL = 2 * sqrt(R^2 - d^2) for a sphere cross-section
        thickness = np.zeros_like(dist_sq)
        thickness[inside] = 2.0 * np.sqrt(r_sq - dist_sq[inside])

        opl_ss[iy_min:iy_max, ix_min:ix_max] += thickness
        mask_ss[iy_min:iy_max, ix_min:ix_max][inside] = cell.group_id

    # Downsample to native resolution
    if supersampling > 1:
        opl_native = rescale(opl_ss, 1.0 / supersampling, anti_aliasing=True, preserve_range=True)
        # For mask, use nearest-neighbor (no interpolation)
        mask_native = rescale(mask_ss.astype(float), 1.0 / supersampling,
                              anti_aliasing=False, order=0, preserve_range=True).astype(np.int32)
    else:
        opl_native = opl_ss
        mask_native = mask_ss

    # Ensure output matches requested canvas_size
    opl_native = opl_native[:canvas_size[0], :canvas_size[1]]
    mask_native = mask_native[:canvas_size[0], :canvas_size[1]]

    return opl_native, mask_native


def _get_scene_bounds(
    simulator: Simulator,
    pixel_scale: float,
    padding_physics: float = 0.0,
) -> tuple[tuple[float, float], tuple[int, int]]:
    """Compute the bounding box for the scene from all cells and static geometry.

    Returns:
        (origin, size) where origin is (x_min, y_min) in physics coords
        and size is (height, width) in native pixels.
    """
    all_positions = []
    all_radii = []

    for cell in simulator.colony.cells:
        for seg in cell.physics_representation.segments:
            all_positions.append((seg.position.x, seg.position.y))
            all_radii.append(seg.radius)

    # Also include static geometry (trench walls, boxes)
    for shape in simulator.space.shapes:
        if shape.body.body_type == 2:  # pymunk.Body.STATIC
            bb = shape.bb
            all_positions.append((bb.left, bb.bottom))
            all_positions.append((bb.right, bb.top))
            all_radii.append(0)
            all_radii.append(0)

    if not all_positions:
        return (0.0, 0.0), (100, 100)

    positions = np.array(all_positions)
    radii = np.array(all_radii)

    max_r = radii.max() if len(radii) > 0 else 0
    padding = max_r + padding_physics + 5 * pixel_scale

    x_min = positions[:, 0].min() - padding
    y_min = positions[:, 1].min() - padding
    x_max = positions[:, 0].max() + padding
    y_max = positions[:, 1].max() + padding

    w = int(np.ceil((x_max - x_min) / pixel_scale))
    h = int(np.ceil((y_max - y_min) / pixel_scale))

    return (x_min, y_min), (h, w)


def draw_scene(
    simulator: Simulator,
    pixel_scale: float,
    supersampling: int = 3,
    label_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the full OPL scene and masks from current simulator state.

    Iterates over all cells, rasterizes each one, and composites them into
    a scene-sized canvas.

    Args:
        simulator: The Simulator instance with current cell states.
        pixel_scale: Microns per pixel at native camera resolution.
        supersampling: Internal rendering upscale factor for anti-aliasing.
        label_masks: If True, mask pixels contain cell group_id (instance labels).
            If False, mask is binary.

    Returns:
        Tuple of (scene_opl, scene_masks).
    """
    origin, (h, w) = _get_scene_bounds(simulator, pixel_scale)
    canvas_size = (h, w)

    scene_opl = np.zeros(canvas_size, dtype=np.float64)
    scene_masks = np.zeros(canvas_size, dtype=np.int32)
    overlap_count = np.zeros(canvas_size, dtype=np.int32)

    for cell in simulator.colony.cells:
        cell_opl, cell_mask = rasterize_segment_chain(
            cell,
            pixel_scale=pixel_scale,
            supersampling=supersampling,
            canvas_origin=origin,
            canvas_size=canvas_size,
        )
        # Add OPL
        scene_opl += cell_opl

        # Handle mask overlaps: where cells overlap, set mask to 0 (boundary)
        cell_present = cell_mask > 0
        overlap_count += cell_present.astype(np.int32)
        scene_masks[cell_present] = cell_mask[cell_present]

    # Zero out overlapping regions in the mask
    overlap_regions = overlap_count > 1
    scene_masks[overlap_regions] = 0

    if not label_masks:
        scene_masks = (scene_masks > 0).astype(np.int32)

    return scene_opl, scene_masks


def draw_scene_with_geometry(
    simulator: Simulator,
    pixel_scale: float,
    supersampling: int = 3,
    label_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw OPL scene, masks, and a device geometry mask.

    Like draw_scene but additionally produces a binary mask of the
    microfluidic device (static bodies in the physics space).

    Args:
        simulator: The Simulator instance.
        pixel_scale: Microns per pixel at native resolution.
        supersampling: Internal upscale factor.
        label_masks: Instance vs binary masks.

    Returns:
        Tuple of (scene_opl, scene_masks, device_mask).
        device_mask: Binary mask where 1 = device wall, 0 = not device.
    """
    origin, (h, w) = _get_scene_bounds(simulator, pixel_scale)
    canvas_size = (h, w)

    scene_opl, scene_masks = draw_scene(
        simulator, pixel_scale, supersampling, label_masks
    )

    # Build device mask from static shapes
    ss_scale = pixel_scale / supersampling
    h_ss = h * supersampling
    w_ss = w * supersampling
    device_ss = np.zeros((h_ss, w_ss), dtype=np.uint8)
    ox, oy = origin

    for shape in simulator.space.shapes:
        if shape.body.body_type != 2:  # Only STATIC bodies (pymunk.Body.STATIC = 2)
            continue

        if isinstance(shape, __import__('pymunk').Segment):
            # Segment shapes (line segments with thickness)
            a = shape.body.local_to_world(shape.a)
            b = shape.body.local_to_world(shape.b)
            thickness = shape.radius

            # Convert to supersampled pixel coords
            ax_ss = (a.x - ox) / ss_scale
            ay_ss = (a.y - oy) / ss_scale
            bx_ss = (b.x - ox) / ss_scale
            by_ss = (b.y - oy) / ss_scale
            t_ss = thickness / ss_scale

            # Rasterize the segment as a thick line
            _rasterize_thick_line(device_ss, ax_ss, ay_ss, bx_ss, by_ss, t_ss)

        elif isinstance(shape, __import__('pymunk').Circle):
            cx = shape.body.position.x + shape.offset.x
            cy = shape.body.position.y + shape.offset.y
            r = shape.radius
            cx_ss = (cx - ox) / ss_scale
            cy_ss = (cy - oy) / ss_scale
            r_ss = r / ss_scale
            yy, xx = np.ogrid[
                max(0, int(cy_ss - r_ss)):min(h_ss, int(cy_ss + r_ss) + 1),
                max(0, int(cx_ss - r_ss)):min(w_ss, int(cx_ss + r_ss) + 1)
            ]
            dist_sq = (xx - cx_ss) ** 2 + (yy - cy_ss) ** 2
            device_ss[
                max(0, int(cy_ss - r_ss)):min(h_ss, int(cy_ss + r_ss) + 1),
                max(0, int(cx_ss - r_ss)):min(w_ss, int(cx_ss + r_ss) + 1)
            ][dist_sq <= r_ss ** 2] = 1

    # Fill the exterior: everything outside the trench = PDMS (device)
    # Use the cell OPL to find seeds for the interior (where cells are)
    # Rasterize cells at supersampled resolution for seeding
    cell_mask_ss = np.zeros((h_ss, w_ss), dtype=np.int32)
    for cell in simulator.colony.cells:
        for seg in cell.physics_representation.segments:
            cx = (seg.position.x - ox) / ss_scale
            cy = (seg.position.y - oy) / ss_scale
            r = seg.radius / ss_scale
            y0 = max(0, int(cy - r))
            y1 = min(h_ss, int(cy + r) + 1)
            x0 = max(0, int(cx - r))
            x1 = min(w_ss, int(cx + r) + 1)
            if y0 < y1 and x0 < x1:
                yy_c, xx_c = np.ogrid[y0:y1, x0:x1]
                dist = (xx_c - cx) ** 2 + (yy_c - cy) ** 2
                cell_mask_ss[y0:y1, x0:x1][dist <= r ** 2] = 1

    device_filled_ss = fill_device_exterior(device_ss, cell_mask_ss)

    # Downsample device mask
    if supersampling > 1:
        device_mask = rescale(device_filled_ss.astype(float), 1.0 / supersampling,
                              anti_aliasing=False, order=0, preserve_range=True).astype(np.uint8)
    else:
        device_mask = device_filled_ss

    device_mask = device_mask[:h, :w]

    return scene_opl, scene_masks, device_mask


def draw_scene_supersampled(
    simulator: Simulator,
    pixel_scale: float,
    supersampling: int = 3,
    label_masks: bool = True,
    cell_optics: "CellOpticsConfig | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Draw OPL, masks, and device mask all at supersampled resolution.

    This is the correct entry point for the rendering pipeline as described
    in the SyMBac paper: render at high resolution, convolve with PSF at
    high resolution, then downscale the result to native resolution.

    Args:
        simulator: The Simulator instance.
        pixel_scale: Microns per pixel at native camera resolution.
        supersampling: Upscale factor (e.g. 3 means render at 3x resolution).
        label_masks: Instance vs binary masks.
        cell_optics: Optional two-layer cell optics configuration. When provided,
            the OPL is computed using physically accurate refractive indices
            for the wall and cytoplasm layers. When None, uses the simple
            sphere cross-section model (2 * sqrt(R^2 - d^2)).

    Returns:
        Tuple of (opl_ss, masks_ss, device_ss, native_size).
        opl_ss: OPL scene at supersampled resolution (float).
        masks_ss: Instance masks at supersampled resolution (int32).
        device_ss: Device mask at supersampled resolution (uint8).
        native_size: (height, width) at native resolution for downscaling.
    """
    import pymunk as pm

    origin, (h_native, w_native) = _get_scene_bounds(simulator, pixel_scale)
    ss_scale = pixel_scale / supersampling
    h_ss = h_native * supersampling
    w_ss = w_native * supersampling

    # Rasterize cells at supersampled resolution
    opl_ss = np.zeros((h_ss, w_ss), dtype=np.float64)
    mask_ss = np.zeros((h_ss, w_ss), dtype=np.int32)
    overlap_count = np.zeros((h_ss, w_ss), dtype=np.int32)
    ox, oy = origin

    for cell in simulator.colony.cells:
        segments = cell.physics_representation.segments
        if not segments:
            continue
        positions = np.array([(seg.position.x, seg.position.y) for seg in segments])
        radii = np.array([seg.radius for seg in segments])

        for seg_pos, seg_r in zip(positions, radii):
            cx = (seg_pos[0] - ox) / ss_scale
            cy = (seg_pos[1] - oy) / ss_scale
            r = seg_r / ss_scale

            ix_min = max(0, int(np.floor(cx - r)))
            ix_max = min(w_ss, int(np.ceil(cx + r)) + 1)
            iy_min = max(0, int(np.floor(cy - r)))
            iy_max = min(h_ss, int(np.ceil(cy + r)) + 1)
            if ix_min >= ix_max or iy_min >= iy_max:
                continue

            yy, xx = np.mgrid[iy_min:iy_max, ix_min:ix_max]
            dx = xx.astype(np.float64) + 0.5 - cx
            dy = yy.astype(np.float64) + 0.5 - cy
            dist_sq = dx * dx + dy * dy
            r_sq = r * r
            inside = dist_sq < r_sq
            if not inside.any():
                continue

            if cell_optics is not None:
                contribution = compute_two_layer_opl(dist_sq, r_sq, r, cell_optics)
            else:
                contribution = np.zeros_like(dist_sq)
                contribution[inside] = 2.0 * np.sqrt(r_sq - dist_sq[inside])
            opl_ss[iy_min:iy_max, ix_min:ix_max] += contribution
            mask_ss[iy_min:iy_max, ix_min:ix_max][inside] = cell.group_id
            overlap_count[iy_min:iy_max, ix_min:ix_max] += inside.astype(np.int32)

    # Zero out overlapping mask regions
    mask_ss[overlap_count > 1] = 0
    if not label_masks:
        mask_ss = (mask_ss > 0).astype(np.int32)

    # Rasterize device geometry at supersampled resolution
    device_ss = np.zeros((h_ss, w_ss), dtype=np.uint8)
    for shape in simulator.space.shapes:
        if shape.body.body_type != 2:
            continue
        if isinstance(shape, pm.Segment):
            a = shape.body.local_to_world(shape.a)
            b = shape.body.local_to_world(shape.b)
            thickness = shape.radius
            ax_ss = (a.x - ox) / ss_scale
            ay_ss = (a.y - oy) / ss_scale
            bx_ss = (b.x - ox) / ss_scale
            by_ss = (b.y - oy) / ss_scale
            t_ss = thickness / ss_scale
            _rasterize_thick_line(device_ss, ax_ss, ay_ss, bx_ss, by_ss, t_ss)
        elif isinstance(shape, pm.Circle):
            cx = shape.body.position.x + shape.offset.x
            cy = shape.body.position.y + shape.offset.y
            r = shape.radius
            cx_ss = (cx - ox) / ss_scale
            cy_ss = (cy - oy) / ss_scale
            r_ss = r / ss_scale
            y0 = max(0, int(cy_ss - r_ss))
            y1 = min(h_ss, int(cy_ss + r_ss) + 1)
            x0 = max(0, int(cx_ss - r_ss))
            x1 = min(w_ss, int(cx_ss + r_ss) + 1)
            if y0 < y1 and x0 < x1:
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist_sq = (xx - cx_ss) ** 2 + (yy - cy_ss) ** 2
                device_ss[y0:y1, x0:x1][dist_sq <= r_ss ** 2] = 1

    # Fill the exterior: everything outside the trench = PDMS (device)
    device_ss = fill_device_exterior(device_ss, mask_ss)

    return opl_ss, mask_ss, device_ss, (h_native, w_native)


def fill_device_exterior(
    device_wall_mask: np.ndarray,
    cell_mask: np.ndarray,
) -> np.ndarray:
    """Fill device exterior so everything outside the trench is marked as device.

    In a mother machine, the device walls define the boundary between the thin
    media channel (trench interior) and thick PDMS (everything outside). The
    walls may form an open shape (e.g. a U-shape trench open at the top).

    The approach (matching old SyMBac):
    1. Scanline: for each row with walls, find gaps between wall groups = interior
    2. Identify the stable "channel columns" from the scanline
    3. Extend the interior along channel columns through open ends, stopping at walls

    This correctly handles U-shaped trenches (open at one end), closed boxes,
    and arbitrary device geometries.

    Args:
        device_wall_mask: Binary mask where 1 = device wall pixels.
        cell_mask: Mask where > 0 = cell pixels (used as fallback).

    Returns:
        Filled device mask where 1 = device (walls + PDMS exterior), 0 = media interior.
    """
    if not device_wall_mask.any():
        return device_wall_mask.copy()

    h, w = device_wall_mask.shape
    walls = device_wall_mask.astype(bool)
    interior = np.zeros((h, w), dtype=bool)

    # Step 1: Scanline - find interior gaps between wall groups in each row
    for y in range(h):
        row = walls[y, :]
        if not row.any():
            continue
        wall_pixels = np.where(row)[0]
        diffs = np.diff(wall_pixels)
        gap_indices = np.where(diffs > 1)[0]

        for gi in gap_indices:
            gap_left = wall_pixels[gi] + 1
            gap_right = wall_pixels[gi + 1]
            interior[y, gap_left:gap_right] = True

    # Step 2: Identify channel columns (interior in at least some wall rows)
    wall_rows = np.any(walls, axis=1)
    n_wall_rows = wall_rows.sum()
    if n_wall_rows == 0 or not interior.any():
        # Walls exist but no interior found - fall back to all-device
        return np.ones_like(device_wall_mask, dtype=np.uint8)

    # Channel columns = columns that are interior in >30% of wall rows
    channel_cols = np.where(
        np.sum(interior[wall_rows, :], axis=0) > n_wall_rows * 0.3
    )[0]

    if len(channel_cols) == 0:
        return np.ones_like(device_wall_mask, dtype=np.uint8)

    # Step 3: Extend interior along channel columns through open ends
    # For each channel column, extend upward and downward from the existing
    # interior region, stopping when hitting a wall pixel.
    for x in channel_cols:
        col_interior = np.where(interior[:, x])[0]
        if len(col_interior) == 0:
            continue
        col_walls = walls[:, x]
        y_min = col_interior.min()
        y_max = col_interior.max()

        # Extend upward
        for y in range(y_min - 1, -1, -1):
            if col_walls[y]:
                break
            interior[y, x] = True

        # Extend downward
        for y in range(y_max + 1, h):
            if col_walls[y]:
                break
            interior[y, x] = True

    # Filled device = everything that's not interior
    filled = np.ones_like(device_wall_mask, dtype=np.uint8)
    filled[interior] = 0

    return filled


def _rasterize_thick_line(
    canvas: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    thickness: float,
) -> None:
    """Rasterize a thick line segment onto a 2D canvas."""
    h, w = canvas.shape

    # Bounding box
    half_t = thickness / 2 + 1
    min_x = max(0, int(min(x0, x1) - half_t))
    max_x = min(w, int(max(x0, x1) + half_t) + 1)
    min_y = max(0, int(min(y0, y1) - half_t))
    max_y = min(h, int(max(y0, y1) + half_t) + 1)

    if min_x >= max_x or min_y >= max_y:
        return

    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]
    xx = xx.astype(np.float64) + 0.5
    yy = yy.astype(np.float64) + 0.5

    # Distance from point to line segment
    dx = x1 - x0
    dy = y1 - y0
    len_sq = dx * dx + dy * dy

    if len_sq < 1e-10:
        # Degenerate: point
        dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2
    else:
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / len_sq, 0, 1)
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2

    inside = dist_sq <= (thickness / 2) ** 2
    canvas[min_y:max_y, min_x:max_x][inside] = 1
