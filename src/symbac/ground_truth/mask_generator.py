"""
Ground truth mask generation for SyMBac simulations.

This module provides functionality to generate instance segmentation masks
from SyMBac simulation data, converting the physics-based cell representations
into pixel-perfect masks suitable for training or evaluation of computer vision models.
"""

import numpy as np
import os
import pickle
import json
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from PIL import Image

try:
    from raster_geometry import circle
except ImportError:
    raise ImportError(
        "raster_geometry is required for mask generation. "
        "Install it with: pip install raster-geometry"
    )


@dataclass
class MaskConfig:
    """Configuration for mask generation."""
    resolution: Tuple[int, int] = (5000, 5000)
    padding_px: int = 100
    n_jobs: int = -1  # Use all available cores


@dataclass
class CoordinateTransform:
    """Coordinate transformation parameters for world to pixel conversion."""
    min_x: float
    min_y: float
    scale: float
    offset_x: float
    offset_y: float


class MaskGenerator:
    """
    Generates instance segmentation masks from SyMBac simulation data.
    
    This class takes simulation data (either from a live simulation or saved files)
    and converts the physics-based cell representations into pixel-perfect masks
    where each cell is assigned a unique ID value.
    
    Attributes:
        config: Configuration for mask generation
        transform: Coordinate transformation parameters
    """
    
    def __init__(self, config: Optional[MaskConfig] = None):
        """
        Initialize the mask generator.
        
        Args:
            config: Configuration for mask generation. If None, uses default values.
        """
        self.config = config or MaskConfig()
        self.transform: Optional[CoordinateTransform] = None
    
    def load_simulation_data(self, filepath: str) -> List[Tuple[int, List[Dict]]]:
        """
        Load simulation data from either a pickle or JSON file.
        
        Expects the format from SimulationLogger (dictionary with frame_number as keys).
        
        Args:
            filepath: Path to the input file
            
        Returns:
            List of tuples (frame_number, frame_data) in the expected format
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file '{filepath}' not found.")
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                frames_dict = json.load(f)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                frames_dict = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        # Convert dictionary format to list of tuples format
        all_frames = []
        for frame_number, frame_data in frames_dict.items():
            # Convert frame_data from dict of cell_id -> segments to flat list of segments
            segments = []
            for cell_id, cell_segments in frame_data.items():
                for segment in cell_segments:
                    # Add cell_id to segment data
                    segment_with_id = segment.copy()
                    segment_with_id['id'] = int(cell_id)
                    segments.append(segment_with_id)
            all_frames.append((int(frame_number), segments))
        
        return all_frames
    
    def get_simulation_bounds(self, all_frames_data: List[Tuple[int, List[Dict]]]) -> Tuple[float, float, float, float]:
        """
        Calculate the global bounding box for all cell segments across all frames.
        
        Args:
            all_frames_data: A list containing the data for every frame.
            
        Returns:
            A tuple (min_x, max_x, min_y, max_y) representing the bounds.
        """
        all_x = []
        all_y = []
        
        for _, frame_data in all_frames_data:
            if not frame_data:
                continue
            for seg in frame_data:
                x, y = seg['position']
                r = seg['radius']
                all_x.extend([x - r, x + r])
                all_y.extend([y - r, y + r])
        
        if not all_x or not all_y:
            return -100, 100, -100, 100
        
        return min(all_x), max(all_x), min(all_y), max(all_y)
    
    def _calculate_transform(self, all_frames_data: List[Tuple[int, List[Dict]]]) -> CoordinateTransform:
        """
        Calculate coordinate transformation parameters.
        
        Args:
            all_frames_data: Simulation data to calculate bounds from
            
        Returns:
            CoordinateTransform object with transformation parameters
        """
        min_x, max_x, min_y, max_y = self.get_simulation_bounds(all_frames_data)
        
        world_width = max_x - min_x
        world_height = max_y - min_y
        canvas_height, canvas_width = self.config.resolution
        
        scale = min(
            (canvas_width - 2 * self.config.padding_px) / world_width,
            (canvas_height - 2 * self.config.padding_px) / world_height
        )
        offset_x = (canvas_width - world_width * scale) / 2
        offset_y = (canvas_height - world_height * scale) / 2
        
        return CoordinateTransform(
            min_x=min_x, min_y=min_y, scale=scale,
            offset_x=offset_x, offset_y=offset_y
        )
    
    def _render_single_frame(
        self,
        frame_content: Tuple[int, List[Dict]],
        output_dir: str,
        transform: CoordinateTransform
    ) -> None:
        """
        Render a single frame using localized drawing for performance.
        
        Args:
            frame_content: Tuple of (frame_number, frame_data)
            output_dir: Directory to save the output mask
            transform: Coordinate transformation parameters
        """
        frame_number, frame_data = frame_content
        canvas_height, canvas_width = self.config.resolution
        id_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
        
        if not frame_data:
            img = Image.fromarray(id_canvas, mode='I;16')
            output_path = os.path.join(output_dir, f"mask_{frame_number:05d}.png")
            img.save(output_path)
            return
        
        sorted_segments = sorted(frame_data, key=lambda s: s['id'])
        
        for segment in sorted_segments:
            world_x, world_y = segment['position']
            world_r = segment['radius']
            cell_id = segment['id']
            
            # Coordinate transformation
            pixel_x = (world_x - transform.min_x) * transform.scale + transform.offset_x
            pixel_y = (world_y - transform.min_y) * transform.scale + transform.offset_y
            pixel_r = world_r * transform.scale
            
            # If radius is tiny, it might not even cover a pixel
            if pixel_r < 0.5:
                continue
            
            # Localized rendering optimization
            # 1. Define the local bounding box for the circle
            x_min = int(np.floor(pixel_x - pixel_r))
            x_max = int(np.ceil(pixel_x + pixel_r))
            y_min = int(np.floor(pixel_y - pixel_r))
            y_max = int(np.ceil(pixel_y + pixel_r))
            
            # 2. Clip the bounding box to the canvas dimensions
            clipped_x_min = max(0, x_min)
            clipped_y_min = max(0, y_min)
            clipped_x_max = min(canvas_width, x_max)
            clipped_y_max = min(canvas_height, y_max)
            
            # If the box is off-canvas, skip to the next segment
            if clipped_x_max <= clipped_x_min or clipped_y_max <= clipped_y_min:
                continue
            
            # 3. Define the shape and relative position for the local rasterization
            local_shape = (clipped_y_max - clipped_y_min, clipped_x_max - clipped_x_min)
            local_center_x = pixel_x - clipped_x_min
            local_center_y = pixel_y - clipped_y_min
            
            # Avoid division by zero if the local shape has a dimension of 0
            if local_shape[0] == 0 or local_shape[1] == 0:
                continue
            
            local_relative_pos = (local_center_y / local_shape[0], local_center_x / local_shape[1])
            
            # 4. Call raster.circle on the small, local canvas
            local_mask = circle(shape=local_shape, radius=pixel_r, position=local_relative_pos)
            
            # 5. "Stamp" the local mask onto the main canvas
            canvas_slice = id_canvas[clipped_y_min:clipped_y_max, clipped_x_min:clipped_x_max]
            canvas_slice[local_mask] = cell_id
        
        # Save the raw ID canvas
        output_path = os.path.join(output_dir, f"mask_{frame_number:05d}.png")
        img = Image.fromarray(id_canvas, mode='I;16')
        img.save(output_path)
    
    def generate_masks_from_file(
        self, 
        input_file: str, 
        output_dir: str,
        clear_output: bool = True
    ) -> None:
        """
        Generate masks from a saved simulation file.
        
        Args:
            input_file: Path to the input simulation file (.json or .pkl)
            output_dir: Directory to save the output masks
            clear_output: Whether to clear existing PNG files in output directory
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file format is not supported
        """
        print(f"Starting mask generation from file: {input_file}")
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        if clear_output:
            print(f"Clearing output directory: {output_dir}")
            for filename in os.listdir(output_dir):
                if filename.endswith(".png"):
                    os.remove(os.path.join(output_dir, filename))
        
        # Load data
        print(f"Loading data from {input_file}...")
        all_frames = self.load_simulation_data(input_file)
        
        if not all_frames:
            print("No frames to render.")
            return
        
        # Calculate transform
        print("Calculating simulation bounds...")
        self.transform = self._calculate_transform(all_frames)
        
        # Parallel rendering
        print(f"Rendering {len(all_frames)} frames in parallel...")
        start_time = time.perf_counter()
        
        Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._render_single_frame)(frame, output_dir, self.transform)
            for frame in tqdm(all_frames, desc="Rendering Frames")
        )
        
        end_time = time.perf_counter()
        print(f"\nMask generation complete.")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        print(f"Output masks saved in: {output_dir}")
    
    def generate_masks_from_simulation_data(
        self, 
        simulation_data: List[Tuple[int, List[Dict]]], 
        output_dir: str,
        clear_output: bool = True
    ) -> None:
        """
        Generate masks directly from simulation data (e.g., from a live simulation).
        
        Args:
            simulation_data: List of (frame_number, frame_data) tuples
            output_dir: Directory to save the output masks
            clear_output: Whether to clear existing PNG files in output directory
        """
        print(f"Starting mask generation from simulation data...")
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        if clear_output:
            print(f"Clearing output directory: {output_dir}")
            for filename in os.listdir(output_dir):
                if filename.endswith(".png"):
                    os.remove(os.path.join(output_dir, filename))
        
        if not simulation_data:
            print("No frames to render.")
            return
        
        # Calculate transform
        print("Calculating simulation bounds...")
        self.transform = self._calculate_transform(simulation_data)
        
        # Parallel rendering
        print(f"Rendering {len(simulation_data)} frames in parallel...")
        start_time = time.perf_counter()
        
        Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._render_single_frame)(frame, output_dir, self.transform)
            for frame in tqdm(simulation_data, desc="Rendering Frames")
        )
        
        end_time = time.perf_counter()
        print(f"\nMask generation complete.")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        print(f"Output masks saved in: {output_dir}")
    
    def get_transform_info(self) -> Optional[Dict]:
        """
        Get information about the current coordinate transformation.
        
        Returns:
            Dictionary with transformation parameters, or None if not calculated
        """
        if self.transform is None:
            return None
        
        return {
            'min_x': self.transform.min_x,
            'min_y': self.transform.min_y,
            'scale': self.transform.scale,
            'offset_x': self.transform.offset_x,
            'offset_y': self.transform.offset_y
        }
