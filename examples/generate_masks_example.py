"""
Example script demonstrating how to use the refactored MaskGenerator.

This script shows different ways to generate instance segmentation masks
from SyMBac simulation data using the new ground_truth module.
"""

from symbac.ground_truth import MaskGenerator, MaskConfig


# Create a mask generator with custom configuration
config = MaskConfig(
    resolution=(3000, 3000),  
    padding_px=50,
    n_jobs=-1  # Use 4 cores
)

mask_generator = MaskGenerator(config)

# Generate masks from a saved simulation file
try:
    mask_generator.generate_masks_from_file(
        input_file="frames.json",
        output_dir="instance_masks_raw",
        clear_output=True
    )
    
    # Get transform information
    transform_info = mask_generator.get_transform_info()
    if transform_info:
        print(f"Transform info: {transform_info}")
        
except FileNotFoundError:
    print("frames.json not found. Run sim_loop.py first to generate simulation data.")
except Exception as e:
    print(f"Error: {e}")
