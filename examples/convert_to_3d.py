import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import tifffile

# --- CONFIGURATION ---
# Directory where the OPL images and raw data were saved
INPUT_DIR = "opl_images"
# Directory to save the final 3D TIFF stacks
OUTPUT_DIR = "3d_volumes"


def process_opl_to_3d(opl_file_path: str, output_dir: str):
    """
    Converts a single 2D OPL data file into a 3D binary volume.

    Args:
        opl_file_path: The full path to the input .npy file.
        output_dir: The directory to save the output .tiff file.
    """
    # Load the raw OPL data
    opl_data = np.load(opl_file_path)

    # Determine the dimensions of the 3D volume
    height, width = opl_data.shape
    # The depth of the volume is determined by the maximum OPL (thickness)
    max_thickness = int(np.ceil(opl_data.max()))

    if max_thickness == 0:
        # If the frame is empty, we can skip or save an empty volume
        return

    # --- Vectorized 3D Volume Creation ---
    # Create a Z-coordinate array that broadcasts against the 2D OPL image.
    # Shape: (max_thickness, 1, 1)
    z_coords = np.arange(max_thickness).reshape(-1, 1, 1)

    # Center the cell mass around the Z-axis midpoint
    center_z = max_thickness / 2.0

    # The condition for a voxel being "inside" the cell is if its z-coordinate
    # is within half the cell's thickness from the center.
    # This comparison is broadcast across the entire volume at once.
    # `opl_data` is broadcast from (H, W) to (Z, H, W)
    # `z_coords` is broadcast from (Z, 1, 1) to (Z, H, W)
    volume_mask = np.abs(z_coords - center_z) <= (opl_data / 2.0)

    # Convert the boolean mask to an 8-bit binary volume (0 or 255)
    binary_volume = volume_mask.astype(np.uint8) * 255

    # --- Save as a TIFF Stack ---
    base_name = os.path.basename(opl_file_path).replace('.npy', '').replace('opl_data_', '')
    output_path = os.path.join(output_dir, f"volume_{base_name}.tiff")
    tifffile.imwrite(output_path, binary_volume)


def main():
    """Main function to find OPL data and process it in parallel."""
    print("Starting 3D volume conversion process...")

    # Define the directory containing the raw .npy files
    raw_opl_dir = os.path.join(INPUT_DIR, "raw_opl_data")

    # --- 1. Setup ---
    if not os.path.isdir(raw_opl_dir):
        print(f"Error: Input directory '{raw_opl_dir}' not found.")
        print("Please run the modified render_opl.py script first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Clearing output directory: ./{OUTPUT_DIR}/")
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".tiff"):
            os.remove(os.path.join(OUTPUT_DIR, filename))

    # --- 2. Find all .npy files to process ---
    opl_files = [
        os.path.join(raw_opl_dir, f)
        for f in os.listdir(raw_opl_dir) if f.endswith('.npy')
    ]

    if not opl_files:
        print("No raw OPL (.npy) files found to process.")
        return

    # --- 3. Parallel Processing ---
    print(f"Converting {len(opl_files)} OPL files to 3D TIFFs in parallel...")
    start_time = time.perf_counter()

    Parallel(n_jobs=-1)(
        delayed(process_opl_to_3d)(file_path, OUTPUT_DIR)
        for file_path in tqdm(opl_files, desc="Converting to 3D")
    )

    end_time = time.perf_counter()
    print("\nConversion complete.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Output 3D volumes are saved in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()