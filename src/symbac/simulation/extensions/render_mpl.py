
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

def render_frame(frame_data: list, frame_number: int, output_dir: str):
    """
    Draws a single frame from pre-collected data using Matplotlib and saves it.
    This function is designed to be called in parallel.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))

    if not frame_data:
        ax.set_xlim(-100, 100)
        ax.set_ylim(100, -100)
    else:
        all_positions = np.array([seg['position'] for seg in frame_data])
        min_coords = all_positions.min(axis=0)
        max_coords = all_positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        view_range = (max_coords - min_coords).max() * 1.2 + 200
        ax.set_xlim(center[0] - view_range / 2, center[0] + view_range / 2)
        ax.set_ylim(center[1] + view_range / 2, center[1] - view_range / 2)

    for segment_info in frame_data:
        x, y = segment_info['position']
        r = segment_info['radius']
        rgba_fill_color = np.array(segment_info['color']) / 255.0
        circle = patches.Circle((x, y), radius=r, facecolor=rgba_fill_color)
        ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Colony State at Frame {frame_number}")
    ax.set_facecolor('black')
    plt.axis('off')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")
    plt.savefig(output_path)
    plt.close(fig)

# Clean output directory
output_directory = "frames"
os.makedirs(output_directory, exist_ok=True)
print(f"Clearing old frames from ./{output_directory}/")
for filename in os.listdir(output_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        os.remove(os.path.join(output_directory, filename))

# --- PARALLEL RENDERING ---
#print(f"\nStarting parallel rendering of {len(my_logger.frames_to_draw_mpl)} frames using all available CPU cores...")
#start_render_time = time.perf_counter()

# Use joblib to parallelize the rendering of saved frames
# n_jobs=-1 uses all available CPU cores
#Parallel(n_jobs=-1)(
#    delayed(render_frame)(data, num, output_directory)
#    for num, data in tqdm(my_logger.frames_to_draw_mpl, desc="Rendering frames")
#)

end_render_time = time.perf_counter()
#print(f"Parallel rendering completed in {end_render_time - start_render_time:.2f} seconds.")
#print(f"Output frames are saved in the '{output_directory}' directory.")

# Add this to the very end of sim_loop.py

import pickle

print("\nSaving simulation data for rendering...")
# The data is in my_logger.frames_to_draw_mpl
# It's a list of tuples: [(frame_number, frame_data), ...]
with open('simulation_output.pkl', 'wb') as f:
    pickle.dump(my_logger.frames_to_draw_mpl, f)

print(f"Simulation data saved to simulation_output.pkl")