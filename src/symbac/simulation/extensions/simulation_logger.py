import time
from tqdm import tqdm
from symbac.utils import generate_color
import typing
import json
import pickle


if typing.TYPE_CHECKING:
    from symbac.simulation import Simulator

class SimulationLogger:
    def __init__(self):
        # Initialize lists to store simple stats
        self.num_cells = []
        self.t = []

        # Create a tqdm progress bar instance, which we will manually update
        self.pbar = tqdm(total=1000, unit="step", desc="Simulation Progress", smoothing =0.0)
        self.last_time = time.time()

        self.frames_to_draw_mpl = []  # Will log the positions of cells at each frame

    # A function to log the number of cells and time at each frame
    def log_frame(self, simulator: 'Simulator') -> None:
        self.num_cells.append(simulator.num_cells)
        if not self.t:
            self.t.append(0)
        else:
            self.t.append(self.t[-1] + simulator.dt)

    # A function to log the time taken for each step
    def get_step_comp_time(self, simulator: 'Simulator') -> None:
        # Don't update the bar every step, only every 20 steps
        update_interval = 20
        if simulator.frame_count % update_interval != 0:
            return
        current_time = time.time()
        # Calculate the time elapsed for this single step
        step_time_ms = (current_time - self.last_time) * 1000 / update_interval  # Convert to milliseconds and average over the last 20 steps
        self.last_time = current_time
        self.pbar.set_postfix(cells=simulator.num_cells, time_per_step=f"{step_time_ms:.2f}ms")
        # Advance the progress bar by one step
        self.pbar.update(update_interval)

    def log_cell_positions(self, simulator: 'Simulator') -> None:
        # Log the positions of all cell segments every 10 frames
        if simulator.frame_count % 10 == 0:
            # Organize data by cell ID
            current_frame_data = {}
            
            for cell in simulator.cells:
                cell_id = cell.group_id
                cell_segments = []
                
                for seg in cell.physics_representation.segments:
                    segment_data = {
                        'position': (seg.body.position.x, seg.body.position.y),
                        'radius': seg.shape.radius,
                        'color': generate_color(cell_id),
                    }
                    cell_segments.append(segment_data)
                
                current_frame_data[cell_id] = cell_segments
            
            self.frames_to_draw_mpl.append((simulator.frame_count, current_frame_data))

    def get_frames_dict(self) -> dict:
        """
        Convert the frames_to_draw_mpl list to a dictionary format where
        keys are frame counts and values are the frame data.
        
        Returns:
            dict: Dictionary with frame_count as keys and frame_data as values
        """
        return {frame_count: frame_data for frame_count, frame_data in self.frames_to_draw_mpl}

    def save_frames_to_json(self, filepath: str) -> None:
        """
        Save the frames data to a JSON file in a natural dictionary format.
        
        Args:
            filepath (str): Path to save the JSON file
        """
        frames_dict = self.get_frames_dict()
        with open(filepath, 'w') as f:
            json.dump(frames_dict, f, indent=2)

    def save_frames_to_pickle(self, filepath: str) -> None:
        """
        Save the frames data to a pickle file for faster loading.
        
        Args:
            filepath (str): Path to save the pickle file
        """
        frames_dict = self.get_frames_dict()
        with open(filepath, 'wb') as f:
            pickle.dump(frames_dict, f)

    def load_frames_from_json(self, filepath: str) -> dict:
        """
        Load frames data from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file
            
        Returns:
            dict: Dictionary with frame_count as keys and frame_data as values
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_frames_from_pickle(self, filepath: str) -> dict:
        """
        Load frames data from a pickle file.
        
        Args:
            filepath (str): Path to the pickle file
            
        Returns:
            dict: Dictionary with frame_count as keys and frame_data as values
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)