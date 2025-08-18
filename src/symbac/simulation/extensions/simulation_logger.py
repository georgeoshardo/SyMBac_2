import time
from tqdm import tqdm
from symbac.utils import generate_color
import typing

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
        # Log the positions of all cell segments every 100 frames
        if simulator.frame_count % 10 == 0:
            current_frame_data = [
                {
                    'position': (seg.body.position.x, seg.body.position.y),
                    'radius': seg.shape.radius,
                    'id': cell.group_id,
                    'color': generate_color(cell.group_id),
                }
                for cell in simulator.cells for seg in cell.physics_representation.segments
            ] # TODO This uses a lot of CPU time, maybe use batched pymunk queries?
            self.frames_to_draw_mpl.append((simulator.frame_count, current_frame_data))