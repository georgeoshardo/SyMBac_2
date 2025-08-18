# Use sudo py-spy top --pid $(ps aux | grep python | grep -v grep | awk '{print $2, $3}' | sort -rn -k2 | head -1 | awk '{print $1}') for profiling
import colorsys
import numpy as np
from pymunk import Vec2d

from symbac.simulation import Simulator
from symbac.simulation.extensions.live_visualisation import LiveVisualisation
from symbac.simulation.extensions.microfluidic_geometry import trench_creator, box_creator

np.random.seed(42)
from symbac.simulation.simcell import SimCell
from symbac.simulation.config import CellConfig, SimViewerConfig, PhysicsConfig
import numpy as np


physics_config = PhysicsConfig(
    THREADED=True,
    THREADS=2,
    ITERATIONS=50,
)

#TODO: higher granularity require higher stiffness
initial_cell_config = CellConfig(
    GRANULARITY=4, # 16 is good for precise division with no gaps, 8 is a good compromise between performance and precision, 3 is for speed
    SEGMENT_RADIUS=10,
    SEGMENT_MASS=1.0,
    GROWTH_RATE=5, # Turning up the growth rate is a good way to speed up the simulation while keeping ITERATIONS high,
    BASE_MAX_LENGTH=180, # This should be stable now!
    MAX_LENGTH_VARIATION=0.3,
    MIN_LENGTH_AFTER_DIVISION=4,
    NOISE_STRENGTH=0.05,
    SEED_CELL_SEGMENTS=30,
    ROTARY_LIMIT_JOINT=True,
    MAX_BEND_ANGLE=0.005,
    START_POS=Vec2d(0, 100),
    START_ANGLE=np.pi/2,
    STIFFNESS=300_000 , # Common values: (bend angle = 0.005, stiffness = 300_000), you can use np.inf for max stiffness but ideally use np.iinfo(np.int64).max for integer type
    #DAMPED_ROTARY_SPRING=True,  # Enable damped rotary springs, makes cells quite rigid
    #ROTARY_SPRING_STIFFNESS=2000_000, # A good starting point
    #ROTARY_SPRING_DAMPING=200_000, # A good starting point
    PIVOT_JOINT_STIFFNESS=5000, # This can be lowered from the default np.inf, and the cell will be able to compress
    SIMPLE_LENGTH=False # If true, cell length is calculated continuously each time the length attribute is requested (QUITE!! slow, but higher precision because allows for cell compression) If false it is simply the segment distance * number of segments in a cell
)

simulator = Simulator(physics_config, initial_cell_config)


import pymunk
def segment_creator(local_xy1, local_xy2, global_xy, thickness):
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1, local_xy2, thickness)
    segment_body.position = global_xy
    segment_shape.friction = 0
    return segment_body, segment_shape

#trench_creator(50, 1000, (-0, -0), simulator.space)

#Use a closure to create a post-init hook that adds a box
def trench_adder(simulator: 'Simulator') -> None:
    trench_creator(30, 1000, (-0, -0), simulator.space)

#Use a closure to create a post-init hook that adds a box
def box_adder(simulator: 'Simulator') -> None:
    box_creator(1000, 1000, (0, 0), simulator.space, barrier_thickness=10, fillet_radius=100)

#simulator.add_and_run_post_init_hook(box_adder)

def cell_remover(simulator: 'Simulator') -> None:
    for cell in simulator.cells:
        if cell.physics_representation.segments[0].body.position.y > 1000:
            simulator.colony.delete_cell(cell)

#simulator.add_post_step_hook(cell_remover)


def cell_growth_rate_updater(cell: SimCell) -> None:
    compression_ratio = cell.physics_representation.get_compression_ratio()
    cell.adjusted_growth_rate = cell.config.GROWTH_RATE * compression_ratio**4

    variation = cell.config.BASE_MAX_LENGTH * cell.config.MAX_LENGTH_VARIATION
    random_max_len = np.random.uniform(
        cell.config.BASE_MAX_LENGTH - variation, cell.config.BASE_MAX_LENGTH + variation
    ) * np.sqrt(compression_ratio)

    cell.max_length = max(cell.length, int(random_max_len))

#simulator.add_pre_cell_grow_hook(cell_growth_rate_updater)

from symbac.simulation.extensions.cell_color import CellColor

cell_colouriser = CellColor()

simulator.add_pre_cell_grow_hook(cell_colouriser.update_colour)
simulator.add_post_cell_grow_hook(cell_colouriser.update_colour)
simulator.add_post_division_hook(cell_colouriser.update_daughter_colour)

# Create an object to log the simulation context each frame for plotting later
import time

from symbac.simulation.extensions.simulation_logger import SimulationLogger

my_logger = SimulationLogger()

simulator.add_post_step_hook(my_logger.log_frame)
simulator.add_post_step_hook(my_logger.get_step_comp_time)
simulator.add_post_step_hook(my_logger.log_cell_positions)


sim_viewer_config = SimViewerConfig(SIM_STEPS_PER_DRAW=20)
live_visualisation = LiveVisualisation(sim_viewer_config)

simulator.add_post_step_hook(live_visualisation.draw)

frames_to_render = [] # List to store data for rendering
image_count = 0
while live_visualisation.running:
        simulator.step()
        if simulator.num_cells > 200:
            print("Simulation stopped.")
            break

my_logger.save_frames_to_json("frames.json")