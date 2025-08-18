import colorsys
import numpy as np
import typing

from symbac.utils import generate_color
from symbac.simulation.simcell import SimCell


class CellColor:
    def __init__(self):
        self.cell_colors = {
            1: generate_color(1)  # Map a cell ID to a color
        }

    def get_daughter_color(self, mother_cell: SimCell, daughter_cell: SimCell) -> tuple[int, int, int]:
        """
        Returns a color for the daughter cell based on the mother's color.
        This can be customized to implement different inheritance strategies.
        """
        # 1. Get the mother's color and normalize it to the 0-1 range for colorsys
        if mother_cell.group_id not in self.cell_colors:
            # Generate a new color if mother's color doesn't exist
            self.cell_colors[mother_cell.group_id] = generate_color(mother_cell.group_id)
        
        r, g, b = self.cell_colors[mother_cell.group_id]
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        
        # 3. Mutate the Hue to change the color while preserving lineage
        #    A small hue shift changes the color along the color wheel (e.g., red -> orange)
        #    Prevent division by zero by ensuring group_id is at least 1
        safe_group_id = max(1, daughter_cell.group_id)
        hue_shift = np.random.uniform(-1, 1) / (np.sqrt(safe_group_id) * 2)  # Shift hue with biased rw
        new_h = (h + hue_shift) % 1.0  # Use modulo to wrap around the color wheel
        
        #    This prevents colors from becoming grayish or dark.
        #    We'll clamp them to a minimum vibrancy level.
        new_s = s
        new_v = v

        # 5. Convert the new HSV color back to RGB
        new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)

        # 6. Scale back to 0-255 and create the final tuple
        daughter_color = (int(new_r * 255), int(new_g * 255), int(new_b * 255))
        return daughter_color

    def update_color(self, cell: SimCell) -> None:
        """
        Update the color of a cell based on its division count.
        """
        a = 255
        
        # Ensure cell has a color assigned
        if cell.group_id not in self.cell_colors:
            self.cell_colors[cell.group_id] = generate_color(cell.group_id)
        
        r, g, b = self.cell_colors[cell.group_id]

        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        new_s = max(s / np.sqrt(cell.num_divisions + 1), 0.3)  # Ensure saturation is not too low
        new_v = max(v / np.sqrt(cell.num_divisions + 1), 0.3)  # Ensure brightness is not too low

        # 5. Convert the new HSV color back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
        r, g, b = (int(r * 255), int(g * 255), int(b * 255))

        body_color = (r, g, b, a)
        head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), a)
        tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), a)
        
        # Check if segments exist before accessing them
        if hasattr(cell.physics_representation, 'segments') and cell.physics_representation.segments:
            for segment in cell.physics_representation.segments:  # You have to set a color attribute for pygame
                segment.shape.color = body_color
            
            # Set head and tail colors if segments exist
            if len(cell.physics_representation.segments) > 0:
                cell.physics_representation.segments[0].shape.color = head_color
                cell.physics_representation.segments[-1].shape.color = tail_color

    def update_daughter_color(self, mother_cell: SimCell, daughter_cell: SimCell) -> None:
        """
        Update the daughter cell's color based on the mother's color.
        This is called after a division occurs.
        """
        daughter_color = self.get_daughter_color(mother_cell, daughter_cell)
        self.cell_colors[daughter_cell.group_id] = daughter_color
        self.update_color(daughter_cell)

    # Backward compatibility aliases
    def get_daughter_colour(self, mother_cell: SimCell, daughter_cell: SimCell) -> tuple[int, int, int]:
        """Alias for get_daughter_color for backward compatibility."""
        return self.get_daughter_color(mother_cell, daughter_cell)
    
    def update_colour(self, cell: SimCell) -> None:
        """Alias for update_color for backward compatibility."""
        return self.update_color(cell)
    
    def update_daughter_colour(self, mother_cell: SimCell, daughter_cell: SimCell) -> None:
        """Alias for update_daughter_color for backward compatibility."""
        return self.update_daughter_color(mother_cell, daughter_cell)
