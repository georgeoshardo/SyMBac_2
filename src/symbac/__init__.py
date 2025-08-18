"""
A physics-based simulation framework for generating synthetic microscopy data
of bacterial colonies with realistic growth, division, and interaction dynamics.
"""

from .simulation import Simulator, SimCell
from .ground_truth import MaskGenerator, MaskConfig


__all__ = [
    "Simulator",
    "SimCell", 
    "MaskGenerator",
    "MaskConfig"
]
