"""
AI Arena Modular Version
A modular, extensible implementation of the AI Life simulation.
"""

__version__ = "2.0.0"
__author__ = "AI Arena Team"

from .core.agent import Agent, Gender
from .core.world import World, CellType
from .core.simulation import Simulation
from .ai.simple_controller import SimpleVLMController

__all__ = [
    "Agent", "Gender", "World", "CellType", "Simulation",
    "SimpleVLMController"
]
