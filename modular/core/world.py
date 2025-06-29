"""
World module - Environment and grid management.
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import random
import numpy as np


class CellType(Enum):
    """Types of cells in the world grid."""
    EMPTY = 0
    FOOD = 1
    WATER = 2
    OBSTACLE = 3
    AGENT = 4


class World:
    """
    Manages the simulation environment and grid.

    Features:
    - Dynamic resource management
    - Natural resource regeneration
    - Obstacle placement
    - Spatial queries and operations
    """

    def __init__(self, width: int, height: int, resource_mode: str = "normal"):
        """
        Initialize the world.

        Args:
            width: Grid width
            height: Grid height
            resource_mode: Resource abundance ("scarce", "normal", "abundant")
        """
        self.width = width
        self.height = height
        self.resource_mode = resource_mode

        # Initialize grid
        self.grid = np.zeros((height, width), dtype=int)
        self.resource_grid = {}  # Position -> CellType mapping for resources
        self.agent_positions = {}  # Position -> Agent ID mapping

        # Resource configuration
        self.resource_configs = {
            "scarce": {"food_ratio": 0.08, "water_ratio": 0.06, "obstacle_ratio": 0.12},
            "normal": {"food_ratio": 0.15, "water_ratio": 0.12, "obstacle_ratio": 0.08},
            "abundant": {"food_ratio": 0.25, "water_ratio": 0.20, "obstacle_ratio": 0.05}
        }

        self.config = self.resource_configs.get(
            resource_mode, self.resource_configs["normal"])

        # Statistics
        self.food_count = 0
        self.water_count = 0
        self.obstacle_count = 0

        # Generate initial world
        self._generate_world()

    def _generate_world(self):
        """Generate the initial world layout."""
        total_cells = self.width * self.height

        # Calculate resource amounts
        food_amount = int(total_cells * self.config["food_ratio"])
        water_amount = int(total_cells * self.config["water_ratio"])
        obstacle_amount = int(total_cells * self.config["obstacle_ratio"])

        # Generate all positions
        all_positions = [(x, y) for x in range(self.width)
                         for y in range(self.height)]
        random.shuffle(all_positions)

        # Place obstacles first
        for i in range(obstacle_amount):
            if i < len(all_positions):
                pos = all_positions[i]
                self._place_cell(pos, CellType.OBSTACLE)

        # Place food
        for i in range(obstacle_amount, obstacle_amount + food_amount):
            if i < len(all_positions):
                pos = all_positions[i]
                self._place_cell(pos, CellType.FOOD)

        # Place water
        for i in range(obstacle_amount + food_amount,
                       obstacle_amount + food_amount + water_amount):
            if i < len(all_positions):
                pos = all_positions[i]
                self._place_cell(pos, CellType.WATER)

        print(f"🌍 World created: {self.obstacle_count} obstacles, "
              f"{self.food_count} food, {self.water_count} water")

    def _place_cell(self, position: Tuple[int, int], cell_type: CellType):
        """Place a cell at the specified position."""
        x, y = position
        if self.is_valid_position(x, y):
            self.grid[y, x] = cell_type.value
            if cell_type != CellType.EMPTY:
                self.resource_grid[position] = cell_type

            # Update counters
            if cell_type == CellType.FOOD:
                self.food_count += 1
            elif cell_type == CellType.WATER:
                self.water_count += 1
            elif cell_type == CellType.OBSTACLE:
                self.obstacle_count += 1

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within world bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell_type(self, x: int, y: int) -> CellType:
        """Get the type of cell at position."""
        if not self.is_valid_position(x, y):
            return CellType.OBSTACLE  # Treat out-of-bounds as obstacle

        return CellType(self.grid[y, x])

    def set_cell_type(self, x: int, y: int, cell_type: CellType):
        """Set the type of cell at position."""
        if self.is_valid_position(x, y):
            old_type = self.get_cell_type(x, y)

            # Update counters
            if old_type == CellType.FOOD:
                self.food_count -= 1
            elif old_type == CellType.WATER:
                self.water_count -= 1

            if cell_type == CellType.FOOD:
                self.food_count += 1
            elif cell_type == CellType.WATER:
                self.water_count += 1

            # Update grid
            self.grid[y, x] = cell_type.value

            # Update resource mapping
            pos = (x, y)
            if cell_type == CellType.EMPTY:
                self.resource_grid.pop(pos, None)
            else:
                self.resource_grid[pos] = cell_type

    def place_agent(self, x: int, y: int, agent_id: str):
        """Place an agent at the specified position."""
        if self.is_valid_position(x, y):
            self.agent_positions[(x, y)] = agent_id

    def remove_agent(self, x: int, y: int):
        """Remove an agent from the specified position."""
        self.agent_positions.pop((x, y), None)

    def move_agent(self, old_x: int, old_y: int, new_x: int, new_y: int, agent_id: str):
        """Move an agent from old position to new position."""
        self.remove_agent(old_x, old_y)
        self.place_agent(new_x, new_y, agent_id)

    def has_agent(self, x: int, y: int) -> bool:
        """Check if there's an agent at the specified position."""
        return (x, y) in self.agent_positions

    def get_agent_at(self, x: int, y: int) -> Optional[str]:
        """Get the agent ID at the specified position."""
        return self.agent_positions.get((x, y))

    def can_move_to(self, x: int, y: int) -> bool:
        """Check if an agent can move to this position."""
        if not self.is_valid_position(x, y):
            return False

        # Check if there's already an agent there
        if self.has_agent(x, y):
            return False

        cell_type = self.get_cell_type(x, y)
        return cell_type in [CellType.EMPTY, CellType.FOOD, CellType.WATER]

    def pickup_resource(self, x: int, y: int) -> Optional[CellType]:
        """
        Pick up a resource at the given position.

        Returns:
            The type of resource picked up, or None if nothing to pick up.
        """
        cell_type = self.get_cell_type(x, y)

        if cell_type in [CellType.FOOD, CellType.WATER]:
            self.set_cell_type(x, y, CellType.EMPTY)
            return cell_type

        return None

    def get_neighbors(self, x: int, y: int, radius: int = 1) -> List[Tuple[int, int, CellType]]:
        """
        Get neighboring cells within radius.

        Returns:
            List of (x, y, cell_type) tuples for neighbors.
        """
        neighbors = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny):
                    cell_type = self.get_cell_type(nx, ny)
                    neighbors.append((nx, ny, cell_type))

        return neighbors

    def find_nearest_resource(self, x: int, y: int, resource_type: CellType,
                              max_distance: int = 10) -> Optional[Tuple[int, int]]:
        """
        Find the nearest resource of the specified type.

        Returns:
            Position (x, y) of nearest resource, or None if not found.
        """
        best_distance = float('inf')
        best_position = None

        for pos, cell_type in self.resource_grid.items():
            if cell_type == resource_type:
                px, py = pos
                distance = abs(px - x) + abs(py - y)  # Manhattan distance

                if distance <= max_distance and distance < best_distance:
                    best_distance = distance
                    best_position = pos

        return best_position

    def get_area_summary(self, x: int, y: int, radius: int = 2) -> Dict[str, int]:
        """
        Get a summary of resources in an area around position.
        Includes boundary cells that are outside the world.

        Returns:
            Dictionary with counts of each resource type and boundary cells.
        """
        summary = {
            "food": 0,
            "water": 0,
            "obstacles": 0,
            "empty": 0,
            "agents": 0,
            "boundary_cells": 0
        }

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny):
                    cell_type = self.get_cell_type(nx, ny)

                    if cell_type == CellType.FOOD:
                        summary["food"] += 1
                    elif cell_type == CellType.WATER:
                        summary["water"] += 1
                    elif cell_type == CellType.OBSTACLE:
                        summary["obstacles"] += 1
                    elif cell_type == CellType.EMPTY:
                        summary["empty"] += 1
                    elif cell_type == CellType.AGENT:
                        summary["agents"] += 1
                else:
                    # Position is outside world boundaries
                    summary["boundary_cells"] += 1

        return summary

    def regenerate_resources(self, turn: int):
        """
        Natural resource regeneration over time.

        Args:
            turn: Current simulation turn for regeneration timing.
        """
        # Regenerate every 5 turns
        if turn % 5 != 0:
            return

        # Calculate regeneration amounts based on current scarcity
        total_cells = self.width * self.height
        target_food = int(total_cells * self.config["food_ratio"])
        target_water = int(total_cells * self.config["water_ratio"])

        food_deficit = max(0, target_food - self.food_count)
        water_deficit = max(0, target_water - self.water_count)

        # Regenerate food
        food_to_add = min(food_deficit, max(1, food_deficit // 3))
        self._regenerate_resource_type(CellType.FOOD, food_to_add)

        # Regenerate water
        water_to_add = min(water_deficit, max(1, water_deficit // 5))
        self._regenerate_resource_type(CellType.WATER, water_to_add)

        if food_to_add > 0 or water_to_add > 0:
            print(
                f"🌱 Natural regeneration: +{food_to_add}🍎 +{water_to_add}💧 (Turn {turn})")
            print(
                f"   Current resources: {self.food_count}🍎 {self.water_count}💧")

    def _regenerate_resource_type(self, resource_type: CellType, amount: int):
        """Regenerate a specific type of resource."""
        placed = 0
        attempts = 0
        max_attempts = amount * 10  # Prevent infinite loops

        while placed < amount and attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)

            if self.get_cell_type(x, y) == CellType.EMPTY:
                self.set_cell_type(x, y, resource_type)
                placed += 1

            attempts += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get world statistics."""
        total_cells = self.width * self.height
        occupied_cells = self.food_count + self.water_count + self.obstacle_count

        return {
            "width": self.width,
            "height": self.height,
            "total_cells": total_cells,
            "food_count": self.food_count,
            "water_count": self.water_count,
            "obstacle_count": self.obstacle_count,
            "empty_cells": total_cells - occupied_cells,
            "resource_mode": self.resource_mode,
            "food_density": self.food_count / total_cells,
            "water_density": self.water_count / total_cells,
            "obstacle_density": self.obstacle_count / total_cells
        }

    def get_cell_type_for_display(self, x: int, y: int) -> CellType:
        """Get the cell type for display purposes (prioritizes agents)."""
        if self.has_agent(x, y):
            return CellType.AGENT
        return self.get_cell_type(x, y)

    def get_underlying_resource(self, x: int, y: int) -> Optional[CellType]:
        """Get the resource type at a position, ignoring agents."""
        return self.resource_grid.get((x, y))

    def get_nearby_cells_detailed(self, x: int, y: int, radius: int = 2) -> Dict[str, str]:
        """
        Get detailed information about nearby cells with their positions.
        Includes boundary information when positions are outside the world.

        Returns:
            Dictionary mapping position strings to cell type names or "BOUNDARY".
        """
        nearby_cells = {}

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip center position

                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny):
                    cell_type = self.get_cell_type(nx, ny)
                    # Check for agents at this position
                    if self.has_agent(nx, ny):
                        nearby_cells[f"({nx},{ny})"] = "AGENT"
                    else:
                        nearby_cells[f"({nx},{ny})"] = cell_type.name
                else:
                    # Position is outside world boundaries
                    nearby_cells[f"({nx},{ny})"] = "BOUNDARY"

        return nearby_cells
