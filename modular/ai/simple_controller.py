"""
Simple VLM Controller - Simplified version without complex string formatting.
"""

import json
import time
import random
from typing import Dict, Any, Optional
import requests

from ..core.agent import Agent, Action
from ..core.world import World, CellType
from ..utils.config_loader import config


class SimpleVLMController:
    """
    Simplified VLM controller with basic decision making.
    """

    def __init__(self, model: str = "gemma3:4b", server: str = "http://localhost:11434"):
        self.model = model
        self.server = server
        self.inference_count = 0
        self.total_time = 0.0
        self.last_stats_report = 0

        # Load configuration
        self.config = config.get_ai_behavior_config().get('simple_controller', {})

    def get_action(self, agent: Agent, world: World, observations: Dict[str, Any]):
        """Get an action decision for the agent."""
        start_time = time.time()

        try:
            # Use simple rule-based logic for now
            action = self._get_rule_based_action(agent, observations)
            action_display = action.value if hasattr(
                action, 'value') else str(action)
            print(f"🤖 Agent {agent.id} chose: {action_display}")

        except Exception as e:
            print(f"❌ Error for {agent.id}: {e}")
            action = Action.WAIT

        # Update stats
        self.inference_count += 1
        self.total_time += time.time() - start_time

        if self.inference_count % 10 == 0:
            avg_time = self.total_time / self.inference_count
            print(
                f"🧠 Decision Stats: {self.inference_count} decisions, avg {avg_time:.3f}s")

        return action

    def _get_rule_based_action(self, agent: Agent, observations: Dict[str, Any]):
        """Enhanced rule-based decision making with action sequences for maximum efficiency."""

        # Emergency survival - use sequences for desperate situations
        emergency_energy = self.config.get('emergency_energy', 20)
        if agent.energy < emergency_energy:
            if agent.inventory['food'] > 0 and agent.inventory['water'] > 0:
                return "EAT,DRINK"  # Consume everything available
            elif agent.inventory['food'] > 0:
                return Action.EAT
            elif agent.inventory['water'] > 0:
                return Action.DRINK
            else:
                return Action.WAIT

        # High priority needs with action sequences
        high_hunger = self.config.get('high_priority_hunger', 50)
        high_thirst = self.config.get('high_priority_thirst', 50)

        # Super efficient: if hungry/thirsty and have both resources, consume both
        if (agent.hunger > high_hunger and agent.thirst > high_thirst and
                agent.inventory['food'] > 0 and agent.inventory['water'] > 0):
            return "EAT,DRINK"
        elif agent.hunger > high_hunger and agent.inventory['food'] > 0:
            return Action.EAT
        elif agent.thirst > high_thirst and agent.inventory['water'] > 0:
            return Action.DRINK

        # Resource gathering with action sequences - check current cell first
        current_cell = observations.get('current_cell', 'AGENT')
        max_food = config.get('resources.max_inventory_food', 2)
        max_water = config.get('resources.max_inventory_water', 2)

        # Ultimate efficiency: pickup and immediately consume if needed
        if current_cell == 'FOOD' and agent.inventory['food'] < max_food:
            medium_hunger = self.config.get('medium_priority_hunger', 30)
            if agent.hunger > medium_hunger:
                # Pickup food and eat immediately, plus drink if thirsty
                if agent.thirst > medium_hunger and agent.inventory['water'] > 0:
                    return "PICKUP,EAT,DRINK"
                else:
                    return "PICKUP,EAT"
            else:
                return Action.PICKUP  # Just pickup for later
        elif current_cell == 'WATER' and agent.inventory['water'] < max_water:
            medium_thirst = self.config.get('medium_priority_thirst', 30)
            if agent.thirst > medium_thirst:
                # Pickup water and drink immediately, plus eat if hungry
                if agent.hunger > medium_thirst and agent.inventory['food'] > 0:
                    return "PICKUP,DRINK,EAT"
                else:
                    return "PICKUP,DRINK"
            else:
                return Action.PICKUP  # Just pickup for later

        # Move towards nearby resources with action sequences
        nearby_food = observations.get('food', 0) > 0
        nearby_water = observations.get('water', 0) > 0
        need_food = agent.inventory['food'] < max_food
        need_water = agent.inventory['water'] < max_water

        if nearby_food and need_food:
            # Move toward food and try to pickup
            direction = random.choice(
                ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"])
            # If we're hungry, plan to eat immediately after pickup
            if agent.hunger > self.config.get('medium_priority_hunger', 30):
                return f"{direction},PICKUP,EAT"
            else:
                return f"{direction},PICKUP"
        elif nearby_water and need_water:
            # Move toward water and try to pickup
            direction = random.choice(
                ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"])
            # If we're thirsty, plan to drink immediately after pickup
            if agent.thirst > self.config.get('medium_priority_thirst', 30):
                return f"{direction},PICKUP,DRINK"
            else:
                return f"{direction},PICKUP"

        # Breeding behavior (for mature, healthy agents)
        breeding_energy = config.get('breeding.min_energy', 60)
        breeding_hunger_max = self.config.get('medium_priority_hunger', 40)
        breeding_thirst_max = self.config.get('medium_priority_thirst', 40)
        if (agent.is_mature and agent.can_breed() and
                agent.energy > breeding_energy and agent.hunger < breeding_hunger_max and agent.thirst < breeding_thirst_max):

            if observations.get('agents', 0) > 0:
                # Try breeding first, then seek mate
                if random.random() < 0.3:
                    return Action.BREED
                else:
                    return Action.SEEK_MATE

        # Medium priority needs with action sequences
        medium_hunger = self.config.get('medium_priority_hunger', 30)
        medium_thirst = self.config.get('medium_priority_thirst', 30)

        # Build an action sequence for medium priority needs
        actions_needed = []
        if agent.hunger > medium_hunger and agent.inventory['food'] > 0:
            actions_needed.append("EAT")
        if agent.thirst > medium_thirst and agent.inventory['water'] > 0:
            actions_needed.append("DRINK")

        if actions_needed:
            return ",".join(actions_needed)

        # Exploration/movement
        if agent.energy > 30:
            # Personality-based movement bias
            if "social" in agent.personality and observations.get('agents', 0) > 0:
                # Stay put if other agents nearby
                return Action.WAIT
            elif "aggressive" in agent.personality:
                # More likely to move and explore
                return random.choice([Action.MOVE_UP, Action.MOVE_DOWN,
                                      Action.MOVE_LEFT, Action.MOVE_RIGHT])
            else:
                # Random movement or wait
                actions = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT,
                           Action.MOVE_RIGHT, Action.WAIT, Action.WAIT]  # Wait twice as likely
                return random.choice(actions)

        # Low energy - rest
        return Action.WAIT

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = self.total_time / self.inference_count if self.inference_count > 0 else 0

        return {
            "model": self.model,
            "server": self.server,
            "total_inferences": self.inference_count,
            "total_time": self.total_time,
            "average_time": avg_time,
            "inferences_per_second": 1.0 / avg_time if avg_time > 0 else 0
        }
