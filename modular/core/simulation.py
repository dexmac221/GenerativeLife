"""
Simulation module - Main simulation engine and coordination.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import time

from .agent import Agent, Gender, Action
from .world import World, CellType


class Simulation:
    """
    Main simulation engine that coordinates agents, world, and AI.

    Features:
    - Agent lifecycle management
    - Breeding and reproduction system
    - Resource management
    - Turn-based simulation loop
    - Statistics tracking
    """

    def __init__(self, world: World, ai_controller, config: Dict[str, Any]):
        """
        Initialize simulation.

        Args:
            world: The world environment
            ai_controller: AI controller for agent decisions
            config: Simulation configuration parameters
        """
        self.world = world
        self.ai_controller = ai_controller
        self.config = config

        # Simulation state
        self.turn = 0
        self.agents: List[Agent] = []
        self.agent_counter = 0
        self.running = False

        # Statistics
        self.stats = {
            "births": 0,
            "deaths": 0,
            "total_agents_created": 0,
            "peak_population": 0,
            "breeding_events": 0,
            "resource_pickups": 0
        }

        # Breeding configuration
        self.min_breeding_age = config.get("min_breeding_age", 50)
        self.pregnancy_duration = config.get("pregnancy_duration", 30)
        self.max_population = config.get("max_population", 50)

        # Observation configuration
        self.observation_radius = config.get("observation_radius", 2)

        # Personality templates
        self.personalities = [
            "cautious explorer who prioritizes safety",
            "aggressive resource hoarder who takes risks",
            "social collaborator who seeks other agents",
            "methodical strategist who plans ahead",
            "opportunistic survivor who adapts quickly",
            "efficient collector who minimizes energy waste",
            "curious wanderer who explores extensively",
            "protective guardian who helps others"
        ]

    def create_agent(self, position: Optional[Tuple[int, int]] = None,
                     gender: Optional[Gender] = None,
                     personality: Optional[str] = None) -> Agent:
        """
        Create a new agent in the simulation.

        Args:
            position: Starting position (random if None)
            gender: Agent gender (alternating if None)
            personality: Agent personality (random if None)

        Returns:
            The created agent
        """
        # Generate position
        if position is None:
            position = self._find_spawn_position()

        # Assign gender (alternating pattern)
        if gender is None:
            gender = Gender.MALE if self.agent_counter % 2 == 0 else Gender.FEMALE

        # Assign personality
        if personality is None:
            personality = random.choice(self.personalities)

        # Create agent
        agent = Agent(
            id=f"Agent_{self.agent_counter}",
            position=position,
            gender=gender,
            personality=personality,
            maturity_age=self.min_breeding_age,
            pregnancy_duration=self.pregnancy_duration
        )

        # Place in world - track agent position separately from resources
        x, y = position
        self.world.place_agent(x, y, agent.id)

        # Add to simulation
        self.agents.append(agent)
        self.agent_counter += 1
        self.stats["total_agents_created"] += 1

        # Update peak population
        if len(self.agents) > self.stats["peak_population"]:
            self.stats["peak_population"] = len(self.agents)

        print(
            f"✅ Created {agent.id} ({getattr(agent.gender, 'value', str(agent.gender))}) at {position} - {personality}")

        return agent

    def _find_spawn_position(self) -> Tuple[int, int]:
        """Find a valid spawn position for a new agent."""
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            x = random.randint(0, self.world.width - 1)
            y = random.randint(0, self.world.height - 1)

            if self.world.can_move_to(x, y):
                return (x, y)

            attempts += 1

        # Fallback: find any empty space
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.can_move_to(x, y):
                    return (x, y)

        # Last resort: return center
        return (self.world.width // 2, self.world.height // 2)

    def step(self):
        """Execute one simulation step."""
        self.turn += 1
        print(f"\n🔄 === TURN {self.turn} ===")

        # Remove dead agents
        self._remove_dead_agents()

        if not self.agents:
            print("❌ No agents remaining. Simulation ended.")
            self.running = False
            return

        print(f"Processing {len(self.agents)} agents...")

        # Process each agent
        actions = []
        for i, agent in enumerate(self.agents):
            if agent.alive:
                action = self._process_agent(agent, i)
                actions.append((agent, action))

        # Execute all actions simultaneously
        self._execute_actions(actions)

        # Handle births from pregnancies
        self._handle_births()

        # World updates
        self.world.regenerate_resources(self.turn)

        print(f"🔄 Turn {self.turn} complete\n")

    def _process_agent(self, agent: Agent, index: int) -> Action:
        """Process a single agent's decision making."""
        print(f"\n👤 Processing Agent {agent.id} (index {index})")
        print(f"   Alive: {agent.alive}, Position: {agent.position}")
        print(
            f"   💓 Vitals - Energy: {agent.energy}, Hunger: {agent.hunger}, Thirst: {agent.thirst}")

        # Update vitals
        agent.update_vitals()

        if not agent.alive:
            print(f"💀 Agent {agent.id} died!")
            return Action.WAIT

        # Get observations
        x, y = agent.position
        observations = self.world.get_area_summary(
            x, y, radius=self.observation_radius)

        # Get detailed nearby cells with positions (like monolithic version)
        nearby_cells_detailed = self.world.get_nearby_cells_detailed(
            x, y, radius=self.observation_radius)
        observations['nearby_cells'] = nearby_cells_detailed

        # Add world boundaries and current position information
        observations['world_width'] = self.world.width
        observations['world_height'] = self.world.height
        observations['current_position'] = f"({x},{y})"

        # Get the underlying resource type (what the agent is standing on)
        underlying_resource = self.world.get_underlying_resource(x, y)
        current_cell_name = underlying_resource.name if underlying_resource else "EMPTY"

        # Add current cell information to observations
        observations['current_cell'] = current_cell_name

        print(
            f"   📡 Observations - Current: {current_cell_name}, Nearby: {observations}")

        # Get AI decision
        action = self.ai_controller.get_action(agent, self.world, observations)
        action_display = action.value if hasattr(
            action, 'value') else str(action)
        print(f"   🎯 Chosen action: {action_display}")

        return action

    def _execute_actions(self, actions: List[Tuple[Agent, Any]]):
        """Execute all agent actions simultaneously."""
        print(f"\n🎬 Executing {len(actions)} actions simultaneously...")

        # Separate actions by type
        movement_actions = []
        other_actions = []

        for agent, action in actions:
            action_value = action.value if hasattr(
                action, 'value') else str(action)
            if action_value.startswith("MOVE_"):
                movement_actions.append((agent, action))
            else:
                other_actions.append((agent, action))

        # Execute non-movement actions first
        for agent, action in other_actions:
            self._execute_single_action(agent, action)

        # Execute movement actions
        if movement_actions:
            print(f"🚶 Processing {len(movement_actions)} movement actions...")
            for agent, action in movement_actions:
                self._execute_single_action(agent, action)

    def _execute_single_action(self, agent: Agent, action):
        """Execute a single agent action (including action sequences)."""
        # Handle both Action enum and string sequences
        action_value = action.value if hasattr(
            action, 'value') else str(action)

        # Check if this is an action sequence (comma-separated)
        if ',' in action_value:
            return self._execute_action_sequence_from_string(agent, action_value)

        # Convert string back to Action enum if needed
        if isinstance(action, str):
            try:
                action = Action(action)
            except ValueError:
                print(f"❌ Invalid action: {action}")
                return

        print(
            f"🎯 Agent {agent.id} executing {action.value} from {agent.position}")

        # Store initial state for data collection
        initial_energy = agent.energy
        initial_inventory = agent.inventory.copy()
        success = False
        outcome_data = {}

        # Handle compound actions by breaking them into components
        if self._is_compound_action(action):
            success, outcome_data = self._execute_compound_action(
                agent, action, initial_inventory)
        elif action.value.startswith("MOVE_"):
            success = self._handle_movement(agent, action)
        elif action == Action.PICKUP:
            success = self._handle_pickup(agent)
            # Check if anything was collected
            if agent.inventory['food'] > initial_inventory['food']:
                outcome_data['food_collected'] = agent.inventory['food'] - \
                    initial_inventory['food']
            if agent.inventory['water'] > initial_inventory['water']:
                outcome_data['water_collected'] = agent.inventory['water'] - \
                    initial_inventory['water']
        elif action == Action.EAT:
            success = self._handle_eat(agent)
            if success:
                outcome_data['food_consumed'] = 1
        elif action == Action.DRINK:
            success = self._handle_drink(agent)
            if success:
                outcome_data['water_consumed'] = 1
        elif action == Action.SEEK_MATE:
            success = self._handle_seek_mate(agent)
        elif action == Action.BREED:
            success = self._handle_breeding(agent)
            if success:
                outcome_data['breeding_attempt'] = True
        elif action == Action.WAIT:
            success = self._handle_wait(agent)

        # Collect outcome data for training
        if hasattr(self.ai_controller, 'collect_action_outcome'):
            energy_change = agent.energy - initial_energy
            self.ai_controller.collect_action_outcome(
                agent=agent,
                success=success,
                energy_change=energy_change,
                **outcome_data
            )

        # Add to memory
        action_value = action.value if hasattr(
            action, 'value') else str(action)
        agent.add_memory(action_value, "executed", {"turn": self.turn})

    def _handle_movement(self, agent: Agent, action: Action):
        """Handle agent movement."""
        x, y = agent.position

        # Calculate new position
        if action == Action.MOVE_UP:
            new_x, new_y = x, y - 1
        elif action == Action.MOVE_DOWN:
            new_x, new_y = x, y + 1
        elif action == Action.MOVE_LEFT:
            new_x, new_y = x - 1, y
        elif action == Action.MOVE_RIGHT:
            new_x, new_y = x + 1, y
        else:
            return False

        print(f"🚶 Agent {agent.id} trying to move to ({new_x}, {new_y})")

        # Check if movement is valid
        if self.world.can_move_to(new_x, new_y):
            target_cell = self.world.get_cell_type(new_x, new_y)
            print(f"   Target cell type: {target_cell}")

            # Move agent position tracking
            self.world.move_agent(x, y, new_x, new_y, agent.id)
            agent.position = (new_x, new_y)

            print(f"✅ Agent {agent.id} moved to ({new_x}, {new_y})")
            return True
        else:
            target_cell = self.world.get_cell_type(new_x, new_y)
            print(f"❌ Agent {agent.id} movement blocked by {target_cell.name}")
            return False

    def _handle_pickup(self, agent: Agent):
        """Handle resource pickup."""
        print(f"⏸️ Agent {agent.id} staying at current position")

        x, y = agent.position
        resource = self.world.pickup_resource(x, y)

        if resource == CellType.FOOD:
            agent.inventory["food"] += 1
            self.stats["resource_pickups"] += 1
            print(
                f"✅ Agent {agent.id} picked up FOOD! Inventory: {agent.inventory}")
            return True
        elif resource == CellType.WATER:
            agent.inventory["water"] += 1
            self.stats["resource_pickups"] += 1
            print(
                f"✅ Agent {agent.id} picked up WATER! Inventory: {agent.inventory}")
            return True
        else:
            underlying = self.world.resource_grid.get((x, y))
            print(
                f"❌ Agent {agent.id} tried to pickup but nothing here. Underlying: {underlying}")
            return False

    def _handle_eat(self, agent: Agent):
        """Handle eating action."""
        print(f"⏸️ Agent {agent.id} staying at current position")

        if agent.consume_food():
            print(
                f"✅ Agent {agent.id} ate food! Hunger: {agent.hunger}, Energy +15")
            return True
        else:
            print(f"❌ Agent {agent.id} has no food to eat!")
            return False

    def _handle_drink(self, agent: Agent):
        """Handle drinking action."""
        print(f"⏸️ Agent {agent.id} staying at current position")

        if agent.consume_water():
            print(
                f"✅ Agent {agent.id} drank water! Thirst: {agent.thirst}, Energy +10")
            return True
        else:
            print(f"❌ Agent {agent.id} has no water to drink!")
            return False

    def _handle_seek_mate(self, agent: Agent):
        """Handle mate seeking behavior."""
        print(f"💕 Agent {agent.id} seeking mate...")

        if not agent.can_breed():
            print(
                f"❌ Agent {agent.id} cannot breed (not mature/fertile/pregnant)")
            return False

        # Find potential mates nearby
        x, y = agent.position
        potential_mates = []

        for other_agent in self.agents:
            if other_agent != agent and other_agent.alive:
                ox, oy = other_agent.position
                distance = abs(ox - x) + abs(oy - y)

                if distance <= 3 and agent.is_compatible_mate(other_agent):
                    potential_mates.append(other_agent)

        if potential_mates:
            mate = random.choice(potential_mates)
            print(f"💕 Agent {agent.id} found potential mate: {mate.id}")

            # Move towards mate if not adjacent
            mx, my = mate.position
            if abs(mx - x) + abs(my - y) > 1:
                # Simple pathfinding towards mate
                if mx > x and self.world.can_move_to(x + 1, y):
                    self._move_agent(agent, (x + 1, y))
                elif mx < x and self.world.can_move_to(x - 1, y):
                    self._move_agent(agent, (x - 1, y))
                elif my > y and self.world.can_move_to(x, y + 1):
                    self._move_agent(agent, (x, y + 1))
                elif my < y and self.world.can_move_to(x, y - 1):
                    self._move_agent(agent, (x, y - 1))
            return True
        else:
            print(f"❌ Agent {agent.id} found no compatible mates nearby")
            return False

    def _handle_breeding(self, agent: Agent):
        """Handle breeding action."""
        print(f"💕 Agent {agent.id} attempting to breed...")

        if not agent.can_breed():
            print(f"❌ Agent {agent.id} cannot breed")
            return False

        # Find adjacent compatible mate
        x, y = agent.position
        for other_agent in self.agents:
            if other_agent != agent and other_agent.alive:
                ox, oy = other_agent.position

                # Check if adjacent (distance = 1)
                if abs(ox - x) + abs(oy - y) == 1 and agent.is_compatible_mate(other_agent):
                    # Breeding successful! - with safe gender checking
                    try:
                        agent_is_female = (hasattr(agent.gender, 'value') and agent.gender.value == 'FEMALE') or \
                            (not hasattr(agent.gender, 'value')
                             and str(agent.gender) == 'FEMALE')
                        other_is_female = (hasattr(other_agent.gender, 'value') and other_agent.gender.value == 'FEMALE') or \
                            (not hasattr(other_agent.gender, 'value')
                             and str(other_agent.gender) == 'FEMALE')

                        female = agent if agent_is_female else other_agent
                        male = other_agent if agent_is_female else agent
                    except AttributeError:
                        # Default assignment if gender access fails
                        female = agent
                        male = other_agent

                    female.start_pregnancy(male)
                    self.stats["breeding_events"] += 1

                    print(
                        f"💕 Breeding successful! {female.id} is now pregnant by {male.id}")
                    print(
                        f"   Pregnancy duration: {female.pregnancy_duration} turns")
                    return True

        print(f"❌ Agent {agent.id} found no adjacent compatible mate")
        return False

    def _handle_wait(self, agent: Agent):
        """Handle wait action."""
        print(f"⏸️ Agent {agent.id} waiting/resting")
        return True  # Wait is always successful
        # Slight energy recovery for waiting
        agent.energy = min(100, agent.energy + 1)

    def _handle_births(self):
        """Handle births from completed pregnancies."""
        new_agents = []

        for agent in self.agents:
            if agent.is_pregnant and agent.update_pregnancy():
                # Birth time!
                if len(self.agents) + len(new_agents) < self.max_population:
                    child = self._create_child(agent)
                    if child:
                        new_agents.append(child)
                        agent.add_child(child.id)
                        if agent.partner:
                            agent.partner.add_child(child.id)

                        self.stats["births"] += 1
                        print(f"👶 Birth! {agent.id} gave birth to {child.id}")

                # Reset pregnancy state
                agent.is_pregnant = False
                agent.partner = None

        # Add new agents to simulation
        self.agents.extend(new_agents)

        # Update peak population
        if len(self.agents) > self.stats["peak_population"]:
            self.stats["peak_population"] = len(self.agents)

    def _create_child(self, mother: Agent) -> Optional[Agent]:
        """Create a child agent from parents."""
        # Find spawn position near mother
        x, y = mother.position
        spawn_positions = [
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
            (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)
        ]

        spawn_pos = None
        for pos in spawn_positions:
            if self.world.can_move_to(pos[0], pos[1]):
                spawn_pos = pos
                break

        if not spawn_pos:
            print(f"❌ No space for birth near {mother.id}")
            return None

        # Inherit traits
        child_gender = random.choice([Gender.MALE, Gender.FEMALE])

        # Mix parent personalities
        parent_personalities = [mother.personality]
        if mother.partner:
            parent_personalities.append(mother.partner.personality)

        # Create hybrid personality or choose from parents
        if len(parent_personalities) > 1 and random.random() < 0.3:
            # Create hybrid
            child_personality = f"hybrid of {' and '.join(parent_personalities[:2])}"
        else:
            # Inherit from one parent
            child_personality = random.choice(parent_personalities)

        return self.create_agent(
            position=spawn_pos,
            gender=child_gender,
            personality=child_personality
        )

    def _remove_dead_agents(self):
        """Remove dead agents from simulation."""
        dead_agents = [agent for agent in self.agents if not agent.alive]

        for agent in dead_agents:
            # Clear agent from world tracking
            x, y = agent.position
            self.world.remove_agent(x, y)

            # Remove from agents list
            self.agents.remove(agent)
            self.stats["deaths"] += 1

            print(f"💀 Agent {agent.id} removed (died)")

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        alive_agents = [a for a in self.agents if a.alive]

        # Safe gender counting
        males = 0
        females = 0
        for a in alive_agents:
            try:
                is_male = (hasattr(a.gender, 'value') and a.gender.value == 'MALE') or \
                    (not hasattr(a.gender, 'value') and str(a.gender) == 'MALE')
                if is_male:
                    males += 1
                else:
                    females += 1
            except AttributeError:
                males += 1  # Default to male if gender access fails

        return {
            "turn": self.turn,
            "alive_agents": len(alive_agents),
            "total_agents": len(self.agents),
            "births": self.stats["births"],
            "deaths": self.stats["deaths"],
            "total_created": self.stats["total_agents_created"],
            "peak_population": self.stats["peak_population"],
            "breeding_events": self.stats["breeding_events"],
            "resource_pickups": self.stats["resource_pickups"],
            "males": males,
            "females": females,
            "mature_agents": len([a for a in alive_agents if a.is_mature]),
            "pregnant_agents": len([a for a in alive_agents if a.is_pregnant]),
            "world_stats": self.world.get_statistics()
        }

    def finalize_data_collection(self):
        """Finalize and save collected training data."""
        if hasattr(self.ai_controller, 'finalize_data_collection'):
            print("📊 Finalizing data collection for LoRA training...")
            stats = self.ai_controller.finalize_data_collection()
            if stats:
                print(f"✅ Data collection complete:")
                print(f"   Session: {stats['session_id']}")
                print(f"   Total entries: {stats['total_entries']}")
                print(f"   Training entries: {stats['training_entries']}")
                print(
                    f"   Personality distribution: {stats['personality_distribution']}")
                print(
                    f"   Action distribution: {stats['action_distribution']}")
            return stats
        return None

    def _is_compound_action(self, action: Action) -> bool:
        """Check if action is a compound action requiring multiple steps."""
        compound_keywords = ['AND', 'PICKUP_EAT', 'PICKUP_DRINK']
        return any(keyword in action.value for keyword in compound_keywords)

    def _execute_compound_action(self, agent: Agent, action: Action, initial_inventory: dict) -> tuple[bool, dict]:
        """Execute a compound action by breaking it into component parts."""
        print(f"  🔄 Executing compound action: {action.value}")

        overall_success = True
        outcome_data = {}

        # Parse action components
        action_parts = self._parse_compound_action(action)

        for i, component_action in enumerate(action_parts):
            print(f"    Step {i+1}: {component_action.value}")

            # Execute each component
            component_success = False

            if component_action.value.startswith("MOVE_"):
                component_success = self._handle_movement(
                    agent, component_action)
            elif component_action == Action.PICKUP:
                component_success = self._handle_pickup(agent)
                # Track pickup outcomes
                if agent.inventory['food'] > initial_inventory['food']:
                    outcome_data['food_collected'] = agent.inventory['food'] - \
                        initial_inventory['food']
                if agent.inventory['water'] > initial_inventory['water']:
                    outcome_data['water_collected'] = agent.inventory['water'] - \
                        initial_inventory['water']
            elif component_action == Action.EAT:
                component_success = self._handle_eat(agent)
                if component_success:
                    outcome_data['food_consumed'] = outcome_data.get(
                        'food_consumed', 0) + 1
            elif component_action == Action.DRINK:
                component_success = self._handle_drink(agent)
                if component_success:
                    outcome_data['water_consumed'] = outcome_data.get(
                        'water_consumed', 0) + 1

            # If any component fails, mark overall as failed
            if not component_success:
                overall_success = False
                print(f"    ❌ Step {i+1} failed: {component_action.value}")
            else:
                print(f"    ✅ Step {i+1} succeeded: {component_action.value}")

        return overall_success, outcome_data

    def _parse_compound_action(self, action: Action) -> list[Action]:
        """Parse compound action into individual component actions."""
        action_str = action.value
        components = []

        # Handle movement + pickup combinations
        if action_str.endswith("_AND_PICKUP"):
            move_part = action_str.replace("_AND_PICKUP", "")
            components.append(Action(move_part))
            components.append(Action.PICKUP)

        # Handle pickup + consume combinations
        elif action_str == "PICKUP_AND_EAT":
            components.append(Action.PICKUP)
            components.append(Action.EAT)
        elif action_str == "PICKUP_AND_DRINK":
            components.append(Action.PICKUP)
            components.append(Action.DRINK)

        # Handle eat + drink combination
        elif action_str == "EAT_AND_DRINK":
            components.append(Action.EAT)
            components.append(Action.DRINK)

        # Handle ultimate efficiency combinations (move + pickup + consume)
        elif action_str.endswith("_PICKUP_EAT"):
            move_part = action_str.replace("_PICKUP_EAT", "")
            components.append(Action(move_part))
            components.append(Action.PICKUP)
            components.append(Action.EAT)
        elif action_str.endswith("_PICKUP_DRINK"):
            move_part = action_str.replace("_PICKUP_DRINK", "")
            components.append(Action(move_part))
            components.append(Action.PICKUP)
            components.append(Action.DRINK)

        return components

    def _execute_action_sequence_from_string(self, agent: Agent, action_sequence: str):
        """Execute a sequence of actions specified as comma-separated string."""
        print(
            f"🎬 Agent {agent.id} executing action sequence: {action_sequence}")

        # Store initial state for data collection
        initial_energy = agent.energy
        initial_inventory = agent.inventory.copy()

        # Parse the sequence
        action_names = [name.strip() for name in action_sequence.split(',')]
        total_success = True
        combined_outcome_data = {}

        # Execute each action in sequence
        for i, action_name in enumerate(action_names):
            print(f"  🎯 Step {i+1}/{len(action_names)}: {action_name}")

            # Convert action name to Action enum
            try:
                individual_action = Action(action_name)
            except ValueError:
                print(f"    ❌ Invalid action: {action_name}")
                total_success = False
                continue

            # Execute the individual action
            step_success = self._execute_individual_step(
                agent, individual_action, initial_inventory, combined_outcome_data)

            if not step_success:
                total_success = False
                print(f"    ❌ Step {i+1} failed: {action_name}")
            else:
                print(f"    ✅ Step {i+1} succeeded: {action_name}")

            # Check if agent died during sequence
            if not agent.alive:
                print(f"    💀 Agent {agent.id} died during step {i+1}")
                break

        # Collect outcome data for training
        if hasattr(self.ai_controller, 'collect_action_outcome'):
            energy_change = agent.energy - initial_energy
            self.ai_controller.collect_action_outcome(
                agent=agent,
                success=total_success,
                energy_change=energy_change,
                sequence_length=len(action_names),
                **combined_outcome_data
            )

        # Add to memory with sequence info
        agent.add_memory(f"SEQUENCE:{action_sequence}", "executed", {
            "turn": self.turn,
            "steps": len(action_names),
            "success": total_success
        })

    def _execute_individual_step(self, agent: Agent, action: Action, initial_inventory: dict, combined_outcome_data: dict) -> bool:
        """Execute a single step within an action sequence."""
        step_success = False

        if action.value.startswith("MOVE_"):
            step_success = self._handle_movement(agent, action)
        elif action == Action.PICKUP:
            step_success = self._handle_pickup(agent)
            # Track pickup outcomes
            if agent.inventory['food'] > initial_inventory['food']:
                combined_outcome_data['food_collected'] = agent.inventory['food'] - \
                    initial_inventory['food']
            if agent.inventory['water'] > initial_inventory['water']:
                combined_outcome_data['water_collected'] = agent.inventory['water'] - \
                    initial_inventory['water']
        elif action == Action.EAT:
            step_success = self._handle_eat(agent)
            if step_success:
                combined_outcome_data['food_consumed'] = combined_outcome_data.get(
                    'food_consumed', 0) + 1
        elif action == Action.DRINK:
            step_success = self._handle_drink(agent)
            if step_success:
                combined_outcome_data['water_consumed'] = combined_outcome_data.get(
                    'water_consumed', 0) + 1
        elif action == Action.SEEK_MATE:
            step_success = self._handle_seek_mate(agent)
        elif action == Action.BREED:
            step_success = self._handle_breeding(agent)
            if step_success:
                combined_outcome_data['breeding_attempt'] = True
        elif action == Action.WAIT:
            step_success = self._handle_wait(agent)
        else:
            print(f"    ⚠️ Unknown action in sequence: {action.value}")
            step_success = False

        return step_success
