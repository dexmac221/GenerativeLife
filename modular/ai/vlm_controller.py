"""
VLM Controller - Handles generative AI decision making for agents using Vision Language Models.
"""

import json
import time
import os
from typing import Dict, Any, Optional, Tuple, List
import requests

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.agent import Agent, Action, Gender
from ..core.world import World, CellType
from ..utils.data_collector import DataCollector
from ..utils.config_loader import config
from .enhanced_spatial_system import SpatialIntelligenceSystem
from .spatial_prompt_enhancer import SpatialPromptEnhancer


class VLMController:
    """
    Controls agent decision making using generative Vision Language Models.

    Features:
    - Dynamic prompt generation based on agent state
    - Context-aware generative decision making
    - Breeding behavior integration
    - Performance monitoring
    """

    def __init__(self, model: str = "gemma2:2b", server: str = "http://localhost:11434",
                 enable_data_collection: bool = True, max_retries: int = 1,
                 api_type: str = "ollama", openai_api_key: Optional[str] = None,
                 compact_prompts: bool = True):
        """
        Initialize VLM Controller with support for multiple AI APIs.

        Args:
            model: Model name (e.g., "gemma2:2b" for Ollama, "gpt-4o" for OpenAI)
            server: Ollama server URL (ignored for OpenAI)
            enable_data_collection: Whether to collect training data
            max_retries: Maximum number of retry attempts for failed JSON parsing
            api_type: "ollama" or "openai"
            openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
            compact_prompts: Use compact prompts to reduce token usage (default: True)
        """
        self.model = model
        self.server = server
        self.max_retries = max_retries
        self.api_type = api_type.lower()
        self.compact_prompts = compact_prompts
        self.inference_count = 0
        self.total_time = 0.0
        self.last_stats_report = 0

        # Initialize OpenAI client if needed
        self.openai_client = None
        if self.api_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai")

            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter")

            self.openai_client = openai.OpenAI(api_key=api_key)
            print(f"🔗 OpenAI API initialized with model: {model}")
        elif self.api_type == "ollama":
            print(f"🦙 Ollama API initialized with model: {model} @ {server}")
        else:
            raise ValueError(
                f"Unsupported API type: {api_type}. Use 'ollama' or 'openai'")

        # Data collection for LoRA training
        self.data_collector = DataCollector() if enable_data_collection else None
        if self.data_collector:
            print("📊 Data collection enabled for LoRA training")

        # Load configuration
        self.survival_config = config.get_survival_config()
        self.ai_config = config.get_ai_behavior_config().get('vlm_controller', {})

        # Load prompt templates
        self.templates = self._load_prompt_templates()
        print("📝 Prompt templates loaded from external files")

        # Initialize enhanced spatial intelligence systems
        self.spatial_system = SpatialIntelligenceSystem()
        self.prompt_enhancer = SpatialPromptEnhancer()
        print("🛡️ Enhanced Spatial Intelligence System ACTIVATED")
        print("🧠 LLM agents will now receive crystal-clear spatial constraints")

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from external files."""
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to reach the project root, then into prompts
        prompts_dir = os.path.join(current_dir, '..', '..', 'prompts')

        templates = {}

        # Template files to load
        template_files = {
            'json_instruction': 'json_instruction.txt',
            'death_warning': 'death_warning.txt',
            'crisis_template': 'crisis_template.txt',
            'cautious_explorer_crisis': 'cautious_explorer_crisis.txt',
            'aggressive_hoarder_crisis': 'aggressive_hoarder_crisis.txt',
            'social_collaborator_crisis': 'social_collaborator_crisis.txt',
            'methodical_strategist_crisis': 'methodical_strategist_crisis.txt',
            'opportunistic_survivor_crisis': 'opportunistic_survivor_crisis.txt',
            'efficient_collector_crisis': 'efficient_collector_crisis.txt',
            'curious_wanderer_crisis': 'curious_wanderer_crisis.txt',
            'protective_guardian_crisis': 'protective_guardian_crisis.txt',
            # Compact templates for token optimization
            'compact_json_instruction': 'compact_json_instruction.txt',
            'compact_cautious_explorer': 'compact_cautious_explorer.txt',
            'compact_aggressive_hoarder': 'compact_aggressive_hoarder.txt',
            'compact_curious_wanderer': 'compact_curious_wanderer.txt',
            'compact_efficient_collector': 'compact_efficient_collector.txt',
            'compact_methodical_strategist': 'compact_methodical_strategist.txt',
            'compact_opportunistic_survivor': 'compact_opportunistic_survivor.txt',
            'compact_social_collaborator': 'compact_social_collaborator.txt',
            'compact_protective_guardian': 'compact_protective_guardian.txt',
            'compact_crisis': 'compact_crisis.txt',
        }

        for template_name, filename in template_files.items():
            filepath = os.path.join(prompts_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    templates[template_name] = f.read().strip()
            except FileNotFoundError:
                print(f"⚠️ Warning: Template file not found: {filepath}")
                templates[template_name] = f"[Template {template_name} not found]"
            except Exception as e:
                print(
                    f"⚠️ Warning: Error loading template {template_name}: {e}")
                templates[template_name] = f"[Error loading {template_name}]"

        # Debug check for specific missing templates
        missing_templates = []
        check_templates = ['json_footer', 'json_header', 'json_enforcement']
        for template_name in check_templates:
            if template_name not in templates:
                missing_templates.append(template_name)

        if missing_templates:
            print(f"Missing templates check:")
            for template in check_templates:
                status = "MISSING" if template in missing_templates else "OK"
                print(f"{template}: {status}")

        return templates

    def _get_available_actions(self, include_compound: bool = True, agent_position: tuple = None, world: 'World' = None) -> list[str]:
        """Enhanced version: Uses spatial intelligence instead of basic filtering."""
        if not agent_position or not world:
            # Fallback to basic actions when no spatial context
            return ['PICKUP', 'EAT', 'DRINK', 'WAIT', 'EXPLORE', 'SEEK_MATE', 'BREED']

        # Create temporary agent object for spatial analysis
        temp_agent = Agent(id="temp", position=agent_position,
                           gender=Gender.MALE, personality="")

        # Get spatial intelligence
        spatial_intel = self.spatial_system.get_movement_intelligence(
            temp_agent, world)
        safe_movements = spatial_intel['safe_movements']

        # Base non-movement actions (always available)
        non_movement_actions = ['PICKUP', 'EAT', 'DRINK',
                                'WAIT', 'EXPLORE', 'SEEK_MATE', 'BREED']

        # Log spatial analysis
        print(f"🔍 SPATIAL ANALYSIS for {agent_position}:")
        print(f"   Safe movements: {safe_movements}")
        if not safe_movements:
            print(f"   ⚠️ Agent is TRAPPED - no movement possible!")

        basic_actions = safe_movements + non_movement_actions

        if include_compound:
            # Only create compounds from SAFE movements
            compound_actions = []

            for move in safe_movements:
                compound_actions.extend([
                    f"{move}_AND_PICKUP",
                    f"{move}_PICKUP_EAT",
                    f"{move}_PICKUP_DRINK"
                ])

            # Non-movement compounds
            compound_actions.extend(
                ['PICKUP_AND_EAT', 'PICKUP_AND_DRINK', 'EAT_AND_DRINK'])

            return basic_actions + compound_actions

        return basic_actions

    def get_action(self, agent: Agent, world: World, observations: Dict[str, Any]):
        """
        Get an action decision from the VLM for the given agent.

        Args:
            agent: The agent making the decision
            world: The world state
            observations: Current observations

        Returns:
            The chosen action
        """
        start_time = time.time()

        # Store initial state for data collection
        initial_energy = agent.energy

        # Prepare context for data collection
        context = self._prepare_context_for_collection(
            agent, observations) if self.data_collector else None

        # Get VLM response with retry mechanism
        try:
            action, response_text = self._try_get_action_with_retry(
                agent, world, observations, max_retries=self.max_retries)

            # Store complete VLM response for visualization
            formatted_speech = self._format_response_for_speech(
                response_text, action)
            agent.set_speech(formatted_speech)

            # Log the decision - handle both Action enum and string sequences
            print(f"🤖 Agent {agent.id} VLM response: '{response_text}'")
            action_display = action.value if hasattr(
                action, 'value') else str(action)
            print(f"✅ Agent {agent.id} chose: {action_display}")

        except Exception as e:
            print(f"❌ VLM error for {agent.id}: {e}")
            action = self._get_fallback_action(agent, observations)
            # Set detailed error speech for debugging (clean text)
            error_speech = f"VLM SYSTEM ERROR\n\nError: {str(e)}\n\nFallback Action: {action.value}\n\nCheck system logs"
            agent.set_speech(self._clean_text_for_display(error_speech))
            print(f"🔄 Agent {agent.id} fallback: {action.value}")

        # Collect decision data for training (before executing action)
        if self.data_collector and context:
            # We'll collect the outcome after action execution in a separate method
            action_value = action.value if hasattr(
                action, 'value') else str(action)
            setattr(agent, '_pending_data_collection', {
                'context': context,
                'decision': action_value,
                'initial_energy': initial_energy,
                'timestamp': time.time()
            })

        # Update performance statistics
        self.inference_count += 1
        self.total_time += time.time() - start_time

        # Report stats periodically
        if self.inference_count % 10 == 0 and self.inference_count != self.last_stats_report:
            avg_time = self.total_time / self.inference_count
            print(
                f"🧠 VLM Stats: {self.inference_count} inferences, avg {avg_time:.2f}s")
            self.last_stats_report = self.inference_count

        return action

    def _generate_prompt(self, agent: Agent, observations: Dict[str, Any], world: 'World' = None) -> str:
        """Enhanced version: Adds spatial intelligence to prompts."""
        # Get comprehensive status awareness
        status_awareness = self._get_comprehensive_status_awareness(agent)

        # Get personality-specific prompt template
        base_prompt = self._get_personality_prompt(
            agent, observations, status_awareness, world)

        # Enhance with spatial intelligence
        if world:
            enhanced_prompt = self.prompt_enhancer.enhance_prompt_with_spatial_intelligence(
                base_prompt, agent, world
            )
            return enhanced_prompt
        else:
            return base_prompt + "\n⚠️ WARNING: No spatial intelligence available - use caution with movement!"

    def _get_json_instruction(self, available_actions: List[str]) -> str:
        """Generate standardized JSON response instruction."""
        actions_list = ", ".join(available_actions)

        if self.compact_prompts:
            # Use compact instruction to save tokens
            return self.templates['compact_json_instruction'].format(
                available_actions=actions_list)
        else:
            # Use full instruction (legacy mode)
            instruction = self.templates['json_instruction'].format(
                available_actions=actions_list)

            # Add sequence explanation with concrete examples
            sequence_instruction = f"""

🚀 MANDATORY SEQUENCES: Always use multi-action sequences when possible!

📋 CONCRETE EXAMPLES:

Scenario 1 - Agent on FOOD cell, low hunger/thirst:
❌ BAD: {{"command": "EAT", "think": "I need food"}}
✅ GOOD: {{"command": "EAT,DRINK", "think": "Eating food and drinking to maintain health efficiently"}}

Scenario 2 - Agent sees food nearby, moderate hunger:
❌ BAD: {{"command": "MOVE_UP", "think": "Moving to food"}}
✅ GOOD: {{"command": "MOVE_UP,PICKUP,EAT,DRINK", "think": "Moving to food, collecting it, eating, then drinking for complete nutrition"}}

Scenario 3 - Male agent sees female agent nearby:
❌ BAD: {{"command": "MOVE_LEFT", "think": "Moving towards agent"}}
✅ GOOD: {{"command": "MOVE_LEFT,REPLICATE", "think": "Moving to female mate and reproducing to expand population"}}

Scenario 4 - Agent with high hunger/thirst on resource cluster:
❌ BAD: {{"command": "PICKUP", "think": "Getting resource"}}
✅ GOOD: {{"command": "PICKUP,EAT,PICKUP,DRINK", "think": "Efficiently collecting multiple resources and consuming immediately"}}

SURVIVAL SEQUENCES (High Priority):
- "EAT,DRINK" = Address hunger and thirst together
- "PICKUP,EAT,DRINK" = Collect and consume resources
- "MOVE_TO_FOOD,PICKUP,EAT,DRINK" = Complete nutrition cycle

REPRODUCTION SEQUENCES (When male+female nearby):
- "MOVE_TO_MATE,REPLICATE" = Find compatible mate and reproduce
- "EAT,DRINK,REPLICATE" = Prepare health then reproduce
- "REPLICATE,EAT,DRINK" = Reproduce then recover

EFFICIENCY SEQUENCES:
- "MOVE_UP,PICKUP,EAT,MOVE_DOWN,PICKUP,DRINK" = Chain multiple resource collection
- "EXPLORE,PICKUP,EAT" = Discover and consume
- "MOVE_RIGHT,PICKUP,DRINK,EAT" = Move, pickup, drink, then eat

⚡ ALWAYS prefer sequences over single actions! Single actions waste turns."""

            return instruction + sequence_instruction

        return instruction + sequence_instruction

    def _get_comprehensive_status_awareness(self, agent: Agent) -> Dict[str, Any]:
        """Generate comprehensive status awareness for the agent."""
        urgency = self._assess_urgency_level(agent)

        # Health status assessment
        health_status = "HEALTHY"
        if urgency >= 4:
            health_status = "CRITICAL"
        elif urgency >= 3:
            health_status = "SEVERE"
        elif urgency >= 2:
            health_status = "WARNING"
        elif urgency >= 1:
            health_status = "CAUTION"

        # Specific condition awareness
        conditions = []
        if agent.energy < 15:
            conditions.append("ENERGY_CRITICAL")
        elif agent.energy < 30:
            conditions.append("ENERGY_LOW")

        if agent.hunger > 95:
            conditions.append("STARVING")
        elif agent.hunger > 85:
            conditions.append("VERY_HUNGRY")
        elif agent.hunger > 70:
            conditions.append("HUNGRY")

        if agent.thirst > 95:
            conditions.append("DEHYDRATING")
        elif agent.thirst > 85:
            conditions.append("VERY_THIRSTY")
        elif agent.thirst > 70:
            conditions.append("THIRSTY")

        # Survival priority assessment
        survival_priority = "EXPLORATION"
        if urgency >= 3:
            survival_priority = "IMMEDIATE_SURVIVAL"
        elif urgency >= 2:
            survival_priority = "URGENT_NEEDS"
        elif urgency >= 1:
            survival_priority = "BASIC_NEEDS"

        # Time to live estimation (rough)
        estimated_turns_left = min(
            (100 - agent.hunger) // 2 if agent.hunger > 50 else 50,
            (100 - agent.thirst) // 2 if agent.thirst > 50 else 50,
            agent.energy // 1 if agent.energy < 50 else 50
        )

        return {
            'urgency_level': urgency,
            'health_status': health_status,
            'conditions': conditions,
            'survival_priority': survival_priority,
            'estimated_turns_left': estimated_turns_left,
            'is_suffering': urgency >= 2,
            'is_critical': urgency >= 3
        }

    def _get_personality_prompt(self, agent: Agent, observations: Dict[str, Any], status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Generate a personality-driven prompt based on agent character."""

        # Get basic info
        nearby_cells = observations.get('nearby_cells', {})
        current_cell = observations.get('current_cell', 'EMPTY')
        recent_memories = agent.get_recent_memories(3)
        recent_actions = [m.get('action', '') for m in recent_memories]

        # Enhanced status info with awareness
        urgency_indicator = ""
        if status_awareness['is_critical']:
            urgency_indicator = "💀 CRITICAL STATE "
        elif status_awareness['is_suffering']:
            urgency_indicator = "⚠️ SUFFERING "

        status = f"{urgency_indicator}Energy: {agent.energy}/100 | Hunger: {agent.hunger}/100 | Thirst: {agent.thirst}/100"

        # Add condition awareness
        if status_awareness['conditions']:
            condition_text = ", ".join(status_awareness['conditions'])
            status += f" | Conditions: {condition_text}"

        # Add survival priority
        status += f" | Priority: {status_awareness['survival_priority']}"

        # Add estimated survival time if critical
        if status_awareness['is_suffering']:
            status += f" | Est. survival: ~{status_awareness['estimated_turns_left']} turns"
        inventory = f"Inventory: {agent.inventory['food']} food, {agent.inventory['water']} water"

        # Enhanced location with world boundary context
        world_width = observations.get('world_width', 'unknown')
        world_height = observations.get('world_height', 'unknown')
        boundary_cells = observations.get('boundary_cells', 0)

        location = f"Position: {agent.position} | Current cell: {current_cell}"
        if isinstance(world_width, int) and isinstance(world_height, int):
            location += f" | World size: {world_width}x{world_height} (valid positions: 0-{world_width-1}, 0-{world_height-1})"
        else:
            location += f" | World size: {world_width}x{world_height}"

        if boundary_cells > 0:
            location += f" | {boundary_cells} boundary cells detected nearby (cannot move there)"
            location += f" | AVOID moving to any cell marked as 'BOUNDARY' or 'OBSTACLE'"

        # Get nearby resources and agents with directions
        nearby_info = self._get_directional_resources(
            agent.position, nearby_cells)

        # Detect if stuck in repetitive behavior
        is_stuck = len(set(recent_actions)) == 1 and len(recent_actions) >= 2

        # Choose personality-specific prompt
        personality = agent.personality.lower()

        # Use compact prompts if enabled (massive token savings)
        if self.compact_prompts:
            return self._generate_compact_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)

        # Legacy verbose prompts
        if "cautious explorer" in personality:
            return self._cautious_explorer_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "aggressive resource hoarder" in personality:
            return self._aggressive_hoarder_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "social collaborator" in personality:
            return self._social_collaborator_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "methodical strategist" in personality:
            return self._methodical_strategist_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "opportunistic survivor" in personality:
            return self._opportunistic_survivor_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "efficient collector" in personality:
            return self._efficient_collector_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "curious wanderer" in personality:
            return self._curious_wanderer_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        elif "protective guardian" in personality:
            return self._protective_guardian_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)
        else:
            # Default fallback
            return self._generic_survivor_prompt(agent, status, inventory, location, nearby_info, is_stuck, status_awareness, world)

    def _get_directional_resources(self, position: Tuple[int, int], nearby_cells: Dict[str, str]) -> Dict[str, Any]:
        """Get resources organized by direction from agent position."""
        x, y = position
        directions = {
            'north': [],
            'south': [],
            'east': [],
            'west': [],
            'northeast': [],
            'northwest': [],
            'southeast': [],
            'southwest': []
        }

        for pos_str, cell_type in nearby_cells.items():
            # Parse position
            pos_clean = pos_str.strip('()')
            try:
                px, py = map(int, pos_clean.split(','))

                # Calculate direction
                dx = px - x
                dy = py - y

                if dx == 0 and dy < 0:
                    directions['north'].append(f"{cell_type} at {pos_str}")
                elif dx == 0 and dy > 0:
                    directions['south'].append(f"{cell_type} at {pos_str}")
                elif dx > 0 and dy == 0:
                    directions['east'].append(f"{cell_type} at {pos_str}")
                elif dx < 0 and dy == 0:
                    directions['west'].append(f"{cell_type} at {pos_str}")
                elif dx > 0 and dy < 0:
                    directions['northeast'].append(f"{cell_type} at {pos_str}")
                elif dx < 0 and dy < 0:
                    directions['northwest'].append(f"{cell_type} at {pos_str}")
                elif dx > 0 and dy > 0:
                    directions['southeast'].append(f"{cell_type} at {pos_str}")
                elif dx < 0 and dy > 0:
                    directions['southwest'].append(f"{cell_type} at {pos_str}")

            except (ValueError, IndexError):
                continue

        return directions

    def _cautious_explorer_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for cautious explorers who prioritize safety."""

        # Enhanced crisis detection with status awareness
        crisis_actions = self._get_survival_actions(agent)

        # Use status awareness for more urgent messaging
        if status_awareness['is_critical'] or crisis_actions:
            urgency_msg = "🔴 LIFE-THREATENING EMERGENCY!" if status_awareness[
                'is_critical'] else "⚠️ CRISIS DETECTED!"
            survival_time = f" Only ~{status_awareness['estimated_turns_left']} turns left!" if status_awareness['is_suffering'] else ""

            if self.compact_prompts:
                return self.templates['compact_crisis'].format(
                    urgency_msg=urgency_msg,
                    status=status,
                    inventory=inventory,
                    location=location,
                    survival_time=survival_time,
                    crisis_actions=crisis_actions or 'Critical health status detected',
                    survival_priority=status_awareness['survival_priority'],
                    crisis_instructions="Find food/water immediately!",
                    json_instruction=self._get_json_instruction(
                        self._get_available_actions(agent_position=agent.position, world=world))
                )
            else:
                # Legacy mode with fallback templates
                json_enforcement = self.templates.get(
                    'json_enforcement', 'Respond with valid JSON only.')
                json_header = self.templates.get(
                    'json_header', '{"action": "action_name", "reasoning": "explanation"}')
                json_footer = self.templates.get(
                    'json_footer', 'End JSON response.')
                crisis_template = self.templates.get(
                    'crisis_template', 'CRISIS: {urgency_msg}\nStatus: {status}\nInventory: {inventory}\nLocation: {location}{survival_time}\nPriority: {survival_priority}\nCrisis Actions: {crisis_actions}\n{crisis_instructions}\n{json_instruction}')

                return json_enforcement + "\n\n" + json_header + "\n\n" + crisis_template.format(
                    personality_type="CAUTIOUS EXPLORER",
                    urgency_msg=urgency_msg,
                    status=status,
                    inventory=inventory,
                    location=location,
                    survival_time=survival_time,
                    crisis_actions=crisis_actions or 'Critical health status detected',
                    survival_priority=status_awareness['survival_priority'],
                    crisis_instructions=self.templates.get(
                        'cautious_explorer_crisis', 'Take immediate survival action!'),
                    json_instruction=self._get_json_instruction(
                        self._get_available_actions(agent_position=agent.position, world=world))
                ) + "\n\n" + json_footer

        # Get movement variety guidance
        recent_memories = agent.get_recent_memories(4)
        recent_moves = [m.get('action', '') for m in recent_memories if m.get(
            'action', '').startswith('MOVE_')]
        movement_analysis = self._analyze_movement_pattern(recent_moves)

        exploration_guidance = ""
        if is_stuck:
            exploration_guidance = "⚠️ STUCK! Must try a completely different direction for safety!"
        elif movement_analysis:
            exploration_guidance = f"🔄 {movement_analysis} - vary your cautious exploration!"

        # Get specific directional resources
        direction_resources = self._get_specific_directional_guidance(nearby)
        safe_moves = self._get_safe_directions(nearby)

        if self.compact_prompts:
            return self.templates['compact_cautious_explorer'].format(
                status=status,
                inventory=inventory,
                location=location,
                direction_resources=direction_resources,
                safe_moves=safe_moves,
                exploration_guidance=exploration_guidance,
                json_instruction=self._get_json_instruction(
                    self._get_available_actions(agent_position=agent.position, world=world))
            )
        else:
            # Legacy mode with fallback templates
            json_enforcement = self.templates.get(
                'json_enforcement', 'Respond with valid JSON only.')
            json_header = self.templates.get(
                'json_header', '{"action": "action_name", "reasoning": "explanation"}')
            json_footer = self.templates.get(
                'json_footer', 'End JSON response.')
            cautious_explorer = self.templates.get(
                'cautious_explorer', 'CAUTIOUS EXPLORER\nStatus: {status}\nInventory: {inventory}\nLocation: {location}\nDirection Resources: {direction_resources}\nSafe Moves: {safe_moves}\n{exploration_guidance}\n{json_instruction}')

            return json_enforcement + "\n\n" + json_header + "\n\n" + cautious_explorer.format(
                status=status,
                inventory=inventory,
                location=location,
                direction_resources=direction_resources,
                safe_moves=safe_moves,
                exploration_guidance=exploration_guidance,
                json_instruction=self._get_json_instruction(
                    self._get_available_actions(agent_position=agent.position, world=world))
            ) + "\n\n" + json_footer

    def _aggressive_hoarder_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for aggressive resource hoarders."""

        # Enhanced crisis detection with status awareness
        crisis_actions = self._get_survival_actions(agent)
        survival_guidance = self._get_survival_guidance(
            agent, status_awareness)

        # Aggressive types should be even more decisive in crisis
        if status_awareness['is_critical'] or crisis_actions:
            urgency_msg = "💀 DEATH IMMINENT!" if status_awareness[
                'is_critical'] else "🔥 CRITICAL RESOURCE SHORTAGE!"
            survival_time = f" DYING IN ~{status_awareness['estimated_turns_left']} TURNS!" if status_awareness['is_suffering'] else ""

            prompt_content = f"""⚔️ AGGRESSIVE HOARDER - {urgency_msg}
{status} | {inventory}
{location}

{urgency_msg}{survival_time}
CRISIS: {crisis_actions or 'Resources critically low'}
Priority: {status_awareness['survival_priority']}

{survival_guidance}

AGGRESSIVE ACTION REQUIRED NOW!
- Consume resources IMMEDIATELY if you have them
- RUSH to nearest resource if inventory empty
- NO TIME for careful planning - SURVIVE FIRST!

{self._get_json_instruction(self._get_available_actions(agent_position=agent.position, world=world))}"""

            return self._wrap_with_json_enforcement(prompt_content, use_death_warning=self._needs_death_warning())

        # Find the best resource targets in all directions
        resource_targets = []
        direction_priorities = []

        for direction, items in nearby.items():
            resources = [
                item for item in items if 'FOOD' in item or 'WATER' in item]
            if resources:
                count = len(resources)
                direction_priorities.append((direction, count, resources[0]))

        # Sort by resource count for aggressive targeting
        direction_priorities.sort(key=lambda x: x[1], reverse=True)

        # Movement pattern analysis
        recent_memories = agent.get_recent_memories(4)
        recent_moves = [m.get('action', '') for m in recent_memories if m.get(
            'action', '').startswith('MOVE_')]
        movement_analysis = self._analyze_movement_pattern(recent_moves)

        hoarding_guidance = ""
        if agent.inventory['food'] < 3 or agent.inventory['water'] < 3:
            hoarding_guidance = "🎯 STOCKPILE CRITICAL! Must hoard more resources aggressively!"
        else:
            hoarding_guidance = "💰 Good stockpile - but dominance requires MORE!"

        if is_stuck or movement_analysis:
            hoarding_guidance += f" ⚡ {movement_analysis or 'STUCK!'} Break through with different tactics!"

        # Get specific directional guidance
        direction_resources = self._get_specific_directional_guidance(nearby)

        target_guidance = "🏹 TARGET PRIORITIES:\n"
        if direction_priorities:
            for i, (direction, count, sample) in enumerate(direction_priorities[:3]):
                action = self._direction_to_action(direction)
                target_guidance += f"  {i+1}. {action}: {count} resources ({sample})\n"
        else:
            target_guidance += "  No immediate targets - expand search radius!\n"

        # Use the external template file with formatting
        template_content = self.templates['aggressive_hoarder'].format(
            status=status,
            inventory=inventory,
            location=location,
            target_guidance=target_guidance,
            direction_resources=direction_resources,
            hoarding_guidance=hoarding_guidance,
            json_instruction=self._get_json_instruction(
                self._get_available_actions(agent_position=agent.position, world=world))
        )

        return self._wrap_with_json_enforcement(template_content)

    def _social_collaborator_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for social agents who seek others."""

        # Enhanced crisis detection with status awareness
        crisis_actions = self._get_survival_actions(agent)

        # Social types should seek help when in crisis
        if status_awareness['is_critical'] or crisis_actions:
            urgency_msg = "🆘 NEED IMMEDIATE HELP!" if status_awareness[
                'is_critical'] else "⚠️ SURVIVAL CRISIS!"
            survival_time = f" Only ~{status_awareness['estimated_turns_left']} turns to live!" if status_awareness['is_suffering'] else ""

            return self._wrap_with_json_enforcement(f"""🤝 SOCIAL COLLABORATOR - {urgency_msg}
{status} | {inventory}
{location}

{urgency_msg}{survival_time}
CRISIS: {crisis_actions or 'Critical health status'}
Priority: {status_awareness['survival_priority']}

SEEK HELP OR SELF-RESCUE NOW!
- If you have resources: USE THEM IMMEDIATELY
- If near other agents: move closer for potential sharing
- If alone: prioritize immediate survival actions
- Your survival matters for the group!

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, SEEK_MATE, BREED, WAIT""")

        # Look for agents and analyze movement patterns
        agent_locations = []
        for direction, items in nearby.items():
            for item in items:
                if 'AGENT' in item:
                    agent_locations.append(f"{direction}: {item}")

        # Analyze recent movement for variety
        recent_memories = agent.get_recent_memories(4)
        recent_moves = [m.get('action', '') for m in recent_memories if m.get(
            'action', '').startswith('MOVE_')]
        movement_analysis = self._analyze_movement_pattern(recent_moves)

        social_guidance = ""
        if agent_locations:
            social_guidance = f"👥 COMPANIONS DETECTED: {agent_locations[:2]} - Approach them NOW!"
            if agent.is_mature and agent.can_breed():
                social_guidance += " Perfect for breeding opportunities!"
        else:
            social_guidance = "🔍 No companions visible - must search in ALL directions systematically!"

        if is_stuck or movement_analysis:
            social_guidance += f" 🚪 {movement_analysis or 'Stuck in pattern'} - try opposite directions!"

        # Get specific directional guidance for social search
        direction_resources = self._get_specific_directional_guidance(nearby)

        return self._wrap_with_json_enforcement(f"""SOCIAL COLLABORATOR

{status} | {inventory}
{location}

Your social nature seeks connection and cooperation above all!

DIRECTIONAL SCAN:
{direction_resources}

{social_guidance}

SOCIAL SEARCH STRATEGY: Search systematically in different directions each turn.
- North/South: Check for distant agents
- East/West: Explore horizontal areas
- Do not repeat the same direction - vary your social search pattern!

Values: Companionship over solitude, cooperation over competition.
Remember: Other agents could be ANYWHERE - explore all directions!

Choose ONE action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, SEEK_MATE, BREED, WAIT""")

    def _methodical_strategist_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for methodical strategists who plan ahead."""

        # Enhanced crisis detection with status awareness
        crisis_actions = self._get_survival_actions(agent)

        # Methodical types should have systematic crisis response
        if status_awareness['is_critical'] or crisis_actions:
            urgency_msg = "🚨 SYSTEMATIC EMERGENCY PROTOCOL!" if status_awareness[
                'is_critical'] else "⚠️ CRISIS ANALYSIS!"
            survival_time = f" Calculated survival: ~{status_awareness['estimated_turns_left']} turns!" if status_awareness['is_suffering'] else ""

            return self._wrap_with_json_enforcement(f"""🧠 METHODICAL STRATEGIST - {urgency_msg}
{status} | {inventory}
{location}

{urgency_msg}{survival_time}
CRISIS: {crisis_actions or 'Health parameters critical'}
Priority: {status_awareness['survival_priority']}

SYSTEMATIC EMERGENCY RESPONSE:
1. Assess available resources(inventory check)
2. If resources available: CONSUME IMMEDIATELY
3. If no resources: calculate nearest resource location
4. Execute optimal survival action with PRECISION

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

        # Analyze resource distribution
        resource_analysis = self._analyze_resource_layout(nearby)
        efficiency_plan = ""

        if agent.inventory['food'] == 0 and any('FOOD' in str(items) for items in nearby.values()):
            efficiency_plan = "PHASE 1: Secure food sources systematically"
        elif agent.inventory['water'] == 0 and any('WATER' in str(items) for items in nearby.values()):
            efficiency_plan = "PHASE 1: Secure water sources systematically"
        elif agent.inventory['food'] < 2 or agent.inventory['water'] < 2:
            efficiency_plan = "PHASE 2: Build strategic resource reserves"
        else:
            efficiency_plan = "PHASE 3: Explore for long-term opportunities"

        if is_stuck:
            efficiency_plan += " | ADAPT: Current approach ineffective, recalculate route!"

        # Use the external template file with formatting
        template_content = self.templates['methodical_strategist'].format(
            status=status,
            inventory=inventory,
            location=location,
            resource_analysis=resource_analysis,
            efficiency_plan=efficiency_plan,
            json_instruction=self._get_json_instruction(
                self._get_available_actions(agent_position=agent.position, world=world))
        )

        return self._wrap_with_json_enforcement(template_content)

    def _opportunistic_survivor_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for opportunistic survivors who adapt quickly."""

        crisis_actions = self._get_survival_actions(agent)
        if crisis_actions:
            return self._wrap_with_json_enforcement(f"""⚡ OPPORTUNISTIC SURVIVOR - CRISIS MODE
{status} | {inventory}
{location}

CRISIS: {crisis_actions}
Adapt fast - survival is everything!

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

        # Look for immediate opportunities
        immediate_opportunities = []
        for direction, items in nearby.items():
            if any('FOOD' in item or 'WATER' in item for item in items):
                immediate_opportunities.append(f"Quick grab {direction}")

        adaptation_mode = ""
        if is_stuck:
            adaptation_mode = "🔄 ADAPTATION MODE: Current strategy failed - pivoting immediately!"
        elif agent.hunger > 50 or agent.thirst > 50:
            adaptation_mode = "🎯 OPPORTUNITY MODE: Basic needs rising - seize resources!"
        elif len(immediate_opportunities) > 2:
            adaptation_mode = "💎 JACKPOT MODE: Multiple opportunities - grab the best!"
        else:
            adaptation_mode = "🔍 SCANNING MODE: Looking for new opportunities..."

        # Use the external template file with formatting
        template_content = self.templates['opportunistic_survivor'].format(
            status=status,
            inventory=inventory,
            location=location,
            immediate_opportunities=immediate_opportunities[:3],
            adaptation_mode=adaptation_mode,
            json_instruction=self._get_json_instruction(
                self._get_available_actions(agent_position=agent.position, world=world))
        )

        return self._wrap_with_json_enforcement(template_content)

    def _efficient_collector_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for efficient collectors who minimize energy waste."""

        crisis_actions = self._get_survival_actions(agent)
        if crisis_actions:
            return self._wrap_with_json_enforcement(f"""⚙️ EFFICIENT COLLECTOR - EMERGENCY MODE
{status} | {inventory}
{location}

CRISIS: {crisis_actions}
Optimize energy usage while handling crisis!

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

        # Calculate most efficient moves
        efficiency_analysis = self._calculate_movement_efficiency(
            agent.position, nearby)
        energy_guidance = ""

        if agent.energy < 50:
            energy_guidance = "⚠️ ENERGY CONSERVATION: Move only when necessary!"
        elif agent.energy > 80:
            energy_guidance = "✅ ENERGY SURPLUS: Can afford longer efficient routes"
        else:
            energy_guidance = "⚖️ BALANCED ENERGY: Optimize for efficiency"

        if is_stuck:
            energy_guidance += " 🔧 EFFICIENCY LOSS: Recalculate optimal path!"

        return self._wrap_with_json_enforcement(f"""⚙️ EFFICIENT COLLECTOR
{status} | {inventory}
{location}

Your efficiency-focused mind optimizes every move!

EFFICIENCY ANALYSIS: {efficiency_analysis}
{energy_guidance}

Strategy: Minimize energy waste, maximize resource gathering per move.
Goal: Perfect efficiency - no wasted actions or movements.

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

    def _curious_wanderer_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for curious wanderers who explore extensively."""

        crisis_actions = self._get_survival_actions(agent)
        if crisis_actions:
            return self._wrap_with_json_enforcement(f"""🌟 CURIOUS WANDERER - SURVIVAL PRIORITY
{status} | {inventory}
{location}

CRISIS: {crisis_actions}
Survive first, then continue exploring!

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

        # Analyze recent movement for exploration variety
        recent_memories = agent.get_recent_memories(5)
        recent_moves = [m.get('action', '') for m in recent_memories if m.get(
            'action', '').startswith('MOVE_')]
        movement_analysis = self._analyze_movement_pattern(recent_moves)

        # Get unexplored directions
        unexplored_directions = self._get_unexplored_directions(recent_moves)

        # Encourage exploration variety
        exploration_options = []
        for direction in ['north', 'south', 'east', 'west']:
            if direction in nearby and nearby[direction]:
                exploration_options.append(
                    f"{direction} ({len(nearby[direction])} things)")

        wanderlust_guidance = ""
        if is_stuck or movement_analysis:
            wanderlust_guidance = f"🗺️ {movement_analysis or 'EXPLORATION BLOCKED!'} Try completely different direction!"
        elif unexplored_directions:
            wanderlust_guidance = f"🧭 UNEXPLORED: {', '.join(unexplored_directions)} - your curiosity calls you there!"
        elif len(exploration_options) > 1:
            wanderlust_guidance = f"🔍 MULTIPLE PATHS: {exploration_options} - which calls to your curious spirit?"
        else:
            wanderlust_guidance = "🌟 LIMITED VISIBILITY: Venture into the unknown!"

        # Encourage movement variety based on recent patterns
        direction_variety_tip = ""
        if recent_moves:
            recent_directions = [move.split('_')[1].lower()
                                 for move in recent_moves if '_' in move]
            if len(set(recent_directions)) <= 1 and recent_directions:
                opposite_dir = self._get_opposite_direction(
                    recent_directions[0])
                direction_variety_tip = f"🔄 You've been exploring {recent_directions[0]} - try {opposite_dir} for true wanderlust!"

        # Get specific directional guidance
        direction_resources = self._get_specific_directional_guidance(nearby)

        return self._wrap_with_json_enforcement(f"""🌟 CURIOUS WANDERER
{status} | {inventory}
{location}

Your wandering spirit seeks new discoveries and experiences!

DIRECTIONAL SCAN:
{direction_resources}

{wanderlust_guidance}
{direction_variety_tip}

WANDERER'S CODE:
- True curiosity explores ALL directions, not just one!
- Repetition is the enemy of discovery
- Each direction holds different mysteries
- The best adventures come from unexpected paths

Strategy: Explore varied paths, discover new areas, satisfy boundless curiosity.
Philosophy: "The journey matters as much as the destination!"

Choose ONE action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, EXPLORE, WAIT""")

    def _protective_guardian_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Prompt for protective guardians who help others."""

        crisis_actions = self._get_survival_actions(agent)
        if crisis_actions:
            return self._wrap_with_json_enforcement(f"""🛡️ PROTECTIVE GUARDIAN - PERSONAL CRISIS
{status} | {inventory}
{location}

CRISIS: {crisis_actions}
Secure yourself first to better protect others!

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

        # Look for agents to protect
        nearby_agents = []
        for direction, items in nearby.items():
            for item in items:
                if 'AGENT' in item:
                    nearby_agents.append(f"{direction}: {item}")

        protection_mode = ""
        if nearby_agents:
            protection_mode = f"👥 GUARDIANSHIP MODE: Agents detected {nearby_agents[:2]} - move to assist!"
            if agent.inventory['food'] > 1 or agent.inventory['water'] > 1:
                protection_mode += " Consider sharing resources!"
        else:
            protection_mode = "🔍 PATROL MODE: Searching for agents who need protection..."

        if is_stuck:
            protection_mode += " 🚨 BLOCKED PATH: Find alternate route to reach those in need!"

        return self._wrap_with_json_enforcement(f"""🛡️ PROTECTIVE GUARDIAN
{status} | {inventory}
{location}

Your protective nature seeks to help and defend others!

{protection_mode}

Strategy: Find other agents, share resources, provide protection and guidance.
Values: Community over self, protection over personal gain.

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, SEEK_MATE, WAIT""")

    def _generic_survivor_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Generic fallback prompt."""
        crisis_actions = self._get_survival_actions(agent)
        if crisis_actions:
            return self._wrap_with_json_enforcement(f"""🔄 SURVIVOR - EMERGENCY
{status} | {inventory}
{location}

CRISIS: {crisis_actions}

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

        return self._wrap_with_json_enforcement(f"""🔄 ADAPTIVE SURVIVOR
{status} | {inventory}
{location}

Survive, explore, and thrive in this world!

Choose: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, EAT, DRINK, WAIT""")

    def _get_survival_actions(self, agent: Agent) -> str:
        """Get critical survival actions if agent is in danger."""
        actions = []
        urgency_level = self._assess_urgency_level(agent)

        # Critical energy - immediate action needed
        if agent.energy < 15:
            actions.append("⚠️ CRITICAL ENERGY! Death imminent!")
        elif agent.energy < 30:
            actions.append("🔋 Low energy - conserve movements!")

        # Hunger assessment with urgency
        hunger_critical = self.survival_config.get(
            'hunger', {}).get('critical', 85)
        hunger_urgent = self.survival_config.get(
            'hunger', {}).get('urgent', 60)
        hunger_high = self.survival_config.get(
            'hunger', {}).get('high_priority', 40)
        hunger_medium = self.survival_config.get(
            'hunger', {}).get('medium_priority', 25)

        if agent.hunger > hunger_critical:
            if agent.inventory['food'] > 0:
                actions.append("🍎 EAT NOW OR DIE!")
            else:
                actions.append("🆘 FIND FOOD IMMEDIATELY - STARVING!")
        elif agent.hunger > hunger_urgent:
            if agent.inventory['food'] > 0:
                actions.append("🍎 EAT food immediately!")
            else:
                actions.append("🔍 Find food urgently!")
        elif agent.hunger > hunger_high:
            if agent.inventory['food'] > 0:
                actions.append("🍽️ EAT food when possible!")
            else:
                actions.append("🍽️ Seek food soon")
        elif agent.hunger > hunger_medium:
            if agent.inventory['food'] > 0:
                actions.append("🍎 Consider eating to maintain energy")

        # Thirst assessment with urgency
        thirst_critical = self.survival_config.get(
            'thirst', {}).get('critical', 85)
        thirst_urgent = self.survival_config.get(
            'thirst', {}).get('urgent', 60)
        thirst_high = self.survival_config.get(
            'thirst', {}).get('high_priority', 40)
        thirst_medium = self.survival_config.get(
            'thirst', {}).get('medium_priority', 25)

        if agent.thirst > thirst_critical:
            if agent.inventory['water'] > 0:
                actions.append("💧 DRINK NOW OR DIE!")
            else:
                actions.append("🆘 FIND WATER IMMEDIATELY - DEHYDRATING!")
        elif agent.thirst > thirst_urgent:
            if agent.inventory['water'] > 0:
                actions.append("💧 DRINK water immediately!")
            else:
                actions.append("🔍 Find water urgently!")
        elif agent.thirst > thirst_high:
            if agent.inventory['water'] > 0:
                actions.append("🥤 DRINK water when possible!")
            else:
                actions.append("🥤 Seek water soon")
        elif agent.thirst > thirst_medium:
            if agent.inventory['water'] > 0:
                actions.append("💧 Consider drinking to maintain hydration")

        # Compound suffering - multiple critical needs
        critical_count = sum([
            agent.energy < 20,
            agent.hunger > 90,
            agent.thirst > 90
        ])

        if critical_count >= 2:
            actions.insert(0, "💀 MULTIPLE CRITICAL NEEDS - DEATH RISK HIGH!")
        elif urgency_level >= 3:
            actions.insert(0, "⚡ EMERGENCY STATE - ACT FAST!")

        return " ".join(actions)

    def _assess_urgency_level(self, agent: Agent) -> int:
        """Assess overall urgency level (0-4) based on agent status."""
        urgency = 0

        # Energy urgency
        if agent.energy < 10:
            urgency += 2
        elif agent.energy < 25:
            urgency += 1

        # Hunger urgency
        if agent.hunger > 95:
            urgency += 2
        elif agent.hunger > 80:
            urgency += 1

        # Thirst urgency
        if agent.thirst > 95:
            urgency += 2
        elif agent.thirst > 80:
            urgency += 1

        return min(urgency, 4)  # Cap at 4

    def _get_safe_directions(self, nearby: Dict) -> str:
        """Get directions that seem safe (no obstacles)."""
        safe = []
        for direction, items in nearby.items():
            if not any('OBSTACLE' in item for item in items):
                safe.append(direction)
        return ", ".join(safe[:4]) if safe else "Limited safe paths"

    def _get_resource_directions(self, nearby: Dict) -> str:
        """Get directions with resources."""
        resources = []
        for direction, items in nearby.items():
            resource_items = [
                item for item in items if 'FOOD' in item or 'WATER' in item]
            if resource_items:
                resources.append(f"{direction}: {resource_items[0]}")
        return ", ".join(resources[:3]) if resources else "No visible resources"

    def _analyze_resource_layout(self, nearby: Dict) -> str:
        """Analyze resource distribution for strategic planning."""
        analysis = []
        total_resources = 0

        for direction, items in nearby.items():
            resources_here = len(
                [item for item in items if 'FOOD' in item or 'WATER' in item])
            if resources_here > 0:
                total_resources += resources_here
                analysis.append(f"{direction}({resources_here})")

        if total_resources == 0:
            return "No immediate resources detected"
        elif total_resources < 3:
            return f"Sparse resources: {', '.join(analysis)}"
        else:
            return f"Rich area: {', '.join(analysis)}"

    def _calculate_movement_efficiency(self, position: Tuple[int, int], nearby: Dict) -> str:
        """Calculate most efficient movement options."""
        x, y = position
        efficiency_scores = {}

        # Simple efficiency scoring based on resource density per direction
        for direction, items in nearby.items():
            resource_count = len(
                [item for item in items if 'FOOD' in item or 'WATER' in item])
            obstacle_count = len(
                [item for item in items if 'OBSTACLE' in item])

            # Higher score = more efficient
            score = resource_count - (obstacle_count * 0.5)
            if score > 0:
                efficiency_scores[direction] = score

        if not efficiency_scores:
            return "No clear efficiency advantage detected"

        best_direction = max(efficiency_scores.items(), key=lambda x: x[1])
        return f"Most efficient: {best_direction[0]} (score: {best_direction[1]:.1f})"

    def _analyze_movement_pattern(self, recent_moves: list) -> str:
        """Analyze recent movement patterns for repetitive behavior."""
        if not recent_moves:
            return ""

        if len(recent_moves) >= 3:
            # Check for exact repetition
            if len(set(recent_moves[-3:])) == 1:
                repeated_action = recent_moves[-1]
                direction = repeated_action.split('_')[1].lower(
                ) if '_' in repeated_action else repeated_action
                return f"WARNING: Repeated {direction} movement detected!"

            # Check for simple back-and-forth
            if len(recent_moves) >= 4:
                if recent_moves[-2] == recent_moves[-4] and recent_moves[-1] == recent_moves[-3]:
                    return "WARNING: Back-and-forth movement pattern detected!"

        # Check for directional bias (too much of one direction)
        if len(recent_moves) >= 3:
            direction_counts = {}
            for move in recent_moves[-4:]:  # Look at last 4 moves
                direction = move.split('_')[1].lower() if '_' in move else move
                direction_counts[direction] = direction_counts.get(
                    direction, 0) + 1

            max_direction = max(direction_counts.items(
            ), key=lambda x: x[1]) if direction_counts else None
            if max_direction and max_direction[1] >= 3:
                return f"WARNING: Too much {max_direction[0]} movement - try other directions!"

        return ""

    def _get_specific_directional_guidance(self, nearby: Dict) -> str:
        """Get specific guidance for each direction with resources and obstacles."""
        direction_map = {
            'north': 'MOVE_UP',
            'south': 'MOVE_DOWN',
            'east': 'MOVE_RIGHT',
            'west': 'MOVE_LEFT'
        }

        guidance = []
        for direction, action in direction_map.items():
            items = nearby.get(direction, [])
            if not items:
                guidance.append(f"• {action}: Unknown area")
                continue

            resources = [
                item for item in items if 'FOOD' in item or 'WATER' in item]
            obstacles = [item for item in items if 'OBSTACLE' in item]
            agents = [item for item in items if 'AGENT' in item]

            desc = []
            if resources:
                desc.append(f"{len(resources)} resources")
            if agents:
                desc.append(f"{len(agents)} agents")
            if obstacles:
                desc.append(f"{len(obstacles)} obstacles")
            if not desc:
                desc.append("empty space")

            guidance.append(f"• {action}: {', '.join(desc)}")

        return '\n'.join(guidance)

    def _get_decision_guidance(self, agent: Agent, observations: Dict[str, Any]) -> str:
        """Legacy guidance method - now handled by personality prompts."""
        # This method is now largely replaced by personality-specific prompts
        # but kept for backwards compatibility
        return ""

    def _query_vlm(self, prompt: str, use_structured_output: bool = True) -> str:
        """Query the VLM with the given prompt, supporting both Ollama and OpenAI APIs."""
        if self.api_type == "openai":
            return self._query_openai(prompt, use_structured_output)
        else:
            return self._query_ollama(prompt, use_structured_output)

    def _query_ollama(self, prompt: str, use_structured_output: bool = True) -> str:
        """Query Ollama API with the given prompt."""
        url = f"{self.server}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }

        # Add structured output format if enabled
        if use_structured_output:
            payload["format"] = self._get_json_schema()
            # Reduce temperature for more consistent JSON output
            payload["options"]["temperature"] = 0.3

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            raw_response = result.get('response', '')

            # Check if we got a malformed response (common Ollama issue)
            malformed_patterns = [
                '', ' (open brace)\n2', '(open brace)', '{', '}',
                '\n2', '2', ' 2', '(', ')', 'open brace', 'brace'
            ]
            if raw_response.strip() in malformed_patterns or len(raw_response.strip()) < 10:
                print(
                    f"🐛 DEBUG - Malformed Ollama response detected: '{raw_response}'")
                if use_structured_output:
                    print("🔄 Retrying without structured output...")
                    return self._query_vlm(prompt, use_structured_output=False)
                else:
                    raise Exception(
                        f"Ollama returned malformed response: '{raw_response}'")

            # Debug: Log raw response for troubleshooting if it's not valid JSON
            if not self._is_valid_json_response(raw_response.strip()):
                print(f"🐛 DEBUG - Invalid JSON response: '{raw_response}'")
                print(f"🐛 DEBUG - Raw response repr: {repr(raw_response)}")
                # If structured output failed, try without it
                if use_structured_output:
                    print("🔄 Retrying without structured output...")
                    return self._query_vlm(prompt, use_structured_output=False)

            return raw_response.strip()

        except requests.exceptions.RequestException as e:
            print(f"🐛 DEBUG - Request error: {e}")
            raise Exception(f"VLM request failed: {e}")
        except Exception as e:
            print(f"🐛 DEBUG - Unexpected error in _query_ollama: {e}")
            raise

    def _query_openai(self, prompt: str, use_structured_output: bool = True) -> str:
        """Query OpenAI API with the given prompt."""
        try:
            messages = [{"role": "user", "content": prompt}]

            # Configure parameters based on structured output requirement
            if use_structured_output:
                # For structured output, we append JSON format instruction with example
                json_instruction = "\n\nRespond ONLY with valid JSON in this exact format:\n" + \
                    '{\n  "command": "MOVE_UP",\n  "think": "explanation of reasoning"\n}'
                messages[0]["content"] += json_instruction

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent JSON
                    max_tokens=500,
                    # Force JSON output for GPT-4o and newer
                    response_format={"type": "json_object"}
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )

            raw_response = response.choices[0].message.content.strip()

            # Validate JSON response if structured output was requested
            if use_structured_output and not self._is_valid_json_response(raw_response):
                print(
                    f"🐛 DEBUG - OpenAI returned invalid JSON: '{raw_response}'")
                # Retry without structured formatting
                return self._query_openai(prompt, use_structured_output=False)

            return raw_response

        except Exception as e:
            print(f"🐛 DEBUG - OpenAI API error: {e}")
            # If structured output failed, try without it
            if use_structured_output:
                print("🔄 Retrying without structured output...")
                return self._query_openai(prompt, use_structured_output=False)
            raise Exception(f"OpenAI request failed: {e}")

    def _get_json_schema(self) -> dict:
        """Get the JSON schema for structured outputs."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Action command: single action (e.g. 'EAT') or comma-separated sequence (e.g. 'MOVE_UP,PICKUP,EAT')"
                },
                "think": {
                    "type": "string",
                    "description": "The reasoning behind the chosen action or action sequence"
                }
            },
            "required": ["command", "think"],
            "additionalProperties": False
        }

    def _parse_action(self, response: str, agent_position=None, world=None):
        """Enhanced version: Absolute validation before returning action."""
        import json
        import re

        # First, try to parse as direct JSON (structured outputs)
        try:
            parsed = json.loads(response.strip())
            if 'command' in parsed:
                # Ensure command is a string before calling upper()
                raw_command = parsed['command']
                if not isinstance(raw_command, str):
                    raise ValueError(
                        f"Expected string for 'command', got {type(raw_command)}: {raw_command}")
                command = raw_command.upper().strip()

                # Check if this is an action sequence (contains comma)
                if ',' in command:
                    # Validate sequence components and check boundaries
                    sequence_parts = [part.strip()
                                      for part in command.split(',')]
                    valid_sequence = True

                    for part in sequence_parts:
                        if not self._is_valid_action_name(part):
                            valid_sequence = False
                            break
                        # Enhanced validation using spatial system
                        if agent_position and world:
                            temp_agent = Agent(
                                id="temp", position=agent_position, gender=Gender.MALE, personality="")
                            is_valid, reason = self.spatial_system.validate_action_absolutely(
                                part, temp_agent, world)
                            if not is_valid:
                                print(
                                    f"🚨 BLOCKED INVALID ACTION in sequence: {part}")
                                print(f"   Reason: {reason}")
                                valid_sequence = False
                                break

                    if valid_sequence:
                        return command  # Return the comma-separated string
                    else:
                        print(
                            f"⚠️ Invalid or boundary-violating action sequence: {command}")
                        # Get emergency fallback using spatial system
                        if agent_position and world:
                            temp_agent = Agent(
                                id="temp", position=agent_position, gender=Gender.MALE, personality="")
                            fallback_action = self.spatial_system.get_emergency_fallback(
                                temp_agent, world)
                            print(
                                f"🔄 Using spatial emergency fallback: {fallback_action}")
                            try:
                                return Action(fallback_action)
                            except ValueError:
                                return Action.WAIT
                        return self._get_boundary_safe_fallback(agent_position, world)

                # Enhanced single command validation using spatial system
                if agent_position and world:
                    temp_agent = Agent(
                        id="temp", position=agent_position, gender=Gender.MALE, personality="")
                    is_valid, reason = self.spatial_system.validate_action_absolutely(
                        command, temp_agent, world)
                    if not is_valid:
                        print(f"🚨 BLOCKED INVALID ACTION: {command}")
                        print(f"   Reason: {reason}")

                        # Get emergency fallback
                        fallback_action = self.spatial_system.get_emergency_fallback(
                            temp_agent, world)
                        print(
                            f"🔄 Using spatial emergency fallback: {fallback_action}")
                        try:
                            return Action(fallback_action)
                        except ValueError:
                            return Action.WAIT

                # Try to match single command to a valid Action
                for action in Action:
                    if action.value == command:
                        return action

                # Try partial matches for common variations
                action_mappings = {
                    'UP': Action.MOVE_UP,
                    'DOWN': Action.MOVE_DOWN,
                    'LEFT': Action.MOVE_LEFT,
                    'RIGHT': Action.MOVE_RIGHT,
                    'PICK': Action.PICKUP,
                    'PICKUP': Action.PICKUP,
                    'EAT': Action.EAT,
                    'DRINK': Action.DRINK,
                    'WAIT': Action.WAIT,
                    'REST': Action.WAIT,
                    'EXPLORE': Action.EXPLORE,
                    'MATE': Action.SEEK_MATE,
                    'BREED': Action.BREED
                }

                for keyword, action in action_mappings.items():
                    if keyword in command:
                        # Enhanced validation using spatial system
                        if agent_position and world:
                            temp_agent = Agent(
                                id="temp", position=agent_position, gender=Gender.MALE, personality="")
                            action_str = action.value if hasattr(
                                action, 'value') else str(action)
                            is_valid, reason = self.spatial_system.validate_action_absolutely(
                                action_str, temp_agent, world)
                            if is_valid:
                                return action
                        else:
                            return action

        except json.JSONDecodeError as e:
            # Debug: Show the exact JSON error
            print(f"🐛 DEBUG - JSON decode error: {e}")
            print(
                f"🐛 DEBUG - Failed to parse: '{response[:100]}{'...' if len(response) > 100 else ''}'")
            # Not direct JSON, continue to regex parsing
            pass

        # Fallback: try to extract JSON from embedded text (traditional prompts)
        try:
            # Clean up the response - extract JSON if it's embedded in other text
            json_match = re.search(r'\{.*?"command".*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)

                if 'command' in parsed:
                    # Ensure command is a string before calling upper()
                    raw_command = parsed['command']
                    if not isinstance(raw_command, str):
                        raise ValueError(
                            f"Expected string for 'command', got {type(raw_command)}: {raw_command}")
                    command = raw_command.upper().strip()

                    # Check if this is an action sequence (contains comma)
                    if ',' in command:
                        # Validate sequence components
                        sequence_parts = [part.strip()
                                          for part in command.split(',')]
                        valid_sequence = True
                        for part in sequence_parts:
                            if not self._is_valid_action_name(part):
                                valid_sequence = False
                                break

                        if valid_sequence:
                            return command  # Return the comma-separated string
                        else:
                            print(f"⚠️ Invalid action sequence: {command}")

                    # Try to match single command to a valid Action
                    for action in Action:
                        if action.value == command:
                            return action

                    # Try partial matches for common variations
                    action_mappings = {
                        'UP': Action.MOVE_UP,
                        'DOWN': Action.MOVE_DOWN,
                        'LEFT': Action.MOVE_LEFT,
                        'RIGHT': Action.MOVE_RIGHT,
                        'PICK': Action.PICKUP,
                        'PICKUP': Action.PICKUP,
                        'EAT': Action.EAT,
                        'DRINK': Action.DRINK,
                        'WAIT': Action.WAIT,
                        'REST': Action.WAIT,
                        'EXPLORE': Action.EXPLORE,
                        'MATE': Action.SEEK_MATE,
                        'BREED': Action.BREED
                    }

                    for keyword, action in action_mappings.items():
                        if keyword in command:
                            return action

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ JSON parsing failed: {e}, falling back to text parsing")

        # Fallback: parse as regular text (for backwards compatibility)
        response = response.upper().strip()

        # Try to find a valid action in the response
        for action in Action:
            if action.value in response:
                return action

        # Fallback: try to match partial strings
        action_mappings = {
            'UP': Action.MOVE_UP,
            'DOWN': Action.MOVE_DOWN,
            'LEFT': Action.MOVE_LEFT,
            'RIGHT': Action.MOVE_RIGHT,
            'PICK': Action.PICKUP,
            'EAT': Action.EAT,
            'DRINK': Action.DRINK,
            'WAIT': Action.WAIT,
            'REST': Action.WAIT,
            'EXPLORE': Action.EXPLORE,
            'MATE': Action.SEEK_MATE,
            'BREED': Action.BREED
        }

        for keyword, action in action_mappings.items():
            if keyword in response:
                return action

        # Last resort: random valid action
        return Action.WAIT

    def _is_valid_action_name(self, action_name: str) -> bool:
        """Check if an action name is valid."""
        action_name = action_name.strip().upper()

        # Check if it's a valid Action enum value
        for action in Action:
            if action.value == action_name:
                return True

        # Check common mappings
        valid_mappings = {
            'UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'PICKUP',
            'EAT', 'DRINK', 'WAIT', 'REST', 'EXPLORE', 'MATE', 'BREED'
        }

        return action_name in valid_mappings

    def _get_fallback_action(self, agent: Agent, observations: Dict[str, Any]) -> Action:
        """Get a fallback action when VLM fails."""
        # Simple rule-based fallback with movement variety and boundary awareness

        # Emergency survival
        emergency_energy = self.survival_config.get(
            'energy', {}).get('critical', 20)
        if agent.energy < emergency_energy:
            if agent.inventory['food'] > 0:
                return Action.EAT
            elif agent.inventory['water'] > 0:
                return Action.DRINK
            else:
                return Action.WAIT

        # Resource gathering
        current_cell = observations.get('current_cell', 'EMPTY')
        if current_cell == 'FOOD' and agent.inventory['food'] < 2:
            return Action.PICKUP
        elif current_cell == 'WATER' and agent.inventory['water'] < 2:
            return Action.PICKUP

        # Basic needs
        basic_hunger = self.ai_config.get('recommended_hunger', 40)
        basic_thirst = self.ai_config.get('recommended_thirst', 40)
        if agent.hunger > basic_hunger and agent.inventory['food'] > 0:
            return Action.EAT
        elif agent.thirst > basic_thirst and agent.inventory['water'] > 0:
            return Action.DRINK

        # BOUNDARY-AWARE Movement with variety (avoid repetitive patterns)
        import random
        recent_memories = agent.get_recent_memories(3)
        recent_moves = [m.get('action', '') for m in recent_memories if m.get(
            'action', '').startswith('MOVE_')]

        # Get boundary-safe moves from agent position and world boundaries
        x, y = agent.position
        valid_moves = []

        # Only add moves that don't lead to boundaries - using world bounds directly
        world_width = observations.get('world_width', 10)  # Default fallback
        world_height = observations.get('world_height', 10)  # Default fallback

        if x > 0:  # Can move left
            valid_moves.append(Action.MOVE_LEFT)
        if x < world_width - 1:  # Can move right
            valid_moves.append(Action.MOVE_RIGHT)
        if y > 0:  # Can move up
            valid_moves.append(Action.MOVE_UP)
        if y < world_height - 1:  # Can move down
            valid_moves.append(Action.MOVE_DOWN)

        # If no valid moves (shouldn't happen), default to WAIT
        if not valid_moves:
            return Action.WAIT

        # If recent moves show repetition, avoid that direction
        if len(recent_moves) >= 2 and len(set(recent_moves)) == 1:
            # All recent moves are the same, try different directions
            last_move = recent_moves[0]
            available_alternatives = []

            if last_move == 'MOVE_RIGHT':
                available_alternatives = [m for m in valid_moves if m in [
                    Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT]]
            elif last_move == 'MOVE_LEFT':
                available_alternatives = [m for m in valid_moves if m in [
                    Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_RIGHT]]
            elif last_move == 'MOVE_UP':
                available_alternatives = [m for m in valid_moves if m in [
                    Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]]
            elif last_move == 'MOVE_DOWN':
                available_alternatives = [m for m in valid_moves if m in [
                    Action.MOVE_UP, Action.MOVE_LEFT, Action.MOVE_RIGHT]]

            if available_alternatives:
                return random.choice(available_alternatives)

        # Default: random boundary-safe movement
        return random.choice(valid_moves)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get VLM performance statistics."""
        avg_time = self.total_time / self.inference_count if self.inference_count > 0 else 0

        return {
            "model": self.model,
            "server": self.server,
            "total_inferences": self.inference_count,
            "total_time": self.total_time,
            "average_time": avg_time,
            "inferences_per_second": 1.0 / avg_time if avg_time > 0 else 0
        }

    def _direction_to_action(self, direction: str) -> str:
        """Convert direction name to action."""
        direction_map = {
            'north': 'MOVE_UP',
            'south': 'MOVE_DOWN',
            'east': 'MOVE_RIGHT',
            'west': 'MOVE_LEFT'
        }
        return direction_map.get(direction, 'WAIT')

    def _get_unexplored_directions(self, recent_moves: list) -> list:
        """Get directions that haven't been used recently."""
        if not recent_moves:
            return ['north', 'south', 'east', 'west']

        recent_directions = set()
        for move in recent_moves[-4:]:  # Last 4 moves
            if '_' in move:
                direction = move.split('_')[1].lower()
                direction_map = {'up': 'north', 'down': 'south',
                                 'right': 'east', 'left': 'west'}
                recent_directions.add(direction_map.get(direction, direction))

        all_directions = {'north', 'south', 'east', 'west'}
        return list(all_directions - recent_directions)

    def _get_opposite_direction(self, direction: str) -> str:
        """Get the opposite direction."""
        opposites = {
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left',
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east'
        }
        return opposites.get(direction, 'different direction')

    def _prepare_context_for_collection(self, agent: Agent, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context data for collection."""
        nearby_cells = observations.get('nearby_cells', {})

        # Extract nearby resources
        nearby_resources = []
        for pos_str, cell_type in nearby_cells.items():
            if cell_type in ['FOOD', 'WATER']:
                try:
                    # Parse position string like "(5,3)" or "(5, 3)"
                    pos_clean = pos_str.strip('()')
                    # Handle both formats: "5,3" and "5, 3"
                    x_str, y_str = pos_clean.split(',')
                    x, y = int(x_str.strip()), int(y_str.strip())
                    nearby_resources.append({
                        'type': cell_type.lower(),
                        'position': [x, y]
                    })
                except (ValueError, IndexError):
                    continue

        # Extract nearby agents
        nearby_agents = []
        for pos_str, cell_type in nearby_cells.items():
            if cell_type == 'AGENT':
                try:
                    # Parse position string like "(5,3)" or "(5, 3)"
                    pos_clean = pos_str.strip('()')
                    # Handle both formats: "5,3" and "5, 3"
                    x_str, y_str = pos_clean.split(',')
                    x, y = int(x_str.strip()), int(y_str.strip())
                    nearby_agents.append({
                        'position': [x, y],
                        # Manhattan distance
                        'distance': abs(x - agent.position[0]) + abs(y - agent.position[1])
                    })
                except (ValueError, IndexError):
                    continue

        # Available actions based on current state (use dummy world for basic actions)
        available_actions = self._get_available_actions()

        return {
            'nearby_resources': nearby_resources[:5],  # Limit to first 5
            'nearby_agents': nearby_agents,
            'available_actions': available_actions,
            'current_cell': observations.get('current_cell', 'EMPTY'),
            'turn': getattr(agent, 'current_turn', 0)
        }

    def collect_action_outcome(self, agent: Agent, success: bool = True,
                               energy_change: int = 0, **kwargs):
        """
        Collect the outcome of an action for training data.

        This should be called after an action is executed to record the results.
        """
        if not self.data_collector or not hasattr(agent, '_pending_data_collection'):
            return

        pending = getattr(agent, '_pending_data_collection')

        # Calculate actual energy change
        actual_energy_change = agent.energy - pending['initial_energy']

        # Build outcome
        outcome = {
            'success': success,
            'energy_change': actual_energy_change,
            'timestamp': time.time(),
            # Additional outcome data (food_collected, water_collected, etc.)
            **kwargs
        }

        # Collect the decision data
        self.data_collector.collect_decision(
            agent=agent,
            context=pending['context'],
            decision=pending['decision'],
            outcome=outcome
        )

        # Clean up pending data
        delattr(agent, '_pending_data_collection')

    def _extract_speech_text(self, response_text: str, action: Action, agent: Agent) -> str:
        """Extract meaningful speech text from VLM response."""
        # Remove the action from the response to get the reasoning
        action_text = action.value
        response_clean = response_text.replace(action_text, "").strip()

        # Create personality-based speech
        personality = agent.personality.lower()

        # Generate contextual speech based on agent state and action
        if agent.energy < 20:
            if "cautious" in personality:
                return "I need to be careful... energy is low"
            elif "aggressive" in personality:
                return "Running low on fuel, need to act fast!"
            else:
                return "Getting tired..."

        elif action == Action.EAT:
            if "efficient" in personality:
                return "Optimal nutrition timing"
            elif "social" in personality:
                return "Hope others have food too"
            else:
                return "Time to eat!"

        elif action == Action.DRINK:
            if "methodical" in personality:
                return "Hydration protocol activated"
            elif "opportunistic" in personality:
                return "Grabbing this water while I can"
            else:
                return "Need water!"

        elif action.value.startswith("MOVE"):
            direction = action.value.split("_")[1].lower()
            if "curious" in personality:
                return f"Let's explore {direction}!"
            elif "cautious" in personality:
                return f"Carefully going {direction}"
            elif "aggressive" in personality:
                return f"Charging {direction}!"
            else:
                return f"Moving {direction}"

        elif action == Action.PICKUP:
            if "hoarder" in personality:
                return "Mine! All mine!"
            elif "social" in personality:
                return "This might help the group"
            else:
                return "Useful resource!"

        elif action == Action.WAIT:
            if "strategist" in personality:
                return "Analyzing situation..."
            elif "efficient" in personality:
                return "Conserving energy"
            else:
                return "Waiting..."

        elif action == Action.SEEK_MATE:
            return "Looking for companionship"

        elif action == Action.BREED:
            return "Time to start a family!"

        # Default fallback
        if response_clean and len(response_clean) > 5:
            # Use first meaningful part of response
            words = response_clean.split()[:4]
            return " ".join(words) + "..."

        return f"I'll {action_text.lower()}"

    def finalize_data_collection(self):
        """Finalize and save collected data."""
        if self.data_collector:
            stats = self.data_collector.finalize()
            self.data_collector.create_training_dataset()
            return stats
        return None

    def _format_response_for_speech(self, response_text: str, action) -> str:
        """Format VLM response for speech bubble display, showing think+action or full error details."""
        import json
        import re

        # Try to parse as JSON first
        try:
            json_match = re.search(
                r'\{.*?"command".*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)

                command = parsed.get('command', 'UNKNOWN')
                think = parsed.get('think', 'No reasoning provided')

                # Format nicely for speech bubble with both think and action (clean text)
                clean_action = self._clean_text_for_display(
                    f"Action: {command}")
                clean_think = self._clean_text_for_display(f"Think: {think}")
                return f"{clean_action}\n\n{clean_think}"
            else:
                # JSON structure not found in response - ERROR (clean text)
                action_value = action.value if hasattr(
                    action, 'value') else str(action)
                error_msg = f"JSON PARSE ERROR\n\nRaw Response:\n{response_text[:150]}{'...' if len(response_text) > 150 else ''}\n\nUsing Fallback: {action_value}"
                return self._clean_text_for_display(error_msg)

        except json.JSONDecodeError as e:
            # JSON found but invalid format - ERROR (clean text)
            action_value = action.value if hasattr(
                action, 'value') else str(action)
            error_msg = f"JSON DECODE ERROR\n\nError: {str(e)[:50]}{'...' if len(str(e)) > 50 else ''}\n\nRaw:\n{response_text[:100]}{'...' if len(response_text) > 100 else ''}\n\nFallback: {action_value}"
            return self._clean_text_for_display(error_msg)
        except KeyError as e:
            # JSON valid but missing required keys - ERROR (clean text)
            action_value = action.value if hasattr(
                action, 'value') else str(action)
            error_msg = f"JSON KEY ERROR\n\nMissing: {str(e)}\n\nRaw:\n{response_text[:100]}{'...' if len(response_text) > 100 else ''}\n\nFallback: {action_value}"
            return self._clean_text_for_display(error_msg)
        except Exception as e:
            # Any other parsing error - ERROR (clean text)
            action_value = action.value if hasattr(
                action, 'value') else str(action)
            error_msg = f"PARSE ERROR\n\nError: {str(e)[:50]}{'...' if len(str(e)) > 50 else ''}\n\nRaw:\n{response_text[:100]}{'...' if len(response_text) > 100 else ''}\n\nFallback: {action_value}"
            return self._clean_text_for_display(error_msg)

    def _wrap_with_json_enforcement(self, prompt: str, use_death_warning: bool = False) -> str:
        """Wrap any prompt with strong JSON enforcement headers to prevent parser crashes."""
        # If using compact prompts, just return the prompt as-is since it already includes JSON instructions
        if self.compact_prompts:
            return prompt

        # For legacy mode, check if templates exist
        json_enforcement = self.templates.get(
            'json_enforcement', 'Respond with valid JSON only.')
        json_header = self.templates.get(
            'json_header', '{"action": "action_name", "reasoning": "explanation"}')
        json_footer = self.templates.get('json_footer', 'End JSON response.')
        death_warning = self.templates.get(
            'death_warning', '⚠️ CRITICAL WARNING')

        if use_death_warning:
            # Extra psychological pressure for problematic models
            return (death_warning + "\n\n" +
                    json_enforcement + "\n\n" +
                    json_header + "\n\n" +
                    prompt + "\n\n" +
                    json_footer + "\n\n" +
                    death_warning)
        else:
            # Standard enforcement
            return json_enforcement + "\n\n" + json_header + "\n\n" + prompt + "\n\n" + json_footer

    def _clean_text_for_display(self, text: str) -> str:
        """Clean text by removing problematic Unicode characters that don't render well in pygame."""
        import re
        import unicodedata

        # Replace common emoji with ASCII equivalents
        emoji_replacements = {
            '🎯': '->',
            '💭': '',
            '❌': 'X',
            '🔍': 'View:',
            '🔄': '>',
            '🚨': '!',
            '🐛': 'Bug:',
            '⚠️': '!',
            '💀': 'X',
            '🔥': '!',
            '⚡': '!',
            '🤖': 'AI:',
            '✅': 'OK',
            '📊': 'Data:',
            '📝': 'Note:',
            '🎉': '!',
            '🛡️': 'Guard:',
            '🔒': 'Lock:',
            '🎮': 'Game:',
            '💬': '',
            '🔮': '',
            '🌟': '*',
            '⭐': '*',
            '🚀': '->',
            '🎲': 'dice',
            '🎪': '',
            '🎨': 'art',
            # Add more as needed
        }

        # Replace emoji with ASCII
        for emoji, ascii_replacement in emoji_replacements.items():
            text = text.replace(emoji, ascii_replacement)

        # Remove emoji using Unicode categories (more comprehensive)
        # This removes most emoji while preserving accented letters
        text = ''.join(char for char in text if not unicodedata.category(
            char).startswith('So'))

        # Remove other problematic Unicode categories that cause pygame issues
        # Keep letters, numbers, punctuation, symbols, whitespace
        # Remove: Control characters, format characters, private use, surrogates
        problematic_categories = {'Cc', 'Cf', 'Co', 'Cs'}
        text = ''.join(char for char in text if unicodedata.category(
            char) not in problematic_categories)

        # Clean up extra whitespace but preserve line breaks
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove extra spaces within lines
            cleaned_line = re.sub(r' +', ' ', line.strip())
            cleaned_lines.append(cleaned_line)

        text = '\n'.join(cleaned_lines)

        return text

    def _needs_death_warning(self) -> bool:
        """Check if current model likely needs extra psychological pressure for JSON compliance."""
        # Models known to be problematic with JSON generation
        problematic_models = [
            'gemma2:2b',    # Small models often struggle with strict formatting
            'phi3:4b',      # 4B models mentioned by user as error-prone
            'llama3.2:3b',  # Small parameter counts = more likely to ignore instructions
            'qwen2.5:3b',
            'codellama:7b',  # Code models sometimes add explanations
            'mistral:7b'    # Some versions ignore JSON requirements
        ]

        model_lower = self.model.lower()
        for problematic in problematic_models:
            if problematic.lower() in model_lower:
                return True

        # Check for "b" parameter indicators (suggesting smaller models)
        if any(size in model_lower for size in ['1b', '2b', '3b', '4b']):
            return True

        return False

    def _try_get_action_with_retry(self, agent: Agent, world: World, observations: Dict[str, Any], max_retries: int = 1) -> tuple:
        """
        Try to get an action from VLM with retry mechanism for failed JSON parsing.
        First tries structured outputs, then falls back to traditional prompts.

        Returns:
            Tuple of (action, final_response_text)
        """
        prompt = self._generate_prompt(agent, observations, world)

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if attempt == 0:
                    # First attempt with structured outputs (most reliable)
                    print(f"🎯 Agent {agent.id} using structured outputs")
                    response_text = self._query_vlm(
                        prompt, use_structured_output=True)
                elif attempt == 1:
                    # Second attempt with traditional strong enforcement
                    retry_prompt = self._create_retry_prompt(
                        prompt, response_text)
                    print(
                        f"🔄 Agent {agent.id} retry with traditional enforcement")
                    response_text = self._query_vlm(
                        retry_prompt, use_structured_output=False)
                else:
                    # Final attempts with maximum psychological pressure
                    retry_prompt = self._create_retry_prompt(
                        prompt, response_text)
                    print(
                        f"🆘 Agent {agent.id} final attempt {attempt} (traditional prompt)")
                    response_text = self._query_vlm(
                        retry_prompt, use_structured_output=False)

                # Try to parse the action
                action = self._parse_action(
                    response_text, agent.position, world)

                # Check if we got a proper JSON response or fell back to text parsing
                if self._is_valid_json_response(response_text):
                    print(
                        f"✅ Agent {agent.id} valid JSON on attempt {attempt + 1}")
                    return action, response_text
                elif attempt < max_retries:
                    # JSON parsing failed, but we have retries left
                    print(
                        f"⚠️ Agent {agent.id} invalid JSON (attempt {attempt + 1}), retrying...")
                    continue
                else:
                    # Last attempt or no retries left, accept whatever we got
                    print(
                        f"🔄 Agent {agent.id} using fallback action after {attempt + 1} attempts")
                    return action, response_text

            except Exception as e:
                if attempt < max_retries:
                    print(
                        f"❌ Agent {agent.id} VLM error (attempt {attempt + 1}): {e}")
                    continue
                else:
                    # Final attempt failed, let the calling method handle fallback
                    raise e

        # Should never reach here, but just in case
        return Action.WAIT, "No response generated"

    def _is_valid_json_response(self, response: str) -> bool:
        """Check if response contains valid JSON with required fields."""
        import json
        import re

        # First, try direct JSON parsing (structured outputs)
        try:
            parsed = json.loads(response.strip())
            return 'command' in parsed and 'think' in parsed
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract JSON from embedded text (traditional prompts)
        try:
            json_match = re.search(r'\{.*?"command".*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return 'command' in parsed and 'think' in parsed
        except:
            pass
        return False

    def _create_retry_prompt(self, original_prompt: str, failed_response: str) -> str:
        """Create a retry prompt with stronger JSON enforcement after a failure."""
        retry_message = f"""
{self.templates['death_warning']}

❌ PREVIOUS RESPONSE FAILED: Your last response did not contain valid JSON!
❌ FAILED RESPONSE: "{failed_response[:100]}{'...' if len(failed_response) > 100 else ''}"

RETRY REQUIRED: Please generate valid JSON this time.
FORMAT: {{"command": "ACTION", "think": "your reasoning"}}
NO TEXT OUTSIDE THE JSON BRACES!

{original_prompt}

{self.templates.get('json_footer', 'End JSON response.')}
"""
        return retry_message

    def _get_blocked_directions(self, position: Tuple[int, int], nearby_cells: Dict[str, str]) -> str:
        """
        Determine which movement directions are blocked by boundaries or obstacles.
        Returns a string describing blocked directions.
        """
        x, y = position
        blocked_dirs = []

        # Check each direction
        directions = {
            'UP': (x, y - 1),
            'DOWN': (x, y + 1),
            'LEFT': (x - 1, y),
            'RIGHT': (x + 1, y)
        }

        for direction, (check_x, check_y) in directions.items():
            cell_key = f"({check_x},{check_y})"
            if cell_key in nearby_cells:
                cell_type = nearby_cells[cell_key]
                if cell_type in ['BOUNDARY', 'OBSTACLE']:
                    blocked_dirs.append(direction)

        if blocked_dirs:
            result = f"BLOCKED DIRECTIONS: {', '.join(blocked_dirs)} (cannot move there)"
            print(f"🚫 Agent at {position} - {result}")
            return result
        return "All directions are accessible"

    def _generate_agent_prompt(self, agent: Agent, world: World, observations: Dict[str, Any]) -> str:
        """Generate the agent prompt including status, inventory, location, and nearby resources."""
        status_awareness = self._get_comprehensive_status_awareness(agent)

        # Get basic info
        nearby_cells = observations.get('nearby_cells', {})
        current_cell = observations.get('current_cell', 'EMPTY')
        recent_memories = agent.get_recent_memories(3)
        recent_actions = [m.get('action', '') for m in recent_memories]

        # Enhanced status info with awareness
        urgency_indicator = ""
        if status_awareness['is_critical']:
            urgency_indicator = "💀 CRITICAL STATE "
        elif status_awareness['is_suffering']:
            urgency_indicator = "⚠️ SUFFERING "

        status = f"{urgency_indicator}Energy: {agent.energy}/100 | Hunger: {agent.hunger}/100 | Thirst: {agent.thirst}/100"

        # Add condition awareness
        if status_awareness['conditions']:
            condition_text = ", ".join(status_awareness['conditions'])
            status += f" | Conditions: {condition_text}"

        # Add survival priority
        status += f" | Priority: {status_awareness['survival_priority']}"

        # Add estimated survival time if critical
        if status_awareness['is_suffering']:
            status += f" | Est. survival: ~{status_awareness['estimated_turns_left']} turns"
        inventory = f"Inventory: {agent.inventory['food']} food, {agent.inventory['water']} water"

        # Enhanced location with world boundary context
        world_width = observations.get('world_width', 'unknown')
        world_height = observations.get('world_height', 'unknown')
        boundary_cells = observations.get('boundary_cells', 0)

        location = f"Position: {agent.position} | Current cell: {current_cell}"
        if isinstance(world_width, int) and isinstance(world_height, int):
            location += f" | World size: {world_width}x{world_height} (valid positions: 0-{world_width-1}, 0-{world_height-1})"
        else:
            location += f" | World size: {world_width}x{world_height}"

        if boundary_cells > 0:
            location += f" | {boundary_cells} boundary cells detected nearby (cannot move there)"
            location += f" | AVOID moving to any cell marked as 'BOUNDARY' or 'OBSTACLE'"

        # Add explicit blocked directions information
        blocked_directions = self._get_blocked_directions(
            agent.position, nearby_cells)
        location += f" | {blocked_directions}"

        # Get nearby resources and agents with directions
        nearby_info = self._get_directional_resources(
            agent.position, nearby_cells)

        # Detect if stuck in repetitive behavior
        is_stuck = len(set(recent_actions)) == 1 and len(recent_actions) >= 2

        # Get blocked directions
        blocked_directions = self._get_blocked_directions(
            agent.position, nearby_cells)

        return self.templates.get('json_header', '{"action": "action_name", "reasoning": "explanation"}') + "\n\n" + f"""AGENT STATUS:
{status}

LOCATION:
{location}

{blocked_directions}

NEARBY RESOURCES:
{nearby_info}

RECENT ACTIONS:
{', '.join(recent_actions[-3:])}

STRATEGIC ANALYSIS:
{self._analyze_resource_layout(nearby_cells)}

{self.templates.get('json_footer', 'End JSON response.')}
"""

    def _get_survival_guidance(self, agent: Agent, status_awareness: Dict[str, Any]) -> str:
        """
        Generate survival guidance for eating and drinking based on agent's needs.
        """
        guidance = []

        # Check if agent has resources and needs them
        has_food = agent.inventory["food"] > 0
        has_water = agent.inventory["water"] > 0

        # High priority actions
        urgent_hunger = self.ai_config.get('urgent_hunger', 70)
        urgent_thirst = self.ai_config.get('urgent_thirst', 70)
        recommended_hunger = self.ai_config.get('recommended_hunger', 40)
        recommended_thirst = self.ai_config.get('recommended_thirst', 40)

        if agent.hunger > urgent_hunger and has_food:
            guidance.append("🍎 URGENT: EAT food NOW to prevent starvation!")
        elif agent.thirst > urgent_thirst and has_water:
            guidance.append(
                "💧 URGENT: DRINK water NOW to prevent dehydration!")

        # Medium priority actions
        elif agent.hunger > recommended_hunger and has_food:
            guidance.append("🍎 RECOMMENDED: EAT food to reduce hunger")
        elif agent.thirst > recommended_thirst and has_water:
            guidance.append("💧 RECOMMENDED: DRINK water to reduce thirst")

        # Low priority actions
        else:
            low_hunger = self.ai_config.get('low_priority_hunger', 20)
            low_thirst = self.ai_config.get('low_priority_thirst', 20)

            if agent.hunger > low_hunger and has_food:
                guidance.append("🍎 Consider eating food while you have it")
            elif agent.thirst > low_thirst and has_water:
                guidance.append("💧 Consider drinking water while you have it")

        # Resource shortage warnings
        shortage_hunger = self.ai_config.get('resource_shortage_hunger', 70)
        shortage_thirst = self.ai_config.get('resource_shortage_thirst', 70)

        # Resource shortage warnings
        if agent.hunger > shortage_hunger and not has_food:
            guidance.append(
                "⚠️ HUNGRY but no food in inventory - find FOOD urgently!")
        if agent.thirst > shortage_thirst and not has_water:
            guidance.append(
                "⚠️ THIRSTY but no water in inventory - find WATER urgently!")

        if guidance:
            return "SURVIVAL PRIORITIES:\n" + "\n".join(guidance)
        return ""

    def _validate_action_for_agent(self, action, agent_position, world):
        """Validate that an action is allowed for an agent at a specific position."""
        if not agent_position or not world:
            return True  # Can't validate, allow

        # Convert action to string if it's an Action enum
        action_str = action.value if hasattr(action, 'value') else str(action)

        x, y = agent_position

        # Check for boundary escape attempts
        if "MOVE_LEFT" in action_str and x == 0:
            print(f"🚨 BLOCKED: Agent at left edge cannot move LEFT (boundary escape)")
            return False
        elif "MOVE_RIGHT" in action_str and x == world.width - 1:
            print(f"🚨 BLOCKED: Agent at right edge cannot move RIGHT (boundary escape)")
            return False
        elif "MOVE_UP" in action_str and y == 0:
            print(f"🚨 BLOCKED: Agent at top edge cannot move UP (boundary escape)")
            return False
        elif "MOVE_DOWN" in action_str and y == world.height - 1:
            print(f"🚨 BLOCKED: Agent at bottom edge cannot move DOWN (boundary escape)")
            return False

        return True

    def _get_boundary_safe_fallback(self, agent_position, world):
        """Get a safe fallback action for agents near boundaries."""
        if not agent_position or not world:
            return Action.WAIT

        x, y = agent_position
        safe_actions = []

        # Only suggest movements that don't hit boundaries
        if x > 0:
            safe_actions.append(Action.MOVE_LEFT)
        if x < world.width - 1:
            safe_actions.append(Action.MOVE_RIGHT)
        if y > 0:
            safe_actions.append(Action.MOVE_UP)
        if y < world.height - 1:
            safe_actions.append(Action.MOVE_DOWN)

        # Always-safe actions
        safe_actions.extend([Action.WAIT, Action.EXPLORE,
                            Action.PICKUP, Action.EAT, Action.DRINK])

        import random
        return random.choice(safe_actions)

    def get_spatial_statistics(self) -> Dict[str, Any]:
        """Get statistics about spatial system performance."""
        return self.spatial_system.get_statistics()

    def _generate_compact_prompt(self, agent: Agent, status: str, inventory: str, location: str, nearby: Dict, is_stuck: bool, status_awareness: Dict[str, Any], world: 'World' = None) -> str:
        """Generate ultra-compact prompts to minimize token usage."""

        # Get basic directional info
        direction_resources = self._get_specific_directional_guidance(nearby)
        exploration_guidance = "STUCK! Try different direction!" if is_stuck else ""

        # Handle crisis situations compactly
        crisis_actions = self._get_survival_actions(agent)
        if status_awareness['is_critical'] or crisis_actions:
            urgency_msg = "🚨 CRITICAL" if status_awareness['is_critical'] else "⚠️ WARNING"
            survival_time = f" {status_awareness['estimated_turns_left']}T left" if status_awareness['is_suffering'] else ""

            return self.templates['compact_crisis'].format(
                urgency_msg=urgency_msg,
                status=status,
                inventory=inventory,
                location=location,
                survival_time=survival_time,
                crisis_actions=crisis_actions or 'Low health',
                survival_priority=status_awareness['survival_priority'],
                crisis_instructions="Get food/water NOW!",
                json_instruction=self._get_json_instruction(
                    self._get_available_actions(agent_position=agent.position, world=world))
            )

        # Determine personality-specific compact template
        personality = agent.personality.lower()
        if "cautious explorer" in personality:
            template_name = 'compact_cautious_explorer'
        elif "aggressive" in personality:
            template_name = 'compact_aggressive_hoarder'
        elif "curious wanderer" in personality:
            template_name = 'compact_curious_wanderer'
        elif "efficient collector" in personality:
            template_name = 'compact_efficient_collector'
        elif "methodical strategist" in personality:
            template_name = 'compact_methodical_strategist'
        elif "opportunistic survivor" in personality:
            template_name = 'compact_opportunistic_survivor'
        elif "social collaborator" in personality:
            template_name = 'compact_social_collaborator'
        elif "protective guardian" in personality:
            template_name = 'compact_protective_guardian'
        else:
            # Generic compact format for other personalities
            return f"{agent.personality.upper()} - {status} | {inventory} | {location}\n\nNearby: {direction_resources}\n{exploration_guidance}\n\n{self._get_json_instruction(self._get_available_actions(agent_position=agent.position, world=world))}"

        return self.templates[template_name].format(
            status=status,
            inventory=inventory,
            location=location,
            direction_resources=direction_resources,
            exploration_guidance=exploration_guidance,
            json_instruction=self._get_json_instruction(
                self._get_available_actions(agent_position=agent.position, world=world))
        )
