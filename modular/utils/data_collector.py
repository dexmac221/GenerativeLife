"""
Data Collector for AI Arena - Collects bot memory data for LoRA training.

This module captures bot interactions, decisions, and outcomes in a format
suitable for LoRA fine-tuning of language models.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from ..core.agent import Agent


class DataCollector:
    """
    Collects and formats bot interaction data for LoRA training.

    The data format follows common LoRA training conventions:
    - Input/output pairs for instruction tuning
    - Context-aware conversation format
    - Personality-driven decision making examples
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data collector.

        Args:
            data_dir: Base directory for saving data
        """
        self.data_dir = data_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(
            data_dir, "sessions", f"session_{self.session_id}.jsonl")
        self.training_file = os.path.join(
            data_dir, "training", f"training_data_{self.session_id}.jsonl")

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.training_file), exist_ok=True)

        # Data buffers
        self.session_data = []
        self.training_data = []

        print(f"📊 Data collector initialized for session: {self.session_id}")

    def collect_decision(self, agent: Agent, context: Dict[str, Any],
                         decision: str, outcome: Dict[str, Any]):
        """
        Collect a bot decision with full context for training.

        Args:
            agent: The agent making the decision
            context: Situational context (world state, nearby agents, etc.)
            decision: The decision made by the agent
            outcome: Results of the decision (success, energy change, etc.)
        """
        timestamp = time.time()

        # Session data (complete record)
        session_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "agent_id": agent.id,
            "agent_info": self._extract_agent_info(agent),
            "context": context,
            "decision": decision,
            "outcome": outcome
        }
        self.session_data.append(session_entry)

        # Training data (LoRA format)
        training_entry = self._format_for_training(
            agent, context, decision, outcome)
        if training_entry:
            self.training_data.append(training_entry)

    def _extract_agent_info(self, agent: Agent) -> Dict[str, Any]:
        """Extract relevant agent information."""
        # Handle gender safely - it could be enum or string
        gender_value = agent.gender
        if hasattr(gender_value, 'value'):
            gender_str = gender_value.value
        else:
            gender_str = str(gender_value)

        return {
            "id": agent.id,
            "gender": gender_str,
            "age": agent.age,
            "personality": getattr(agent, 'personality', 'Unknown'),
            "position": agent.position,
            "energy": agent.energy,
            "hunger": agent.hunger,
            "thirst": agent.thirst,
            "is_mature": agent.is_mature,
            "is_pregnant": getattr(agent, 'is_pregnant', False),
            "is_fertile": getattr(agent, 'is_fertile', False),
            "inventory": agent.inventory.copy()
        }

    def _format_for_training(self, agent: Agent, context: Dict[str, Any],
                             decision: str, outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format data for LoRA training using instruction-following format.

        This creates training examples that teach the model to make decisions
        based on personality, context, and past experiences.
        """
        # Build the instruction prompt
        instruction = self._build_instruction_prompt(agent, context)

        # The agent's actual decision becomes the response
        response = decision

        # Only include if the decision led to a meaningful outcome
        if not self._is_meaningful_outcome(outcome):
            return None

        # LoRA training format (Alpaca-style instruction tuning)
        return {
            "instruction": instruction,
            "input": "",  # Context is included in instruction
            "output": response,
            "metadata": {
                "agent_personality": getattr(agent, 'personality', 'Unknown'),
                "outcome_success": outcome.get('success', False),
                "energy_change": outcome.get('energy_change', 0),
                "timestamp": time.time()
            }
        }

    def _build_instruction_prompt(self, agent: Agent, context: Dict[str, Any]) -> str:
        """
        Build a comprehensive instruction prompt for the AI agent.

        This prompt includes personality, current state, and context to help
        the model learn personality-driven decision making.
        """
        prompt_parts = []

        # Agent personality and identity
        personality = getattr(agent, 'personality', 'balanced individual')
        prompt_parts.append(
            f"You are an AI agent with the personality: {personality}")

        # Current status
        status = f"Current status - Energy: {agent.energy}/100, Hunger: {agent.hunger}/100, Thirst: {agent.thirst}/100"
        prompt_parts.append(status)

        # Position and environment
        position_info = f"Position: {agent.position}"
        prompt_parts.append(position_info)

        # Nearby resources and agents
        if 'nearby_resources' in context:
            resources = context['nearby_resources']
            if resources:
                resource_list = [
                    f"{res['type']} at {res['position']}" for res in resources[:3]]
                prompt_parts.append(
                    f"Nearby resources: {', '.join(resource_list)}")

        if 'nearby_agents' in context:
            agents = context['nearby_agents']
            if agents:
                agent_list = []
                for agent_info in agents[:2]:
                    # Handle missing gender gracefully
                    gender = agent_info.get('gender', 'unknown')
                    position = agent_info.get('position', 'unknown')
                    agent_list.append(f"{gender} agent at {position}")
                prompt_parts.append(f"Nearby agents: {', '.join(agent_list)}")

        # Available actions
        if 'available_actions' in context:
            actions = context['available_actions']
            prompt_parts.append(f"Available actions: {', '.join(actions)}")

        # Inventory
        if agent.inventory:
            inventory_items = [f"{item}: {count}" for item,
                               count in agent.inventory.items() if count > 0]
            if inventory_items:
                prompt_parts.append(f"Inventory: {', '.join(inventory_items)}")

        # Recent memories (if available)
        if hasattr(agent, 'get_recent_memories'):
            recent_memories = agent.get_recent_memories(2)
            if recent_memories:
                memory_text = []
                for memory in recent_memories:
                    action = memory.get('action', 'Unknown')
                    turn = memory.get('turn', '?')
                    memory_text.append(f"Turn {turn}: {action}")
                prompt_parts.append(
                    f"Recent actions: {'; '.join(memory_text)}")

        # Task prompt
        prompt_parts.append(
            "Based on your personality and current situation, what action should you take?")

        return " | ".join(prompt_parts)

    def _is_meaningful_outcome(self, outcome: Dict[str, Any]) -> bool:
        """
        Check if an outcome is meaningful enough to include in training data.

        We want to filter out trivial decisions that don't provide learning value.
        """
        # Include if there was a significant energy change
        energy_change = abs(outcome.get('energy_change', 0))
        if energy_change >= 5:
            return True

        # Include if action was successful or failed notably
        if 'success' in outcome and outcome['success'] is not None:
            return True

        # Include breeding-related outcomes
        if any(key in outcome for key in ['breeding_attempt', 'pregnancy', 'birth']):
            return True

        # Include resource collection
        if any(key in outcome for key in ['food_collected', 'water_collected']):
            return True

        return False

    def save_session_data(self):
        """Save session data to file."""
        try:
            with open(self.session_file, 'w') as f:
                for entry in self.session_data:
                    f.write(json.dumps(entry) + '\n')
            print(
                f"💾 Saved {len(self.session_data)} session entries to {self.session_file}")
        except Exception as e:
            print(f"❌ Error saving session data: {e}")

    def save_training_data(self):
        """Save training data in LoRA-compatible format."""
        try:
            with open(self.training_file, 'w') as f:
                for entry in self.training_data:
                    f.write(json.dumps(entry) + '\n')
            print(
                f"🎯 Saved {len(self.training_data)} training entries to {self.training_file}")
        except Exception as e:
            print(f"❌ Error saving training data: {e}")

    def create_training_dataset(self):
        """
        Create a consolidated training dataset from all collected data.

        This combines data from multiple sessions and formats it for LoRA training.
        """
        all_training_data = []

        # Collect all training files
        training_dir = os.path.join(self.data_dir, "training")
        if os.path.exists(training_dir):
            for filename in os.listdir(training_dir):
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(training_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            for line in f:
                                entry = json.loads(line.strip())
                                all_training_data.append(entry)
                    except Exception as e:
                        print(f"⚠️ Error reading {filepath}: {e}")

        # Save consolidated dataset
        consolidated_file = os.path.join(
            self.data_dir, "consolidated_training_data.jsonl")
        try:
            with open(consolidated_file, 'w') as f:
                for entry in all_training_data:
                    f.write(json.dumps(entry) + '\n')
            print(
                f"📚 Created consolidated dataset with {len(all_training_data)} entries: {consolidated_file}")
        except Exception as e:
            print(f"❌ Error creating consolidated dataset: {e}")

        return len(all_training_data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        personality_counts = {}
        action_counts = {}

        for entry in self.training_data:
            # Count personalities
            personality = entry['metadata']['agent_personality']
            personality_counts[personality] = personality_counts.get(
                personality, 0) + 1

            # Count action types
            action = entry['output']
            action_type = action.split()[0] if action else 'unknown'
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        return {
            "session_id": self.session_id,
            "total_entries": len(self.session_data),
            "training_entries": len(self.training_data),
            "personality_distribution": personality_counts,
            "action_distribution": action_counts
        }

    def finalize(self):
        """Save all data and create final statistics."""
        self.save_session_data()
        self.save_training_data()

        stats = self.get_statistics()

        # Save statistics
        stats_file = os.path.join(
            self.data_dir, "sessions", f"stats_{self.session_id}.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Session statistics saved to {stats_file}")
        except Exception as e:
            print(f"❌ Error saving statistics: {e}")

        return stats
