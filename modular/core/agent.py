"""
Agent module - Core agent definitions and behaviors.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random


class Gender(Enum):
    """Agent gender enumeration."""
    MALE = "MALE"
    FEMALE = "FEMALE"


class Action(Enum):
    """Available actions for agents."""
    # Basic actions
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    PICKUP = "PICKUP"
    DRINK = "DRINK"
    EAT = "EAT"
    EXPLORE = "EXPLORE"
    WAIT = "WAIT"
    SEEK_MATE = "SEEK_MATE"
    BREED = "BREED"

    # Compound actions for efficiency
    MOVE_UP_AND_PICKUP = "MOVE_UP_AND_PICKUP"
    MOVE_DOWN_AND_PICKUP = "MOVE_DOWN_AND_PICKUP"
    MOVE_LEFT_AND_PICKUP = "MOVE_LEFT_AND_PICKUP"
    MOVE_RIGHT_AND_PICKUP = "MOVE_RIGHT_AND_PICKUP"

    PICKUP_AND_EAT = "PICKUP_AND_EAT"
    PICKUP_AND_DRINK = "PICKUP_AND_DRINK"

    EAT_AND_DRINK = "EAT_AND_DRINK"

    # Ultimate efficiency combinations
    MOVE_UP_PICKUP_EAT = "MOVE_UP_PICKUP_EAT"
    MOVE_DOWN_PICKUP_EAT = "MOVE_DOWN_PICKUP_EAT"
    MOVE_LEFT_PICKUP_EAT = "MOVE_LEFT_PICKUP_EAT"
    MOVE_RIGHT_PICKUP_EAT = "MOVE_RIGHT_PICKUP_EAT"

    MOVE_UP_PICKUP_DRINK = "MOVE_UP_PICKUP_DRINK"
    MOVE_DOWN_PICKUP_DRINK = "MOVE_DOWN_PICKUP_DRINK"
    MOVE_LEFT_PICKUP_DRINK = "MOVE_LEFT_PICKUP_DRINK"
    MOVE_RIGHT_PICKUP_DRINK = "MOVE_RIGHT_PICKUP_DRINK"


@dataclass
class Agent:
    """
    Represents an intelligent agent in the simulation.

    Features:
    - Basic survival needs (energy, hunger, thirst)
    - Reproduction system (gender, age, breeding)
    - Memory and learning capabilities
    - Inventory management
    """

    # Core identity
    id: str
    position: Tuple[int, int]
    gender: Gender
    personality: str

    # Vital stats
    energy: int = 100
    hunger: int = 0
    thirst: int = 0
    age: int = 0

    # Reproductive system
    is_mature: bool = False
    maturity_age: int = 50  # Age when agent becomes mature
    is_fertile: bool = True
    fertility_decline_age: int = 200  # Age when fertility starts declining
    is_pregnant: bool = False
    pregnancy_duration: int = 30  # Turns to carry offspring
    pregnancy_timer: int = 0
    partner: Optional['Agent'] = None
    children: List[str] = field(default_factory=list)  # Child IDs

    # Game state
    alive: bool = True
    inventory: Dict[str, int] = field(
        default_factory=lambda: {"food": 0, "water": 0})
    memory: List[Dict[str, Any]] = field(default_factory=list)

    # Thoughts and mental state
    current_thought: str = ""
    thought_timestamp: float = 0.0
    thought_duration: float = 3.0  # How long to show each thought

    # Speech system for communication bubbles
    current_speech: str = ""
    speech_timestamp: float = 0.0
    # How long to show speech bubbles - increased for full VLM responses
    speech_duration: float = 6.0

    def __post_init__(self):
        """Initialize agent after creation."""
        if self.age >= self.maturity_age:
            self.is_mature = True

    def can_breed(self) -> bool:
        """Check if agent can participate in breeding."""
        return (
            self.alive and
            self.is_mature and
            self.is_fertile and
            not self.is_pregnant and
            self.energy >= 50  # Minimum energy for breeding
        )

    def is_compatible_mate(self, other: 'Agent') -> bool:
        """Check if this agent can breed with another agent."""
        if not other or not other.alive:
            return False

        try:
            # Safe gender comparison
            self_gender = getattr(self.gender, 'value', str(self.gender))
            other_gender = getattr(other.gender, 'value', str(other.gender))
            different_genders = self_gender != other_gender
        except AttributeError:
            different_genders = True  # Assume compatible if gender access fails

        return (
            different_genders and  # Different genders
            self.can_breed() and
            other.can_breed() and
            self.partner != other and  # Not already paired
            other.partner != self
        )

    def start_pregnancy(self, partner: 'Agent'):
        """Start pregnancy process."""
        try:
            is_female = (hasattr(self.gender, 'value') and self.gender.value == 'FEMALE') or \
                (not hasattr(self.gender, 'value')
                 and str(self.gender) == 'FEMALE')
            if is_female and self.can_breed():
                self.is_pregnant = True
                self.pregnancy_timer = self.pregnancy_duration
                self.partner = partner
                partner.partner = self
        except AttributeError:
            # If gender access fails, don't start pregnancy
            pass

    def update_pregnancy(self) -> bool:
        """Update pregnancy status. Returns True if ready to give birth."""
        if not self.is_pregnant:
            return False

        self.pregnancy_timer -= 1

        # Pregnancy effects
        if self.pregnancy_timer > 0:
            self.energy = max(0, self.energy - 1)  # Pregnancy drains energy
            return False
        else:
            # Ready to give birth
            self.is_pregnant = False
            self.pregnancy_timer = 0
            return True

    def age_up(self):
        """Age the agent and update maturity/fertility status."""
        self.age += 1

        if not self.is_mature and self.age >= self.maturity_age:
            self.is_mature = True

        if self.is_fertile and self.age >= self.fertility_decline_age:
            # Gradual fertility decline
            fertility_chance = max(
                0.1, 1.0 - (self.age - self.fertility_decline_age) / 100)
            if random.random() > fertility_chance:
                self.is_fertile = False

    def add_memory(self, action: str, result: str, context: Dict[str, Any]):
        """Add experience to agent's memory."""
        memory_entry = {
            "turn": context.get("turn", 0),
            "action": action,
            "result": result,
            "position": self.position,
            "energy": self.energy,
            "hunger": self.hunger,
            "thirst": self.thirst
        }

        self.memory.append(memory_entry)

        # Keep memory size manageable
        if len(self.memory) > 50:
            self.memory.pop(0)

    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories for decision making."""
        return self.memory[-count:] if self.memory else []

    def update_vitals(self):
        """Update agent's vital statistics each turn."""
        # Natural aging
        self.age_up()

        # Energy decay
        self.energy = max(0, self.energy - 1)

        # Hunger and thirst increase
        self.hunger = min(100, self.hunger + 2)
        self.thirst = min(100, self.thirst + 1)

        # Death conditions
        if self.energy <= 0 or self.hunger >= 100 or self.thirst >= 100:
            self.alive = False

        # Pregnancy updates
        if self.is_pregnant:
            self.update_pregnancy()

    def consume_food(self) -> bool:
        """Consume food from inventory."""
        if self.inventory["food"] > 0:
            self.inventory["food"] -= 1
            self.hunger = max(0, self.hunger - 20)
            self.energy = min(100, self.energy + 15)
            return True
        return False

    def consume_water(self) -> bool:
        """Consume water from inventory."""
        if self.inventory["water"] > 0:
            self.inventory["water"] -= 1
            self.thirst = max(0, self.thirst - 20)
            self.energy = min(100, self.energy + 10)
            return True
        return False

    def add_child(self, child_id: str):
        """Add a child to this agent's family."""
        self.children.append(child_id)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of agent's current status."""
        # Safe gender value extraction
        try:
            if hasattr(self.gender, 'value'):
                gender_str = self.gender.value
            else:
                gender_str = str(self.gender)
        except AttributeError:
            gender_str = "UNKNOWN"

        return {
            "id": self.id,
            "position": self.position,
            "gender": gender_str,
            "age": self.age,
            "alive": self.alive,
            "energy": self.energy,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "is_mature": self.is_mature,
            "is_fertile": self.is_fertile,
            "is_pregnant": self.is_pregnant,
            "children_count": len(self.children),
            "inventory": self.inventory.copy(),
            "personality": self.personality
        }

    def set_speech(self, speech_text: str):
        """Set speech text for the agent with timestamp."""
        import time
        self.current_speech = speech_text
        self.speech_timestamp = time.time()

    def get_current_speech(self) -> str:
        """Get current speech if still within duration, empty string otherwise."""
        import time
        if not self.current_speech:
            return ""

        elapsed = time.time() - self.speech_timestamp
        if elapsed > self.speech_duration:
            self.current_speech = ""
            return ""

        return self.current_speech

    def has_active_speech(self) -> bool:
        """Check if agent currently has active speech to display."""
        return bool(self.get_current_speech())
