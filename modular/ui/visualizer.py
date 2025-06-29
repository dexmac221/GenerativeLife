"""
Visualizer - PyGame-based visualization for the simulation.
"""

import pygame
import sys
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from ..core.agent import Agent, Gender
from ..core.world import World, CellType


class SpeechMode(Enum):
    """Speech bubble display modes."""
    OFF = "Off"
    ACTIONS_ONLY = "Actions Only"
    FULL_RESPONSE = "Full Response"


class Visualizer:
    """
    PyGame-based visualizer for the AI Arena simulation.

    Features:
    - Real-time grid visualization
    - Agent status display
    - Gender-specific representations
    - Breeding status indicators
    - Interactive controls
    """

    def __init__(self, world: World, window_size: Tuple[int, int] = (1600, 1000)):
        """
        Initialize the visualizer.

        Args:
            world: The world to visualize
            window_size: Window dimensions (width, height)
        """
        self.world = world
        self.window_size = window_size

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("AI Arena - Modular Version")
        self.clock = pygame.time.Clock()

        # Calculate grid display parameters
        self.sidebar_width = 300
        self.grid_width = window_size[0] - self.sidebar_width
        self.grid_height = window_size[1]

        self.cell_width = self.grid_width // world.width
        self.cell_height = self.grid_height // world.height

        # Colors
        self.colors = {
            'background': (40, 40, 40),
            'grid': (80, 80, 80),
            'empty': (50, 50, 50),
            'food': (100, 200, 100),
            'water': (100, 150, 255),
            'obstacle': (120, 120, 120),
            'agent_male': (255, 100, 100),    # Red for males
            'agent_female': (255, 100, 255),  # Magenta for females
            'text': (255, 255, 255),
            'sidebar': (30, 30, 30),
            'pregnant': (255, 200, 0),        # Gold for pregnancy
            'fertile': (0, 255, 0)            # Green for fertility
        }

        # Font
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)

        self.running = True
        self.selected_agent = None
        self.current_agents = []  # Store current agents for mouse click handling
        self.selection_timestamp = 0  # Track when agent was selected

        # Speech bubble controls
        self.speech_mode = SpeechMode.FULL_RESPONSE

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            True if should continue running, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause could be implemented here
                    pass
                elif event.key == pygame.K_s:
                    # Toggle speech bubble mode
                    self._cycle_speech_mode()
                elif event.key == pygame.K_t:
                    # Quick toggle speech on/off
                    self._toggle_speech_on_off()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle agent selection
                self._handle_mouse_click(event.pos)

        return True

    def _handle_mouse_click(self, pos: Tuple[int, int]):
        """Handle mouse clicks for agent selection."""
        x, y = pos
        print(f"🖱️ Mouse click at screen coords: ({x}, {y})")

        # Only handle clicks in the grid area
        if x >= self.grid_width:
            print(
                f"🖱️ Click outside grid area (grid_width: {self.grid_width})")
            return

        # Convert screen coordinates to grid coordinates
        grid_x = x // self.cell_width
        grid_y = y // self.cell_height
        print(
            f"🖱️ Grid coords: ({grid_x}, {grid_y}), cell_size: ({self.cell_width}, {self.cell_height})")
        print(f"🖱️ Available agents: {len(self.current_agents)}")

        # Find agent at this position
        found_agent = None
        for agent in self.current_agents:
            if agent.alive and agent.position == (grid_x, grid_y):
                found_agent = agent
                break

        if found_agent:
            self.selected_agent = found_agent
            # Add a timestamp to track when selection was made
            self.selection_timestamp = time.time()
            print(
                f"🎯 ✅ Selected agent: {found_agent.id} at {found_agent.position}")
        else:
            # Only clear selection if we're sure there's no agent there
            # and it's been at least 0.1 seconds since last selection
            current_time = time.time()
            if not hasattr(self, 'selection_timestamp') or current_time - self.selection_timestamp > 0.1:
                self.selected_agent = None
                print("🎯 ❌ Cleared agent selection - no agent at clicked position")

    def _cycle_speech_mode(self):
        """Cycle through speech bubble display modes."""
        modes = list(SpeechMode)
        current_index = modes.index(self.speech_mode)
        next_index = (current_index + 1) % len(modes)
        self.speech_mode = modes[next_index]
        print(f"💬 Speech mode: {self.speech_mode.value}")

    def _toggle_speech_on_off(self):
        """Quick toggle between OFF and last used mode."""
        if self.speech_mode == SpeechMode.OFF:
            # Turn on to full response mode
            self.speech_mode = SpeechMode.FULL_RESPONSE
        else:
            # Turn off
            self.speech_mode = SpeechMode.OFF
        print(f"💬 Speech bubbles: {self.speech_mode.value}")

    def render(self, agents: List[Agent], stats: Dict[str, Any]):
        """
        Render the current simulation state.

        Args:
            agents: List of all agents
            stats: Simulation statistics
        """
        # Store agents for mouse click handling
        self.current_agents = agents

        # Update selected agent reference if it exists (agents list is recreated each frame)
        if self.selected_agent:
            # Find the same agent in the new list by ID
            updated_agent = None
            for agent in agents:
                if agent.id == self.selected_agent.id and agent.alive:
                    updated_agent = agent
                    break
            self.selected_agent = updated_agent

        # Clear screen
        self.screen.fill(self.colors['background'])

        # Render grid
        self._render_grid()

        # Render world elements
        self._render_world()

        # Render agents
        self._render_agents(agents)

        # Render sidebar
        self._render_sidebar(agents, stats)

        # Update display
        pygame.display.flip()

    def _render_grid(self):
        """Render the grid lines."""
        for x in range(0, self.grid_width, self.cell_width):
            pygame.draw.line(self.screen, self.colors['grid'],
                             (x, 0), (x, self.grid_height))

        for y in range(0, self.grid_height, self.cell_height):
            pygame.draw.line(self.screen, self.colors['grid'],
                             (0, y), (self.grid_width, y))

    def _render_world(self):
        """Render world resources and obstacles."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                cell_type = self.world.get_cell_type_for_display(x, y)

                screen_x = x * self.cell_width
                screen_y = y * self.cell_height
                rect = pygame.Rect(screen_x, screen_y,
                                   self.cell_width, self.cell_height)

                if cell_type == CellType.FOOD:
                    pygame.draw.rect(self.screen, self.colors['food'], rect)
                    # Draw apple symbol
                    center = rect.center
                    pygame.draw.circle(self.screen, (255, 255, 255), center, 5)

                elif cell_type == CellType.WATER:
                    pygame.draw.rect(self.screen, self.colors['water'], rect)
                    # Draw water symbol
                    center = rect.center
                    points = [(center[0]-5, center[1]+3), (center[0],
                                                           center[1]-3), (center[0]+5, center[1]+3)]
                    pygame.draw.polygon(self.screen, (255, 255, 255), points)

                elif cell_type == CellType.OBSTACLE:
                    pygame.draw.rect(
                        self.screen, self.colors['obstacle'], rect)

    def _render_agents(self, agents: List[Agent]):
        """Render all agents with gender-specific appearance."""
        for agent in agents:
            if not agent.alive:
                continue

            x, y = agent.position
            screen_x = x * self.cell_width
            screen_y = y * self.cell_height

            center_x = screen_x + self.cell_width // 2
            center_y = screen_y + self.cell_height // 2

            # Choose color based on gender
            try:
                is_male = (hasattr(agent.gender, 'value') and agent.gender.value == 'MALE') or \
                    (not hasattr(agent.gender, 'value')
                     and str(agent.gender) == 'MALE')
                color = self.colors['agent_male'] if is_male else self.colors['agent_female']
            except AttributeError:
                # Default to male color if gender access fails
                color = self.colors['agent_male']

            # Draw shape based on gender
            try:
                is_male = (hasattr(agent.gender, 'value') and agent.gender.value == 'MALE') or \
                    (not hasattr(agent.gender, 'value')
                     and str(agent.gender) == 'MALE')
                if is_male:
                    # Square for males
                    size = min(self.cell_width, self.cell_height) // 3
                    rect = pygame.Rect(center_x - size//2,
                                       center_y - size//2, size, size)
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    # Circle for females
                    radius = min(self.cell_width, self.cell_height) // 6
                    pygame.draw.circle(self.screen, color,
                                       (center_x, center_y), radius)
            except AttributeError:
                # Default to male shape if gender access fails
                size = min(self.cell_width, self.cell_height) // 3
                rect = pygame.Rect(center_x - size//2,
                                   center_y - size//2, size, size)
                pygame.draw.rect(self.screen, color, rect)

            # Add status indicators
            self._render_agent_status(agent, center_x, center_y)

            # Add speech bubble if agent is speaking
            self._render_speech_bubble(agent, center_x, center_y)

            # Highlight selected agent
            if self.selected_agent == agent:
                pygame.draw.circle(self.screen, (255, 255, 0), (center_x, center_y),
                                   min(self.cell_width, self.cell_height) // 2, 2)

    def _render_agent_status(self, agent: Agent, center_x: int, center_y: int):
        """Render status indicators around agent."""
        offset = 8

        # Pregnancy indicator
        if agent.is_pregnant:
            pygame.draw.circle(self.screen, self.colors['pregnant'],
                               (center_x - offset, center_y - offset), 3)

        # Fertility indicator
        if agent.is_mature and agent.is_fertile:
            pygame.draw.circle(self.screen, self.colors['fertile'],
                               (center_x + offset, center_y - offset), 2)

        # Energy level (small bar)
        if agent.energy < 50:
            bar_width = 10
            bar_height = 2
            bar_x = center_x - bar_width // 2
            bar_y = center_y + offset

            # Background
            pygame.draw.rect(self.screen, (100, 100, 100),
                             (bar_x, bar_y, bar_width, bar_height))

            # Energy level
            energy_width = int(bar_width * (agent.energy / 100))
            energy_color = (255, 0, 0) if agent.energy < 30 else (255, 255, 0)
            pygame.draw.rect(self.screen, energy_color,
                             (bar_x, bar_y, energy_width, bar_height))

    def _render_speech_bubble(self, agent: Agent, center_x: int, center_y: int):
        """Render speech bubble above agent if they have active speech."""
        # Check if speech bubbles are enabled
        if self.speech_mode == SpeechMode.OFF:
            return

        speech_text = agent.get_current_speech()
        if not speech_text:
            return

        # Modify speech text based on mode
        if self.speech_mode == SpeechMode.ACTIONS_ONLY:
            speech_text = self._extract_action_from_speech(speech_text)
            if not speech_text:
                return

        # Speech bubble parameters
        bubble_color = (255, 255, 255)
        text_color = (0, 0, 0)
        border_color = (0, 0, 0)

        # Check if this is an error message (detect by error keywords)
        error_keywords = ['JSON PARSE ERROR', 'JSON DECODE ERROR',
                          'JSON KEY ERROR', 'PARSE ERROR', 'ERROR', 'Fallback:']
        is_error = any(keyword in speech_text for keyword in error_keywords)
        if is_error:
            # Use red background for error bubbles to make them stand out
            bubble_color = (255, 240, 240)  # Light red background
            border_color = (200, 0, 0)      # Red border

        # Render text to get dimensions
        max_width = 350  # Maximum bubble width - increased for full VLM responses
        max_lines = 8    # Maximum number of lines to prevent bubbles getting too tall
        font = self.font_small

        # Simple line splitting for color tag support
        # Split text into lines, preserving color tags
        text_lines = speech_text.split('\n')
        lines = []

        for text_line in text_lines:
            if len(lines) >= max_lines:
                break

            # Check if line is too long and needs wrapping
            import re
            # No need to remove tags anymore since they're not used
            text_width = font.size(text_line)[0]

            if text_width <= max_width:
                lines.append(text_line)
            else:
                # Simple truncation with ellipsis for long lines
                truncated = text_line[:50] + "..."
                lines.append(truncated)

        # Limit total lines
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines:
                lines[-1] += "..."

        # Calculate bubble dimensions
        line_height = font.get_height()
        padding = 6
        bubble_width = min(max_width, max(
            font.size(line)[0] for line in lines)) + padding * 2
        bubble_height = len(lines) * line_height + padding * 2

        # Position bubble above agent
        bubble_x = center_x - bubble_width // 2
        bubble_y = center_y - self.cell_height // 2 - bubble_height - 5

        # Ensure bubble stays within screen bounds
        bubble_x = max(5, min(bubble_x, self.grid_width - bubble_width - 5))
        bubble_y = max(5, bubble_y)

        # Draw bubble background
        bubble_rect = pygame.Rect(
            bubble_x, bubble_y, bubble_width, bubble_height)
        pygame.draw.rect(self.screen, bubble_color, bubble_rect)
        pygame.draw.rect(self.screen, border_color, bubble_rect, 2)

        # Draw speech bubble tail
        tail_size = 8
        tail_points = [
            (center_x - tail_size // 2, bubble_y + bubble_height),
            (center_x + tail_size // 2, bubble_y + bubble_height),
            (center_x, bubble_y + bubble_height + tail_size)
        ]
        pygame.draw.polygon(self.screen, bubble_color, tail_points)
        pygame.draw.polygon(self.screen, border_color, tail_points, 2)

        # Draw text lines with color support
        text_y = bubble_y + padding
        for line in lines:
            # Process color tags in the line
            self._render_colored_text(
                line, bubble_x, text_y, bubble_width, font, text_color)
            text_y += line_height

    def _extract_action_from_speech(self, speech_text: str) -> str:
        """Extract just the action from VLM response speech."""
        import re
        import json

        # First try to parse as JSON
        try:
            json_match = re.search(
                r'\{.*?"command".*?\}', speech_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed.get('command', '')
        except (json.JSONDecodeError, KeyError):
            pass

        # Check for "Action:" format from our formatted responses
        action_pattern = r'Action:\s*([^\n]+)'
        match = re.search(action_pattern, speech_text)
        if match:
            return match.group(1).strip()

        # Look for patterns like "**Choice: MOVE_RIGHT**" or "Choice: MOVE_RIGHT"
        choice_pattern = r'\*\*Choice:\s*([^*\n]+)\*\*'
        match = re.search(choice_pattern, speech_text)
        if match:
            return match.group(1).strip()

        # Fallback: look for "Choice:" without asterisks
        choice_pattern = r'Choice:\s*([^\n]+)'
        match = re.search(choice_pattern, speech_text)
        if match:
            return match.group(1).strip()

        # Last resort: look for action words at the end
        action_words = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT',
                        'PICKUP', 'EAT', 'DRINK', 'WAIT', 'SEEK_MATE', 'BREED']
        for action in action_words:
            if action in speech_text:
                return action

        return ""

    def _render_sidebar(self, agents: List[Agent], stats: Dict[str, Any]):
        """Render the information sidebar."""
        sidebar_x = self.grid_width

        # Sidebar background
        sidebar_rect = pygame.Rect(
            sidebar_x, 0, self.sidebar_width, self.window_size[1])
        pygame.draw.rect(self.screen, self.colors['sidebar'], sidebar_rect)

        y_offset = 5

        # Title
        title = self.font_large.render(
            "AI Arena v2.0", True, self.colors['text'])
        self.screen.blit(title, (sidebar_x + 10, y_offset))
        y_offset += 40

        # Simulation stats
        self._render_text_block(
            f"Turn: {stats.get('turn', 0)}", sidebar_x + 10, y_offset)
        y_offset += 25

        alive_agents = [a for a in agents if a.alive]
        self._render_text_block(
            f"Agents Alive: {len(alive_agents)}", sidebar_x + 10, y_offset)
        y_offset += 20

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
        self._render_text_block(
            f"Males: {males}, Females: {females}", sidebar_x + 10, y_offset)
        y_offset += 20

        mature = len([a for a in alive_agents if a.is_mature])
        pregnant = len([a for a in alive_agents if a.is_pregnant])
        self._render_text_block(
            f"Mature: {mature}, Pregnant: {pregnant}", sidebar_x + 10, y_offset)
        y_offset += 25

        # Population stats
        self._render_text_block(
            f"Births: {stats.get('births', 0)}", sidebar_x + 10, y_offset)
        y_offset += 20
        self._render_text_block(
            f"Deaths: {stats.get('deaths', 0)}", sidebar_x + 10, y_offset)
        y_offset += 20
        self._render_text_block(
            f"Peak Population: {stats.get('peak_population', 0)}", sidebar_x + 10, y_offset)
        y_offset += 30

        # World stats
        world_stats = stats.get('world_stats', {})
        self._render_text_block(
            f"Food: {world_stats.get('food_count', 0)}", sidebar_x + 10, y_offset)
        y_offset += 20
        self._render_text_block(
            f"Water: {world_stats.get('water_count', 0)}", sidebar_x + 10, y_offset)
        y_offset += 30

        # Speech mode controls
        self._render_text_block(
            "=== CONTROLS ===", sidebar_x + 10, y_offset, self.font_medium)
        y_offset += 25
        self._render_text_block(
            f"Speech: {self.speech_mode.value}", sidebar_x + 10, y_offset)
        y_offset += 18
        self._render_text_block("S: Cycle speech mode",
                                sidebar_x + 10, y_offset, font=self.font_small)
        y_offset += 16
        self._render_text_block("T: Toggle speech on/off",
                                sidebar_x + 10, y_offset, font=self.font_small)
        y_offset += 25

        # Selected agent info
        if self.selected_agent:
            self._render_selected_agent_info(
                self.selected_agent, sidebar_x + 10, y_offset)
        else:
            help_text = self.font_small.render(
                "Click an agent for details", True, self.colors['text'])
            self.screen.blit(help_text, (sidebar_x + 10, y_offset))

        # Legend at bottom - moved up for better visibility
        self._render_legend(sidebar_x + 10, self.window_size[1] - 180)

    def _render_selected_agent_info(self, agent: Agent, x: int, y: int):
        """Render detailed info for selected agent."""
        self._render_text_block(f"=== {agent.id} ===", x, y, self.font_medium)
        y += 25

        # Basic info
        try:
            gender_str = agent.gender.value if hasattr(
                agent.gender, 'value') else str(agent.gender)
        except AttributeError:
            gender_str = "Unknown"
        self._render_text_block(f"Gender: {gender_str}", x, y)
        y += 18
        self._render_text_block(f"Age: {agent.age}", x, y)
        y += 18
        self._render_text_block(f"Position: {agent.position}", x, y)
        y += 18

        # Health status
        self._render_text_block(f"Energy: {agent.energy}/100", x, y)
        y += 18
        self._render_text_block(f"Hunger: {agent.hunger}/100", x, y)
        y += 18
        self._render_text_block(f"Thirst: {agent.thirst}/100", x, y)
        y += 18

        # Inventory
        self._render_text_block(
            f"Food: {agent.inventory.get('food', 0)}", x, y)
        y += 18
        self._render_text_block(
            f"Water: {agent.inventory.get('water', 0)}", x, y)
        y += 18

        # Maturity and breeding status
        if agent.is_mature:
            self._render_text_block(f"Mature: Yes", x, y)
            y += 18
            if agent.is_pregnant:
                self._render_text_block(
                    f"Pregnant: {agent.pregnancy_timer}t", x, y)
                y += 18
            elif agent.is_fertile:
                self._render_text_block(f"Fertile: Yes", x, y)
                y += 18
        else:
            self._render_text_block(f"Mature: No", x, y)
            y += 18

        # Children count
        if hasattr(agent, 'children') and agent.children:
            self._render_text_block(f"Children: {len(agent.children)}", x, y)
            y += 18

        # Personality - wrapped for display
        if hasattr(agent, 'personality'):
            personality_lines = self._wrap_text(agent.personality, 25)
            self._render_text_block(f"Personality:", x, y)
            y += 18
            for line in personality_lines[:3]:  # Show max 3 lines
                self._render_text_block(f"  {line}", x, y)
                y += 15

        # Recent actions
        if hasattr(agent, 'get_recent_memories'):
            recent_memories = agent.get_recent_memories(3)
            if recent_memories:
                self._render_text_block(f"Recent Actions:", x, y)
                y += 18
                for memory in recent_memories:
                    action = memory.get('action', 'Unknown')
                    turn = memory.get('turn', '?')
                    self._render_text_block(f"  T{turn}: {action}", x, y)
                    y += 15

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text to multiple lines."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= max_chars:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def _render_legend(self, x: int, y: int):
        """Render the legend."""
        self._render_text_block("=== LEGEND ===", x, y, self.font_medium)
        y += 25

        # Gender symbols
        pygame.draw.rect(
            self.screen, self.colors['agent_male'], (x, y, 12, 12))
        self._render_text_block("Male", x + 20, y + 2, self.font_small)
        y += 18

        pygame.draw.circle(
            self.screen, self.colors['agent_female'], (x + 6, y + 6), 6)
        self._render_text_block("Female", x + 20, y + 2, self.font_small)
        y += 18

        # Status indicators
        pygame.draw.circle(
            self.screen, self.colors['pregnant'], (x + 6, y + 6), 3)
        self._render_text_block("Pregnant", x + 20, y + 2, self.font_small)
        y += 18

        pygame.draw.circle(
            self.screen, self.colors['fertile'], (x + 6, y + 6), 2)
        self._render_text_block("Fertile", x + 20, y + 2, self.font_small)

    def _render_text_block(self, text: str, x: int, y: int, font=None):
        """Helper to render text."""
        if font is None:
            font = self.font_small

        rendered = font.render(text, True, self.colors['text'])
        self.screen.blit(rendered, (x, y))

    def _render_colored_text(self, text: str, x: int, y: int, max_width: int, font, default_color: tuple):
        """Render text centered horizontally."""
        # Simply render the text centered since we don't use color tags anymore
        text_surface = font.render(text, True, default_color)
        text_width = text_surface.get_width()

        # Center the text horizontally
        start_x = x + (max_width - text_width) // 2
        self.screen.blit(text_surface, (start_x, y))

    def tick(self, fps: int = 60):
        """Advance the visualization clock."""
        self.clock.tick(fps)

    def quit(self):
        """Clean up and quit pygame."""
        pygame.quit()
