"""
Threaded Visualizer - PyGame-based visualization running in a separate thread.
"""

import pygame
import sys
import time
import math
import threading
import queue
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from ..core.agent import Agent, Gender
from ..core.world import World, CellType


class SpeechMode(Enum):
    """Speech bubble display modes."""
    OFF = "Off"
    ACTIONS_ONLY = "Actions Only"
    FULL_RESPONSE = "Full Response"


class ThreadedVisualizer:
    """
    PyGame-based visualizer that runs in its own thread for better event handling.

    Features:
    - Real-time grid visualization
    - Agent status display
    - Gender-specific representations
    - Breeding status indicators
    - Interactive controls
    - Non-blocking event handling
    """

    def __init__(self, world: World, window_size: Tuple[int, int] = (1600, 1000)):
        """
        Initialize the threaded visualizer.

        Args:
            world: The world to visualize
            window_size: Window dimensions (width, height)
        """
        self.world = world
        self.window_size = window_size

        # Thread communication
        self.data_queue = queue.Queue()  # Simulation -> UI
        self.event_queue = queue.Queue()  # UI -> Simulation
        self.running = True
        self.ui_thread = None

        # UI state
        self.selected_agent = None
        self.current_agents = []
        self.current_stats = {}
        self.selection_timestamp = 0

        # Speech bubble controls
        self.speech_mode = SpeechMode.FULL_RESPONSE

        # Animation and performance tracking
        self.animation_time = 0
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_update = time.time()

        # Calculate grid display parameters
        self.sidebar_width = 350  # Increased from 300 to show more info
        self.grid_width = window_size[0] - self.sidebar_width
        self.grid_height = window_size[1]

        self.cell_width = self.grid_width // world.width
        self.cell_height = self.grid_height // world.height

        # Enhanced color scheme
        self.colors = {
            # Backgrounds
            'background': (25, 25, 35),        # Darker blue-gray
            'sidebar': (20, 20, 30),           # Even darker sidebar
            'grid': (60, 60, 80),              # Subtle blue grid

            # World elements with enhanced colors
            'empty': (45, 45, 55),             # Dark gray
            'food': (120, 220, 80),            # Bright green
            'food_glow': (80, 180, 40),        # Darker green for glow
            'water': (80, 160, 255),           # Bright blue
            'water_glow': (40, 120, 215),      # Darker blue for glow
            'obstacle': (100, 100, 110),       # Gray with blue tint
            'obstacle_shadow': (60, 60, 70),   # Shadow for 3D effect

            # Agents with better colors
            'agent_male': (255, 120, 120),     # Soft red
            'agent_male_border': (200, 80, 80),  # Darker red border
            'agent_female': (255, 120, 255),   # Soft magenta
            'agent_female_border': (200, 80, 200),  # Darker magenta border

            # Status indicators
            'pregnant': (255, 220, 60),        # Golden yellow
            'fertile': (100, 255, 100),        # Bright green
            'low_energy': (255, 60, 60),       # Red warning
            'selected': (255, 255, 100),       # Bright yellow
            'selected_glow': (255, 255, 60),   # Yellow glow

            # UI elements
            'text': (240, 240, 250),           # Off-white
            'text_highlight': (255, 255, 120),  # Yellow highlight
            'text_dim': (180, 180, 190),       # Dimmed text
            'panel_border': (80, 80, 100),     # Panel borders
        }

    def start(self):
        """Start the UI thread."""
        self.ui_thread = threading.Thread(
            target=self._ui_thread_main, daemon=True)
        self.ui_thread.start()
        print("🎮 Threaded visualizer started")

    def stop(self):
        """Stop the UI thread."""
        self.running = False
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1.0)
        pygame.quit()

    def update(self, agents: List[Agent], stats: Dict[str, Any]):
        """
        Update the visualizer with new simulation data.

        Args:
            agents: List of all agents
            stats: Simulation statistics
        """
        try:
            # Send data to UI thread (non-blocking)
            self.data_queue.put_nowait({
                'agents': agents,
                'stats': stats,
                'timestamp': time.time()
            })
        except queue.Full:
            # Skip this frame if queue is full
            pass

    def get_events(self) -> List[Dict[str, Any]]:
        """
        Get events from the UI thread.

        Returns:
            List of events (e.g., agent selections, quit signals)
        """
        events = []
        try:
            while True:
                event = self.event_queue.get_nowait()
                events.append(event)
        except queue.Empty:
            pass
        return events

    def _ui_thread_main(self):
        """Main UI thread function."""
        # Initialize pygame in this thread
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(
            "GenerativeLife - AI-Powered Life Simulation")
        self.clock = pygame.time.Clock()

        # Font initialization - smaller fonts to fit more info
        self.font_small = pygame.font.Font(None, 16)    # Reduced from 20
        self.font_medium = pygame.font.Font(None, 20)   # Reduced from 24
        self.font_large = pygame.font.Font(None, 26)    # Reduced from 32

        print("🎮 UI thread initialized")

        while self.running:
            current_time = time.time()

            # Update animation time and FPS
            self.animation_time = current_time
            self.frame_count += 1

            if current_time - self.last_fps_update >= 1.0:
                self.fps_counter = self.frame_count
                self.frame_count = 0
                self.last_fps_update = current_time

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._send_event({'type': 'quit'})
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._send_event({'type': 'quit'})
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self._send_event({'type': 'pause_toggle'})
                    elif event.key == pygame.K_s:
                        # Toggle speech bubble mode
                        self._cycle_speech_mode()
                    elif event.key == pygame.K_t:
                        # Quick toggle speech on/off
                        self._toggle_speech_on_off()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos)
                    self._handle_mouse_click(event.pos)

            # Update data from simulation
            self._update_from_simulation()

            # Render
            self._render()

            # Control frame rate
            self.clock.tick(60)  # UI runs at 60 FPS

        pygame.quit()

    def _send_event(self, event: Dict[str, Any]):
        """Send event to simulation thread."""
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _update_from_simulation(self):
        """Update UI state from simulation data."""
        try:
            # Get the latest data (discard older frames)
            latest_data = None
            while True:
                try:
                    latest_data = self.data_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_data:
                self.current_agents = latest_data['agents']
                self.current_stats = latest_data['stats']

                # Update selected agent reference if it exists
                if self.selected_agent:
                    updated_agent = None
                    for agent in self.current_agents:
                        if agent.id == self.selected_agent.id and agent.alive:
                            updated_agent = agent
                            break
                    self.selected_agent = updated_agent

        except Exception as e:
            print(f"Error updating from simulation: {e}")

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
            self.selection_timestamp = time.time()
            print(
                f"🎯 ✅ Selected agent: {found_agent.id} at {found_agent.position}")

            # Send selection event to simulation
            self._send_event({
                'type': 'agent_selected',
                'agent_id': found_agent.id,
                'position': found_agent.position
            })
        else:
            # Only clear selection if it's been a reasonable time since last selection
            current_time = time.time()
            if current_time - self.selection_timestamp > 0.5:
                self.selected_agent = None
                print("🎯 ❌ Cleared agent selection - no agent at clicked position")

                # Send deselection event
                self._send_event({
                    'type': 'agent_deselected'
                })

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

    def _render(self):
        """Render the current state."""
        # Clear screen
        self.screen.fill(self.colors['background'])

        # Render grid
        self._render_grid()

        # Render world elements
        self._render_world()

        # Render agents
        self._render_agents(self.current_agents)

        # Render sidebar
        self._render_sidebar(self.current_agents, self.current_stats)

        # Update display
        pygame.display.flip()

    def _render_grid(self):
        """Render enhanced grid lines with subtle styling."""
        # Main grid lines
        for x in range(0, self.grid_width, self.cell_width):
            pygame.draw.line(self.screen, self.colors['grid'],
                             (x, 0), (x, self.grid_height), 1)

        for y in range(0, self.grid_height, self.cell_height):
            pygame.draw.line(self.screen, self.colors['grid'],
                             (0, y), (self.grid_width, y), 1)

        # Add subtle intersection points for better visual structure
        for x in range(0, self.grid_width, self.cell_width):
            for y in range(0, self.grid_height, self.cell_height):
                if x % (self.cell_width * 5) == 0 and y % (self.cell_height * 5) == 0:
                    # Highlight every 5th intersection
                    pygame.draw.circle(self.screen, (80, 80, 100), (x, y), 1)

    def _render_world(self):
        """Render world resources and obstacles with enhanced graphics."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                cell_type = self.world.get_cell_type(x, y)
                screen_x = x * self.cell_width
                screen_y = y * self.cell_height
                rect = pygame.Rect(screen_x, screen_y,
                                   self.cell_width, self.cell_height)

                if cell_type == CellType.FOOD:
                    # Food with gradient effect
                    pygame.draw.rect(self.screen, self.colors['food'], rect)
                    # Add inner glow
                    inner_rect = pygame.Rect(screen_x + 2, screen_y + 2,
                                             self.cell_width - 4, self.cell_height - 4)
                    pygame.draw.rect(
                        self.screen, self.colors['food_glow'], inner_rect)
                    # Add small sparkle effect
                    center_x = screen_x + self.cell_width // 2
                    center_y = screen_y + self.cell_height // 2
                    pygame.draw.circle(
                        self.screen, (255, 255, 255), (center_x, center_y), 2)

                elif cell_type == CellType.WATER:
                    # Water with animated wave-like effect
                    pygame.draw.rect(self.screen, self.colors['water'], rect)
                    # Add darker border for depth
                    pygame.draw.rect(
                        self.screen, self.colors['water_glow'], rect, 2)
                    # Add animated ripple effect
                    center_x = screen_x + self.cell_width // 2
                    center_y = screen_y + self.cell_height // 2

                    # Animated ripples based on time and position
                    wave_offset = (self.animation_time * 2) + (x + y) * 0.5
                    for i, base_radius in enumerate([3, 6, 9]):
                        radius = base_radius + \
                            int(2 * abs(math.sin(wave_offset + i)))
                        alpha = max(20, 100 - (i * 30) -
                                    int(20 * abs(math.sin(wave_offset))))
                        if radius < min(self.cell_width, self.cell_height) // 2:
                            color = (*self.colors['water_glow'], alpha)
                            # Create a surface for alpha blending
                            s = pygame.Surface(
                                (radius * 2, radius * 2), pygame.SRCALPHA)
                            pygame.draw.circle(
                                s, color, (radius, radius), radius, 1)
                            self.screen.blit(
                                s, (center_x - radius, center_y - radius))

                elif cell_type == CellType.OBSTACLE:
                    # 3D-looking obstacles
                    # Base shadow
                    shadow_rect = pygame.Rect(screen_x + 2, screen_y + 2,
                                              self.cell_width, self.cell_height)
                    pygame.draw.rect(
                        self.screen, self.colors['obstacle_shadow'], shadow_rect)
                    # Main obstacle
                    pygame.draw.rect(
                        self.screen, self.colors['obstacle'], rect)
                    # Highlight for 3D effect
                    highlight_rect = pygame.Rect(screen_x, screen_y,
                                                 self.cell_width - 2, self.cell_height - 2)
                    pygame.draw.rect(
                        self.screen, (140, 140, 150), highlight_rect, 2)

    def _render_agents(self, agents: List[Agent]):
        """Render all agents with enhanced graphics."""
        for agent in agents:
            if not agent.alive:
                continue

            x, y = agent.position
            screen_x = x * self.cell_width
            screen_y = y * self.cell_height
            center_x = screen_x + self.cell_width // 2
            center_y = screen_y + self.cell_height // 2

            # Choose colors based on gender - with safety checks
            try:
                is_male = (agent.gender == Gender.MALE)
            except (AttributeError, TypeError):
                is_male = True  # Default to male color if gender is invalid

            main_color = self.colors['agent_male'] if is_male else self.colors['agent_female']
            border_color = self.colors['agent_male_border'] if is_male else self.colors['agent_female_border']

            # Calculate size based on energy (agents get smaller when low energy)
            # Slightly higher minimum
            energy_factor = max(0.7, agent.energy / 100.0)
            # Make them even larger (was // 2)
            base_size = min(self.cell_width, self.cell_height) // 1.5

            # Shape based on gender - with safety check
            try:
                is_male = (agent.gender == Gender.MALE)
            except (AttributeError, TypeError):
                is_male = True  # Default to male shape

            if is_male:
                # Enhanced squares for males with energy-based sizing and 3D effect
                size = int(base_size * energy_factor)

                # Shadow for 3D effect
                shadow_rect = pygame.Rect(
                    center_x - size//2 + 1, center_y - size//2 + 1, size, size)
                pygame.draw.rect(self.screen, (80, 80, 80), shadow_rect)

                # Main body
                main_rect = pygame.Rect(
                    center_x - size//2, center_y - size//2, size, size)
                pygame.draw.rect(self.screen, main_color, main_rect)

                # Border
                pygame.draw.rect(self.screen, border_color, main_rect, 2)

                # Inner highlight for 3D effect
                highlight_rect = pygame.Rect(center_x - size//2 + 2, center_y - size//2 + 2,
                                             max(1, size - 4), max(1, size - 4))
                if highlight_rect.width > 0 and highlight_rect.height > 0:
                    pygame.draw.rect(
                        self.screen, (255, 255, 255, 80), highlight_rect, 1)

            else:
                # Enhanced circles for females with energy-based sizing and glow effect
                radius = int(
                    (min(self.cell_width, self.cell_height) // 3.5) * energy_factor)  # Make female agents larger (was // 6)

                # Outer glow
                for i in range(3):
                    # Increased glow range for larger agents
                    glow_radius = radius + (5 - i)
                    alpha = 40 - (i * 10)
                    glow_surface = pygame.Surface(
                        (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surface, (*main_color, alpha),
                                       (glow_radius, glow_radius), glow_radius)
                    self.screen.blit(
                        glow_surface, (center_x - glow_radius, center_y - glow_radius))

                # Main body
                pygame.draw.circle(self.screen, main_color,
                                   (center_x, center_y), radius)

                # Border
                pygame.draw.circle(self.screen, border_color,
                                   (center_x, center_y), radius, 2)

                # Inner highlight
                if radius > 3:
                    highlight_pos = (center_x - radius//3,
                                     center_y - radius//3)
                    pygame.draw.circle(
                        self.screen, (255, 255, 255, 120), highlight_pos, radius//3)

            # Add status indicators
            self._render_agent_status(agent, center_x, center_y)

            # Add speech bubble if agent is speaking
            self._render_speech_bubble(agent, center_x, center_y)

            # Enhanced selection highlight
            if self.selected_agent and self.selected_agent.id == agent.id:
                # Pulsing selection effect
                # Make selection highlight larger to match bigger agents
                pulse_radius = min(self.cell_width, self.cell_height) // 1.3
                for i in range(3):
                    alpha = 120 - (i * 30)
                    selection_surface = pygame.Surface(
                        (pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(selection_surface, (*self.colors['selected'], alpha),
                                       (pulse_radius, pulse_radius), pulse_radius - i, 2)
                    self.screen.blit(
                        selection_surface, (center_x - pulse_radius, center_y - pulse_radius))

    def _render_agent_status(self, agent: Agent, center_x: int, center_y: int):
        """Render enhanced status indicators around agent."""
        offset = 10
        indicator_size = 4

        # Pregnancy indicator - golden heart shape
        if agent.is_pregnant:
            # Create a small heart-like shape
            heart_points = [
                (center_x + offset - 2, center_y - offset),
                (center_x + offset + 2, center_y - offset),
                (center_x + offset, center_y - offset + 3)
            ]
            pygame.draw.polygon(
                self.screen, self.colors['pregnant'], heart_points)
            # Add glow
            glow_surface = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface, (*self.colors['pregnant'], 80), (4, 4), 4)
            self.screen.blit(
                glow_surface, (center_x + offset - 4, center_y - offset - 4))

        # Fertility indicator - green plus sign
        if agent.is_fertile and not agent.is_pregnant:
            # Plus sign for fertility
            pygame.draw.rect(self.screen, self.colors['fertile'],
                             (center_x - offset - 1, center_y - offset - 2, 3, indicator_size))
            pygame.draw.rect(self.screen, self.colors['fertile'],
                             (center_x - offset - 2, center_y - offset - 1, indicator_size, 3))

        # Energy level indicator - colored bar
        if agent.energy < 70:  # Show for medium to low energy
            bar_width = 8
            bar_height = 2
            bar_x = center_x - bar_width // 2
            bar_y = center_y + offset

            # Background bar
            pygame.draw.rect(self.screen, (60, 60, 60),
                             (bar_x, bar_y, bar_width, bar_height))

            # Energy bar with color based on level
            energy_width = int((agent.energy / 100.0) * bar_width)
            if agent.energy > 50:
                energy_color = (100, 255, 100)  # Green
            elif agent.energy > 25:
                energy_color = (255, 255, 100)  # Yellow
            else:
                energy_color = self.colors['low_energy']  # Red

            if energy_width > 0:
                pygame.draw.rect(self.screen, energy_color,
                                 (bar_x, bar_y, energy_width, bar_height))

        # Hunger/Thirst warning indicators
        if agent.hunger > 80:  # High hunger
            # Red triangle pointing down (empty stomach)
            triangle_points = [
                (center_x - offset//2, center_y + offset + 6),
                (center_x + offset//2, center_y + offset + 6),
                (center_x, center_y + offset + 10)
            ]
            pygame.draw.polygon(self.screen, (255, 100, 100), triangle_points)

        if agent.thirst > 80:  # High thirst
            # Blue droplet shape
            droplet_points = [
                (center_x + offset, center_y + offset + 6),
                (center_x + offset - 2, center_y + offset + 8),
                (center_x + offset + 2, center_y + offset + 8),
                (center_x + offset, center_y + offset + 10)
            ]
            pygame.draw.polygon(self.screen, (100, 150, 255), droplet_points)

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

        # Word wrap the text
        words = speech_text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_width = font.size(test_line)[0]

            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    # Check if we've hit the max lines limit
                    if len(lines) >= max_lines:
                        # Add ellipsis to indicate truncation
                        lines[-1] += "..."
                        break
                current_line = word

        if current_line and len(lines) < max_lines:
            lines.append(current_line)
        elif len(lines) >= max_lines and current_line:
            # Truncate last line if needed
            if not lines[-1].endswith("..."):
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

        # Draw text lines
        text_y = bubble_y + padding
        for line in lines:
            text_surface = font.render(line, True, text_color)
            text_x = bubble_x + (bubble_width - text_surface.get_width()) // 2
            self.screen.blit(text_surface, (text_x, text_y))
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
        """Render the enhanced information sidebar."""
        sidebar_x = self.grid_width
        sidebar_rect = pygame.Rect(
            sidebar_x, 0, self.sidebar_width, self.window_size[1])

        # Sidebar background with gradient effect
        pygame.draw.rect(self.screen, self.colors['sidebar'], sidebar_rect)

        # Add border
        pygame.draw.rect(
            self.screen, self.colors['panel_border'], sidebar_rect, 2)

        y_offset = 10

        # Enhanced title with glow effect
        title_text = self.font_large.render(
            "GenerativeLife", True, self.colors['text_highlight'])
        title_shadow = self.font_large.render(
            "GenerativeLife", True, (100, 100, 120))
        self.screen.blit(title_shadow, (sidebar_x +
                         12, y_offset + 2))  # Shadow
        self.screen.blit(title_text, (sidebar_x + 10, y_offset))

        # Subtitle
        subtitle_text = self.font_small.render(
            "Threaded Version", True, self.colors['text_dim'])
        self.screen.blit(subtitle_text, (sidebar_x + 10, y_offset + 30))
        y_offset += 60

        # Population stats with enhanced styling
        alive_agents = [a for a in agents if a.alive]
        males = len([a for a in alive_agents if a.gender == Gender.MALE])
        females = len([a for a in alive_agents if a.gender == Gender.FEMALE])

        # Population section header
        self._render_section_header("Population", sidebar_x + 10, y_offset)
        y_offset += 22  # Reduced from 25

        # Population with color coding
        total_color = self.colors['text_highlight'] if len(
            alive_agents) > 0 else self.colors['low_energy']
        self._render_stat_line("Total:", str(
            len(alive_agents)), sidebar_x + 15, y_offset, total_color)
        y_offset += 17  # Reduced from 20

        self._render_stat_line("Males:", str(
            males), sidebar_x + 15, y_offset, self.colors['agent_male'])
        y_offset += 17

        self._render_stat_line("Females:", str(
            females), sidebar_x + 15, y_offset, self.colors['agent_female'])
        y_offset += 25  # Reduced from 30

        # Simulation stats section
        world_stats = stats.get('world', {})

        self._render_section_header("Simulation", sidebar_x + 10, y_offset)
        y_offset += 22  # Reduced from 25

        self._render_stat_line("Turn:", str(
            stats.get('turn', 0)), sidebar_x + 15, y_offset)
        y_offset += 17  # Reduced from 20

        food_count = world_stats.get('food_count', 0)
        food_color = self.colors['food'] if food_count > 20 else self.colors['low_energy']
        self._render_stat_line("Food:", str(food_count),
                               sidebar_x + 15, y_offset, food_color)
        y_offset += 17

        water_count = world_stats.get('water_count', 0)
        water_color = self.colors['water'] if water_count > 15 else self.colors['low_energy']
        self._render_stat_line("Water:", str(
            water_count), sidebar_x + 15, y_offset, water_color)
        y_offset += 25  # Reduced from 30

        # Performance stats section
        self._render_section_header("Performance", sidebar_x + 10, y_offset)
        y_offset += 22  # Reduced from 25

        fps_color = self.colors['text_highlight'] if self.fps_counter >= 30 else self.colors['low_energy']
        self._render_stat_line("UI FPS:", str(
            self.fps_counter), sidebar_x + 15, y_offset, fps_color)
        y_offset += 17  # Reduced from 20

        sim_fps = stats.get('simulation_fps', 0)
        sim_fps_color = self.colors['text_highlight'] if sim_fps >= 2 else self.colors['low_energy']
        self._render_stat_line(
            "Sim FPS:", f"{sim_fps:.1f}", sidebar_x + 15, y_offset, sim_fps_color)
        y_offset += 25  # Reduced from 30

        # Controls section
        self._render_section_header("Controls", sidebar_x + 10, y_offset)
        y_offset += 22
        self._render_stat_line(
            "Speech:", self.speech_mode.value, sidebar_x + 15, y_offset)
        y_offset += 17
        help_text = self.font_small.render(
            "S: Cycle speech mode", True, self.colors['text_dim'])
        self.screen.blit(help_text, (sidebar_x + 15, y_offset))
        y_offset += 16
        help_text = self.font_small.render(
            "T: Toggle speech on/off", True, self.colors['text_dim'])
        self.screen.blit(help_text, (sidebar_x + 15, y_offset))
        y_offset += 25

        # Selected agent info section
        if self.selected_agent:
            self._render_section_header(
                "Agent Details", sidebar_x + 10, y_offset)
            y_offset += 25
            self._render_selected_agent_info(
                self.selected_agent, sidebar_x + 10, y_offset)
        else:
            self._render_section_header(
                "Instructions", sidebar_x + 10, y_offset)
            y_offset += 25
            help_text = self.font_small.render(
                "Click an agent for details", True, self.colors['text_dim'])
            self.screen.blit(help_text, (sidebar_x + 15, y_offset))
            y_offset += 20
            help_text2 = self.font_small.render(
                "ESC to quit, SPACE to pause", True, self.colors['text_dim'])
            self.screen.blit(help_text2, (sidebar_x + 15, y_offset))

        # Legend at bottom - moved up for better visibility
        self._render_legend(sidebar_x + 10, self.window_size[1] - 200)

    def _render_section_header(self, title: str, x: int, y: int):
        """Render a section header with enhanced styling."""
        # Background bar
        header_rect = pygame.Rect(x - 5, y - 2, self.sidebar_width - 20, 20)
        pygame.draw.rect(self.screen, self.colors['panel_border'], header_rect)

        # Header text
        header_text = self.font_medium.render(
            title, True, self.colors['text_highlight'])
        self.screen.blit(header_text, (x, y))

    def _render_stat_line(self, label: str, value: str, x: int, y: int, value_color=None):
        """Render a stat line with label and colored value."""
        if value_color is None:
            value_color = self.colors['text']

        # Label
        label_text = self.font_small.render(label, True, self.colors['text'])
        self.screen.blit(label_text, (x, y))

        # Value (right-aligned within reasonable space)
        value_text = self.font_small.render(value, True, value_color)
        value_x = x + 120  # Fixed position for alignment
        self.screen.blit(value_text, (value_x, y))

    def _render_selected_agent_info(self, agent: Agent, x: int, y: int):
        """Render detailed info for selected agent."""
        self._render_text_block(f"=== {agent.id} ===", x, y, self.font_medium)
        y += 22

        # Basic info with reduced spacing - safe gender access
        try:
            gender_str = agent.gender.value if hasattr(
                agent.gender, 'value') else str(agent.gender)
        except AttributeError:
            gender_str = "UNKNOWN"
        self._render_text_block(f"Gender: {gender_str}", x, y)
        y += 15  # Reduced from 18
        self._render_text_block(f"Age: {agent.age}", x, y)
        y += 15
        self._render_text_block(f"Position: {agent.position}", x, y)
        y += 15

        # Health status
        self._render_text_block(f"Energy: {agent.energy}/100", x, y)
        y += 15
        self._render_text_block(f"Hunger: {agent.hunger}/100", x, y)
        y += 15
        self._render_text_block(f"Thirst: {agent.thirst}/100", x, y)
        y += 15

        # Inventory
        self._render_text_block(
            f"Food: {agent.inventory.get('food', 0)}", x, y)
        y += 15
        self._render_text_block(
            f"Water: {agent.inventory.get('water', 0)}", x, y)
        y += 15

        # Maturity and breeding status
        if agent.is_mature:
            self._render_text_block(f"Mature: Yes", x, y)
            y += 15
            if agent.is_pregnant:
                self._render_text_block(
                    f"Pregnant: {agent.pregnancy_timer}t", x, y)
                y += 15
            elif agent.is_fertile:
                self._render_text_block(f"Fertile: Yes", x, y)
                y += 15
        else:
            self._render_text_block(f"Mature: No", x, y)
            y += 15

        # Children count
        if hasattr(agent, 'children') and agent.children:
            self._render_text_block(f"Children: {len(agent.children)}", x, y)
            y += 15

        # Personality - wrapped for display
        if hasattr(agent, 'personality'):
            personality_lines = self._wrap_text(
                agent.personality, 30)  # Increased from 25
            self._render_text_block(f"Personality:", x, y)
            y += 15
            for line in personality_lines[:3]:  # Show max 3 lines
                self._render_text_block(f"  {line}", x, y)
                y += 13  # Reduced from 15

        # Recent actions
        if hasattr(agent, 'get_recent_memories'):
            recent_memories = agent.get_recent_memories(3)
            if recent_memories:
                self._render_text_block(f"Recent Actions:", x, y)
                y += 15
                for memory in recent_memories:
                    action = memory.get('action', 'Unknown')
                    turn = memory.get('turn', '?')
                    self._render_text_block(f"  T{turn}: {action}", x, y)
                    y += 13  # Reduced from 15

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text to multiple lines."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def _render_text_block(self, text: str, x: int, y: int, font=None):
        """Render text block."""
        if font is None:
            font = self.font_small
        text_surface = font.render(text, True, self.colors['text'])
        self.screen.blit(text_surface, (x, y))

    def _render_legend(self, x: int, y: int):
        """Render enhanced legend with better graphics."""
        # Legend header
        self._render_section_header("Legend", x, y)
        y += 30

        # Agent types with enhanced graphics
        # Male agent
        size = 10
        male_rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, self.colors['agent_male'], male_rect)
        pygame.draw.rect(
            self.screen, self.colors['agent_male_border'], male_rect, 2)
        self._render_text_block("Male Agent", x + 20, y - 2)
        y += 20

        # Female agent
        pygame.draw.circle(
            self.screen, self.colors['agent_female'], (x + size//2, y + size//2), size//2)
        pygame.draw.circle(
            self.screen, self.colors['agent_female_border'], (x + size//2, y + size//2), size//2, 2)
        self._render_text_block("Female Agent", x + 20, y - 2)
        y += 20

        # Resources with enhanced graphics
        # Food
        food_rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, self.colors['food'], food_rect)
        pygame.draw.rect(self.screen, self.colors['food_glow'],
                         pygame.Rect(x + 2, y + 2, size - 4, size - 4))
        pygame.draw.circle(self.screen, (255, 255, 255),
                           (x + size//2, y + size//2), 1)
        self._render_text_block("Food", x + 20, y - 2)
        y += 20

        # Water
        water_rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, self.colors['water'], water_rect)
        pygame.draw.rect(self.screen, self.colors['water_glow'], water_rect, 2)
        self._render_text_block("Water", x + 20, y - 2)
        y += 20

        # Obstacle
        obstacle_rect = pygame.Rect(x, y, size, size)
        shadow_rect = pygame.Rect(x + 1, y + 1, size, size)
        pygame.draw.rect(
            self.screen, self.colors['obstacle_shadow'], shadow_rect)
        pygame.draw.rect(self.screen, self.colors['obstacle'], obstacle_rect)
        self._render_text_block("Obstacle", x + 20, y - 2)
        y += 25

        # Status indicators
        # Pregnancy
        heart_points = [(x - 1, y), (x + 3, y), (x + 1, y + 3)]
        pygame.draw.polygon(self.screen, self.colors['pregnant'], heart_points)
        self._render_text_block("Pregnant", x + 20, y - 2)
        y += 18

        # Fertility
        pygame.draw.rect(self.screen, self.colors['fertile'], (x, y - 1, 2, 5))
        pygame.draw.rect(self.screen, self.colors['fertile'], (x - 1, y, 5, 2))
        self._render_text_block("Fertile", x + 20, y - 2)
        y += 18

        # Low energy
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, 8, 2))
        pygame.draw.rect(self.screen, self.colors['low_energy'], (x, y, 2, 2))
        self._render_text_block("Low Energy", x + 20, y - 2)
        y += 18

    def _render_colored_text(self, text: str, x: int, y: int, max_width: int, font, default_color: tuple):
        """Render text centered horizontally."""
        # Simply render the text centered since we don't use color tags anymore
        text_surface = font.render(text, True, default_color)
        text_width = text_surface.get_width()

        # Center the text horizontally
        start_x = x + (max_width - text_width) // 2
        self.screen.blit(text_surface, (start_x, y))
