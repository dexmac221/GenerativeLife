"""
Configuration management and argument parsing.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """Configuration parameters for the simulation."""

    # Grid configuration
    width: int = 25
    height: int = 25

    # Agent configuration
    num_agents: int = 5

    # AI configuration
    model: str = "gemma3:4b"
    server: str = "http://localhost:11434"
    api_type: str = "ollama"
    openai_api_key: str = None

    # Simulation parameters
    survival_mode: str = "normal"
    resource_mode: str = "abundant"
    max_turns: int = 2000
    visualization_speed: int = 3

    # Advanced options
    no_gui: bool = False
    verbose: bool = False

    # AI/VLM parameters
    max_retries: int = 1

    # Breeding parameters
    min_breeding_age: int = 50
    pregnancy_duration: int = 30
    max_population: int = 50

    # Observation parameters
    observation_radius: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "num_agents": self.num_agents,
            "model": self.model,
            "server": self.server,
            "survival_mode": self.survival_mode,
            "resource_mode": self.resource_mode,
            "max_turns": self.max_turns,
            "visualization_speed": self.visualization_speed,
            "no_gui": self.no_gui,
            "verbose": self.verbose,
            "max_retries": self.max_retries,
            "min_breeding_age": self.min_breeding_age,
            "pregnancy_duration": self.pregnancy_duration,
            "max_population": self.max_population,
            "observation_radius": self.observation_radius
        }


def parse_arguments() -> Config:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description='GenerativeLife v2.0: AI-Powered Life Simulation with Generative Agents')

    # Grid configuration
    parser.add_argument('--width', type=int, default=25,
                        help='Grid width (default: 25)')
    parser.add_argument('--height', type=int, default=25,
                        help='Grid height (default: 25)')

    # Agent configuration
    parser.add_argument('--agents', type=int, default=5,
                        help='Number of agents (default: 5)')

    # AI/VLM configuration
    parser.add_argument('--model', type=str, default='gemma3:4b',
                        help='Model to use (default: gemma3:4b for Ollama, gpt-4o for OpenAI)')
    parser.add_argument('--server', type=str, default='http://localhost:11434',
                        help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--api-type', choices=['ollama', 'openai'], default='ollama',
                        help='API type to use (default: ollama)')
    parser.add_argument('--openai-api-key', type=str,
                        help='OpenAI API key (can also set OPENAI_API_KEY env var)')

    # Simulation parameters
    parser.add_argument('--survival-mode', choices=['easy', 'normal', 'hard'], default='normal',
                        help='Survival difficulty (default: normal)')
    parser.add_argument('--resources', choices=['scarce', 'normal', 'abundant'], default='abundant',
                        help='Resource abundance (default: abundant)')
    parser.add_argument('--turns', type=int, default=2000,
                        help='Maximum simulation turns (default: 2000)')
    parser.add_argument('--speed', type=int, default=3,
                        help='Visualization speed/FPS (default: 3)')

    # Advanced options
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without graphical interface')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    # AI/VLM parameters
    parser.add_argument('--max-retries', type=int, default=1,
                        help='Maximum JSON parsing retry attempts (default: 1)')

    # Breeding parameters
    parser.add_argument('--breeding-age', type=int, default=50,
                        help='Minimum age for breeding (default: 50)')
    parser.add_argument('--pregnancy-duration', type=int, default=30,
                        help='Pregnancy duration in turns (default: 30)')
    parser.add_argument('--max-population', type=int, default=50,
                        help='Maximum population limit (default: 50)')

    # Observation parameters
    parser.add_argument('--observation-radius', type=int, default=2,
                        help='Agent observation radius in cells (default: 2)')

    args = parser.parse_args()

    return Config(
        width=args.width,
        height=args.height,
        num_agents=args.agents,
        model=args.model,
        server=args.server,
        api_type=args.api_type,
        openai_api_key=args.openai_api_key,
        survival_mode=args.survival_mode,
        resource_mode=args.resources,
        max_turns=args.turns,
        visualization_speed=args.speed,
        no_gui=args.no_gui,
        verbose=args.verbose,
        max_retries=args.max_retries,
        min_breeding_age=args.breeding_age,
        pregnancy_duration=args.pregnancy_duration,
        max_population=args.max_population,
        observation_radius=args.observation_radius
    )
