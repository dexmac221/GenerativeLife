#!/usr/bin/env python3
"""
GenerativeLife v2.0 - AI-Powered Artificial Life Simulation
Main entry point for the generative AI life simulation.
"""

from modular.utils.config import parse_arguments
from modular.ui.threaded_visualizer import ThreadedVisualizer
from modular.ai.vlm_controller import VLMController
from modular.core.simulation import Simulation
from modular.core.world import World
import sys
import time
import os
from typing import Optional

# Add the modular package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class GenerativeLifeRunner:
    """
    Main runner for the GenerativeLife simulation with threaded UI.

    Orchestrates all components and handles the main simulation loop.
    """

    def __init__(self, config):
        """Initialize the arena runner with configuration."""
        self.config = config

        # Initialize components
        print("🚀 Initializing GenerativeLife v2.0 (AI-Powered Simulation)...")

        # Create world
        self.world = World(
            width=config.width,
            height=config.height,
            resource_mode=config.resource_mode
        )

        # Create AI controller with data collection enabled
        self.ai_controller = VLMController(
            model=config.model,
            server=config.server,
            api_type=config.api_type,
            openai_api_key=config.openai_api_key,
            enable_data_collection=True,  # Enable LoRA training data collection
            compact_prompts=True  # Use compact prompts to reduce token usage
        )

        # Create simulation
        self.simulation = Simulation(
            world=self.world,
            ai_controller=self.ai_controller,
            config=config.to_dict()
        )

        # Create threaded visualizer (if GUI enabled)
        self.visualizer: Optional[ThreadedVisualizer] = None
        if not config.no_gui:
            self.visualizer = ThreadedVisualizer(self.world)

        print(f"📐 Grid: {config.width}x{config.height}")
        print(f"🧠 Model: {config.model} @ {config.server}")
        print(
            f"⚙️ Survival: {config.survival_mode}, Resources: {config.resource_mode}")
        print(
            f"🎮 Max turns: {config.max_turns}, Speed: {config.visualization_speed} FPS")

        # Create initial agents
        self._create_initial_agents()

    def _create_initial_agents(self):
        """Create the initial population of agents."""
        print(f"🏗️ Creating {self.config.num_agents} agents...")

        for i in range(self.config.num_agents):
            agent = self.simulation.create_agent()

        print(f"🎯 Total agents created: {len(self.simulation.agents)}")

    def run(self):
        """Run the main simulation loop."""
        self.simulation.running = True

        try:
            if self.config.no_gui:
                self._run_headless()
            else:
                self._run_with_threaded_gui()
        except KeyboardInterrupt:
            print("\n⏹️ Simulation interrupted by user")
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
            raise
        finally:
            self._cleanup()

    def _run_headless(self):
        """Run simulation without GUI."""
        print("\n🚀 Starting headless simulation...")

        for turn in range(self.config.max_turns):
            if not self.simulation.agents:
                print("❌ All agents died. Simulation ended.")
                break

            self.simulation.step()

            # Print periodic status
            if turn % 50 == 0:
                stats = self.simulation.get_statistics()
                alive = len([a for a in self.simulation.agents if a.alive])
                print(f"📊 Turn {turn}: {alive} agents alive, "
                      f"{stats['births']} births, {stats['deaths']} deaths")

        # Final statistics
        self._print_final_stats()

    def _run_with_threaded_gui(self):
        """Run simulation with threaded GUI."""
        print("\n🎮 Starting simulation with threaded GUI...")
        print("Press ESC to quit, click agents for details")

        # Start the UI thread
        self.visualizer.start()

        turn = 0
        last_frame_time = time.time()
        target_frame_time = 1.0 / self.config.visualization_speed

        paused = False

        while turn < self.config.max_turns and self.simulation.running:
            current_time = time.time()

            # Handle events from UI thread
            events = self.visualizer.get_events()
            for event in events:
                if event['type'] == 'quit':
                    print("🛑 Quit requested from UI")
                    self.simulation.running = False
                    break
                elif event['type'] == 'pause_toggle':
                    paused = not paused
                    print(f"⏸️ Simulation {'paused' if paused else 'resumed'}")
                elif event['type'] == 'agent_selected':
                    print(
                        f"🎯 Agent selected: {event['agent_id']} at {event['position']}")
                elif event['type'] == 'agent_deselected':
                    print("🎯 Agent deselected")

            if not self.simulation.running:
                break

            # Check if simulation should continue
            if not self.simulation.agents:
                print("❌ All agents died. Simulation ended.")
                break

            # Run simulation step (if not paused)
            if not paused:
                self.simulation.step()
                turn += 1

            # Send data to UI thread
            stats = self.simulation.get_statistics()
            self.visualizer.update(self.simulation.agents, stats)

            # Control simulation speed
            elapsed = current_time - last_frame_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            last_frame_time = time.time()

            # Print periodic status
            if turn % 100 == 0 and turn > 0:
                alive = len([a for a in self.simulation.agents if a.alive])
                print(f"📊 Turn {turn}: {alive} agents alive")

        # Final statistics
        self._print_final_stats()

    def _print_final_stats(self):
        """Print final simulation statistics."""
        stats = self.simulation.get_statistics()
        ai_stats = self.ai_controller.get_performance_stats()

        print("\n" + "="*50)
        print("🎯 FINAL SIMULATION STATISTICS")
        print("="*50)
        print(f"Total Turns: {stats['turn']}")
        print(f"Final Population: {stats['alive_agents']}")
        print(f"Peak Population: {stats['peak_population']}")
        print(f"Total Births: {stats['births']}")
        print(f"Total Deaths: {stats['deaths']}")
        print(f"Breeding Events: {stats['breeding_events']}")
        print(f"Resource Pickups: {stats['resource_pickups']}")

        print(f"\n🧠 AI PERFORMANCE:")
        print(f"Total Inferences: {ai_stats['total_inferences']}")
        print(f"Average Time: {ai_stats['average_time']:.3f}s")
        print(f"Inferences/sec: {ai_stats['inferences_per_second']:.2f}")

        # Survival analysis
        alive_agents = [a for a in self.simulation.agents if a.alive]
        if alive_agents:
            avg_age = sum(a.age for a in alive_agents) / len(alive_agents)
            max_age = max(a.age for a in alive_agents)
            print(f"\n👥 SURVIVOR ANALYSIS:")
            print(f"Average Age: {avg_age:.1f}")
            print(f"Oldest Agent: {max_age}")

            # Safe gender counting
            males = 0
            females = 0
            for a in alive_agents:
                try:
                    if hasattr(a.gender, 'value'):
                        gender_val = a.gender.value
                    else:
                        gender_val = str(a.gender)

                    if gender_val == 'MALE':
                        males += 1
                    elif gender_val == 'FEMALE':
                        females += 1
                except AttributeError as e:
                    print(f"⚠️ Gender error for agent {a.id}: {e}")

            print(f"Males: {males}")
            print(f"Females: {females}")

        print("="*50)

    def _cleanup(self):
        """Clean up resources."""
        if self.visualizer:
            self.visualizer.stop()

        # Finalize data collection for LoRA training
        print(f"\n📊 TRAINING DATA COLLECTION:")
        data_stats = self.simulation.finalize_data_collection()
        if data_stats:
            print("✅ Training data saved successfully!")
        else:
            print("ℹ️ No training data was collected this session.")

        print("🧹 Cleanup complete")


def main():
    """Main entry point."""
    try:
        # Parse configuration
        config = parse_arguments()

        # Create and run GenerativeLife
        arena = GenerativeLifeRunner(config)
        arena.run()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
