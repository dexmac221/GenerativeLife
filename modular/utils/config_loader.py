"""
Configuration loader for GenerativeLife simulation parameters.
"""

import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to config.yaml in project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            config_path = os.path.join(project_root, 'config.yaml')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_path}")
            return self._config
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}, using defaults")
            self._config = self._get_default_config()
            return self._config
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}, using defaults")
            self._config = self._get_default_config()
            return self._config

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.

        Example: config.get('survival.hunger.critical', 85)
        """
        if self._config is None:
            self.load_config()

        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_survival_config(self) -> Dict[str, Any]:
        """Get survival-related configuration."""
        return self.get('survival', {})

    def get_ai_behavior_config(self) -> Dict[str, Any]:
        """Get AI behavior configuration."""
        return self.get('ai_behavior', {})

    def get_breeding_config(self) -> Dict[str, Any]:
        """Get breeding configuration."""
        return self.get('breeding', {})

    def get_world_config(self) -> Dict[str, Any]:
        """Get world configuration."""
        return self.get('world', {})

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback default configuration if YAML loading fails."""
        return {
            'survival': {
                'hunger': {
                    'critical': 85,
                    'urgent': 60,
                    'high_priority': 40,
                    'medium_priority': 25
                },
                'thirst': {
                    'critical': 85,
                    'urgent': 60,
                    'high_priority': 40,
                    'medium_priority': 25
                },
                'energy': {
                    'critical': 15,
                    'low': 30,
                    'breeding_min': 50
                }
            },
            'ai_behavior': {
                'simple_controller': {
                    'high_priority_hunger': 50,
                    'high_priority_thirst': 50,
                    'medium_priority_hunger': 30,
                    'medium_priority_thirst': 30,
                    'emergency_energy': 20,
                    'basic_needs_hunger': 40,
                    'basic_needs_thirst': 40
                },
                'vlm_controller': {
                    'urgent_hunger': 70,
                    'urgent_thirst': 70,
                    'recommended_hunger': 40,
                    'recommended_thirst': 40,
                    'low_priority_hunger': 20,
                    'low_priority_thirst': 20,
                    'resource_shortage_hunger': 70,
                    'resource_shortage_thirst': 70
                }
            },
            'breeding': {
                'min_age': 50,
                'pregnancy_duration': 30,
                'max_population': 50,
                'min_energy': 50,
                'fertility_decline_age': 200
            },
            'resources': {
                'max_inventory_food': 2,
                'max_inventory_water': 2,
                'pickup_threshold_food': 2,
                'pickup_threshold_water': 2
            },
            'world': {
                'default_width': 15,
                'default_height': 20,
                'observation_radius': 2
            },
            'performance': {
                'stats_report_interval': 10,
                'memory_size': 50,
                'recent_memories_count': 5
            }
        }


# Global config instance
config = ConfigLoader()
