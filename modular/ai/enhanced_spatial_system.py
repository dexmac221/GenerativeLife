"""
Enhanced Spatial Intelligence System for LLM Agents
Addresses the fundamental problem of LLMs not understanding physical constraints
"""

from typing import Dict, List, Tuple, Any, Optional
from ..core.world import World, CellType
from ..core.agent import Agent, Action
import random


class SpatialIntelligenceSystem:
    """
    Advanced spatial reasoning system that prevents LLM agents from attempting 
    impossible moves by providing crystal-clear spatial information and 
    multi-layered validation.
    """
    
    def __init__(self):
        self.violation_count = 0
        self.blocked_attempts = []
        
    def get_movement_intelligence(self, agent: Agent, world: World) -> Dict[str, Any]:
        """
        Generate comprehensive spatial intelligence for an agent.
        Returns detailed movement possibilities with clear explanations.
        """
        x, y = agent.position
        
        # Analyze each direction with detailed reasoning
        movement_analysis = self._analyze_all_directions(x, y, world)
        
        # Generate safe movement options
        safe_movements = self._get_safe_movements(movement_analysis)
        
        # Create spatial context for LLM
        spatial_context = self._create_spatial_context(x, y, world, movement_analysis)
        
        return {
            'safe_movements': safe_movements,
            'movement_analysis': movement_analysis,
            'spatial_context': spatial_context,
            'emergency_actions': self._get_emergency_actions(),
            'physics_rules': self._get_physics_rules()
        }
    
    def _analyze_all_directions(self, x: int, y: int, world: World) -> Dict[str, Dict]:
        """Analyze every possible direction with detailed reasoning."""
        directions = {
            'UP': (x, y - 1),
            'DOWN': (x, y + 1), 
            'LEFT': (x - 1, y),
            'RIGHT': (x + 1, y)
        }
        
        analysis = {}
        
        for direction, (target_x, target_y) in directions.items():
            # Check if position is valid
            if not world.is_valid_position(target_x, target_y):
                analysis[direction] = {
                    'possible': False,
                    'reason': 'WORLD_BOUNDARY',
                    'explanation': f"Outside world limits at ({target_x},{target_y})",
                    'cell_type': 'BOUNDARY',
                    'danger_level': 'IMPOSSIBLE'
                }
            else:
                cell_type = world.get_cell_type(target_x, target_y)
                has_agent = world.has_agent(target_x, target_y)
                
                if cell_type == CellType.OBSTACLE:
                    analysis[direction] = {
                        'possible': False,
                        'reason': 'SOLID_OBSTACLE',
                        'explanation': f"Blocked by impassable obstacle at ({target_x},{target_y})",
                        'cell_type': 'OBSTACLE',
                        'danger_level': 'BLOCKED'
                    }
                elif has_agent:
                    analysis[direction] = {
                        'possible': False,
                        'reason': 'OCCUPIED_BY_AGENT',
                        'explanation': f"Another agent occupies ({target_x},{target_y})",
                        'cell_type': 'AGENT',
                        'danger_level': 'OCCUPIED'
                    }
                else:
                    # Movement is possible
                    analysis[direction] = {
                        'possible': True,
                        'reason': 'CLEAR_PATH',
                        'explanation': f"Safe to move to {cell_type.name} at ({target_x},{target_y})",
                        'cell_type': cell_type.name,
                        'danger_level': 'SAFE',
                        'target_position': (target_x, target_y)
                    }
        
        return analysis
    
    def _get_safe_movements(self, analysis: Dict[str, Dict]) -> List[str]:
        """Extract only the movements that are physically possible."""
        safe_moves = []
        
        for direction, info in analysis.items():
            if info['possible']:
                safe_moves.append(f'MOVE_{direction}')
        
        # Randomize to prevent directional bias
        random.shuffle(safe_moves)
        
        return safe_moves
    
    def _create_spatial_context(self, x: int, y: int, world: World, analysis: Dict) -> str:
        """Create crystal-clear spatial context for LLM understanding."""
        
        context = f"""
🗺️ SPATIAL SITUATION ANALYSIS for position ({x},{y}):

📍 CURRENT POSITION: You are standing at ({x},{y})

🧭 MOVEMENT POSSIBILITIES:
"""
        
        for direction, info in analysis.items():
            target_pos = f"({x},{y})" if not info['possible'] else f"({info.get('target_position', (x,y))[0]},{info.get('target_position', (x,y))[1]})"
            
            if info['possible']:
                context += f"✅ {direction}: CLEAR - {info['explanation']}\n"
            else:
                context += f"❌ {direction}: BLOCKED - {info['explanation']}\n"
        
        safe_directions = [d for d, info in analysis.items() if info['possible']]
        blocked_directions = [d for d, info in analysis.items() if not info['possible']]
        
        context += f"""
📋 SUMMARY:
• SAFE DIRECTIONS: {', '.join(safe_directions) if safe_directions else 'NONE - YOU ARE TRAPPED!'}
• BLOCKED DIRECTIONS: {', '.join(blocked_directions) if blocked_directions else 'NONE - ALL DIRECTIONS CLEAR'}

⚠️ CRITICAL RULES:
• You CANNOT move through OBSTACLES - they are solid walls
• You CANNOT move through other AGENTS - they block your path  
• You CANNOT move outside world boundaries - there is VOID beyond
• You CAN ONLY choose from SAFE DIRECTIONS listed above
"""
        
        return context
    
    def _get_emergency_actions(self) -> List[str]:
        """Get emergency actions when no movement is possible."""
        return ['WAIT', 'EXPLORE', 'PICKUP', 'EAT', 'DRINK']
    
    def _get_physics_rules(self) -> str:
        """Get absolute physics rules for LLM understanding."""
        return """
🔬 ABSOLUTE PHYSICS LAWS:
1. OBSTACLES are SOLID - you bounce off them like walls
2. AGENTS are SOLID - you cannot occupy the same space
3. BOUNDARIES are VOID - stepping there means deletion from existence
4. Only EMPTY, FOOD, and WATER cells allow movement
5. If no safe movement exists, you MUST choose non-movement actions
"""

    def validate_action_absolutely(self, action: str, agent: Agent, world: World) -> Tuple[bool, str]:
        """
        Absolute final validation - catches any action that slipped through.
        This is the last line of defense against impossible moves.
        """
        if not action.startswith('MOVE_'):
            return True, "Non-movement action is always valid"
        
        x, y = agent.position
        
        # Extract direction
        direction = action.replace('MOVE_', '')
        direction_map = {
            'UP': (0, -1),
            'DOWN': (0, 1), 
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }
        
        if direction not in direction_map:
            return False, f"Invalid direction: {direction}"
        
        dx, dy = direction_map[direction]
        target_x, target_y = x + dx, y + dy
        
        # Comprehensive validation
        if not world.is_valid_position(target_x, target_y):
            self.violation_count += 1
            reason = f"BOUNDARY VIOLATION: Attempted to move to ({target_x},{target_y}) which is outside world bounds"
            self.blocked_attempts.append(reason)
            return False, reason
        
        if not world.can_move_to(target_x, target_y):
            self.violation_count += 1
            cell_type = world.get_cell_type(target_x, target_y)
            if cell_type == CellType.OBSTACLE:
                reason = f"OBSTACLE COLLISION: Attempted to move into solid obstacle at ({target_x},{target_y})"
            elif world.has_agent(target_x, target_y):
                reason = f"AGENT COLLISION: Attempted to move into occupied space at ({target_x},{target_y})"
            else:
                reason = f"MOVEMENT BLOCKED: Cannot move to ({target_x},{target_y}) - {cell_type.name}"
            
            self.blocked_attempts.append(reason)
            return False, reason
        
        return True, f"Movement to ({target_x},{target_y}) is valid"
    
    def get_emergency_fallback(self, agent: Agent, world: World) -> str:
        """Get a guaranteed-safe action when everything else fails."""
        x, y = agent.position
        
        # Try to find any valid movement
        analysis = self._analyze_all_directions(x, y, world)
        safe_moves = self._get_safe_movements(analysis)
        
        if safe_moves:
            return random.choice(safe_moves)
        
        # No movement possible - use survival actions
        emergency_actions = ['WAIT', 'EAT', 'DRINK', 'PICKUP', 'EXPLORE']
        
        # Prioritize based on agent needs
        if agent.hunger > 70 and agent.inventory['food'] > 0:
            return 'EAT'
        elif agent.thirst > 70 and agent.inventory['water'] > 0:
            return 'DRINK'
        elif agent.energy < 20:
            return 'WAIT'
        else:
            return random.choice(emergency_actions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            'total_violations_prevented': self.violation_count,
            'recent_blocked_attempts': self.blocked_attempts[-10:],  # Last 10
            'system_effectiveness': 'ACTIVE' if self.violation_count > 0 else 'MONITORING'
        }