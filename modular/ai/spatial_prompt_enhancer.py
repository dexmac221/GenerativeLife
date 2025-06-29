"""
Spatial Prompt Enhancer - Creates crystal-clear spatial prompts for LLM agents
Eliminates spatial reasoning confusion through explicit constraint communication
"""

from typing import Dict, Any, List
from .enhanced_spatial_system import SpatialIntelligenceSystem


class SpatialPromptEnhancer:
    """
    Enhances prompts with explicit spatial intelligence to prevent LLM 
    agents from attempting impossible moves.
    """
    
    def __init__(self):
        self.spatial_system = SpatialIntelligenceSystem()
    
    def enhance_prompt_with_spatial_intelligence(self, base_prompt: str, agent, world) -> str:
        """
        Enhance any prompt with crystal-clear spatial constraints and movement options.
        """
        spatial_intel = self.spatial_system.get_movement_intelligence(agent, world)
        
        enhanced_prompt = f"""
{base_prompt}

{self._create_spatial_constraints_section(spatial_intel)}

{self._create_available_actions_section(spatial_intel)}

{self._create_physics_enforcement_section()}

{self._create_example_section(spatial_intel)}
"""
        
        return enhanced_prompt
    
    def _create_spatial_constraints_section(self, spatial_intel: Dict[str, Any]) -> str:
        """Create explicit spatial constraints section."""
        
        spatial_context = spatial_intel['spatial_context']
        safe_movements = spatial_intel['safe_movements']
        
        section = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           🛡️ SPATIAL CONSTRAINTS SYSTEM 🛡️                        ║
║                                  MANDATORY READING                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

{spatial_context}

🚨 AVAILABLE MOVEMENTS (ONLY THESE ARE POSSIBLE):
"""
        
        if safe_movements:
            for movement in safe_movements:
                section += f"   ✅ {movement} - SAFE TO EXECUTE\n"
        else:
            section += "   ❌ NO MOVEMENTS POSSIBLE - YOU ARE COMPLETELY BLOCKED!\n"
            section += "   ⚠️ MUST USE NON-MOVEMENT ACTIONS: WAIT, EAT, DRINK, PICKUP, EXPLORE\n"
        
        return section
    
    def _create_available_actions_section(self, spatial_intel: Dict[str, Any]) -> str:
        """Create available actions section with movement filtering."""
        
        safe_movements = spatial_intel['safe_movements']
        emergency_actions = spatial_intel['emergency_actions']
        
        section = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              🎯 AVAILABLE ACTIONS                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

🚶 MOVEMENT ACTIONS (PRE-FILTERED FOR SAFETY):"""
        
        if safe_movements:
            for movement in safe_movements:
                section += f"\n   • {movement}"
        else:
            section += "\n   • NONE - ALL MOVEMENT BLOCKED BY OBSTACLES/BOUNDARIES"
        
        section += f"""

🔧 SURVIVAL ACTIONS (ALWAYS AVAILABLE):"""
        
        for action in emergency_actions:
            section += f"\n   • {action}"
        
        section += """

📦 ACTION SEQUENCES (COMBINE SAFELY):
   • You can combine actions: "MOVE_UP,PICKUP,EAT"
   • BUT movement must be from the safe list above
   • Example: If only MOVE_LEFT is safe, you can do "MOVE_LEFT,PICKUP,EAT"
   • NEVER include blocked movements in sequences
"""
        
        return section
    
    def _create_physics_enforcement_section(self) -> str:
        """Create physics enforcement section."""
        
        return """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                            ⚡ PHYSICS ENFORCEMENT ⚡                              ║
║                          VIOLATION = SYSTEM FAILURE                             ║
╚══════════════════════════════════════════════════════════════════════════════════╝

🔬 ABSOLUTE PHYSICAL LAWS:

1. 🧱 OBSTACLES ARE SOLID WALLS
   • You CANNOT walk through obstacles
   • Attempting this causes immediate action failure
   • Like hitting a brick wall in real life

2. 👥 AGENTS ARE SOLID BODIES  
   • You CANNOT occupy the same space as another agent
   • They physically block your movement
   • You must wait for them to move or choose different direction

3. 🌌 BOUNDARIES ARE THE VOID
   • Moving outside world boundaries = instant deletion
   • There is NOTHING beyond the world edges
   • Attempting this causes catastrophic system failure

4. ✅ VALID MOVEMENT TARGETS
   • EMPTY cells - safe to move
   • FOOD cells - safe to move (and pickup food)
   • WATER cells - safe to move (and pickup water)

⚠️ CRITICAL WARNING:
If you attempt any movement not listed in your SAFE MOVEMENTS above,
your action will be automatically rejected and replaced with WAIT.
"""
    
    def _create_example_section(self, spatial_intel: Dict[str, Any]) -> str:
        """Create concrete examples based on current situation."""
        
        safe_movements = spatial_intel['safe_movements']
        movement_analysis = spatial_intel['movement_analysis']
        
        section = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              📋 CONCRETE EXAMPLES                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Based on your CURRENT SITUATION:
"""
        
        # Show blocked examples
        for direction, info in movement_analysis.items():
            if not info['possible']:
                section += f"""
❌ BLOCKED EXAMPLE:
   • DO NOT choose: "MOVE_{direction}"
   • Reason: {info['explanation']}
   • This will cause: Action rejection and fallback to WAIT
"""
        
        # Show safe examples
        if safe_movements:
            section += f"""
✅ SAFE EXAMPLES:"""
            for movement in safe_movements[:2]:  # Show first 2 safe movements
                direction = movement.replace('MOVE_', '')
                info = movement_analysis[direction]
                section += f"""
   • "{movement}" → {info['explanation']}
   • "{movement},PICKUP" → Move and pickup any resource there
   • "{movement},PICKUP,EAT" → Move, pickup, and consume immediately"""
        else:
            section += """
⛔ NO MOVEMENT EXAMPLES:
   • You are completely surrounded - no movement possible
   • Safe choices: "WAIT", "EAT", "DRINK", "PICKUP", "EXPLORE" 
   • Example sequences: "EAT,DRINK", "PICKUP,EAT"
"""
        
        return section
    
    def create_action_validation_prompt(self, proposed_action: str, agent, world) -> str:
        """Create a final validation prompt before executing action."""
        
        is_valid, reason = self.spatial_system.validate_action_absolutely(proposed_action, agent, world)
        
        if is_valid:
            return f"✅ ACTION VALIDATED: {proposed_action} - {reason}"
        else:
            fallback = self.spatial_system.get_emergency_fallback(agent, world)
            return f"""
🚨 ACTION REJECTED: {proposed_action}
❌ Reason: {reason}
🔄 Auto-corrected to: {fallback}
⚠️ System prevented a physics violation
"""