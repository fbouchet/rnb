"""
Influence Context - RnB Framework Model M Implementation

This module implements the context passed to influence operators, based on RnB's
Model M (Mental Model) architecture:

From RnB papers:
- M.A: Agent's personality state (traits, moods, affects, roles)
- M.U: User model (knowledge, preferences, history) - future extension
- M.S: Session model (current conversation state) - future extension  
- M.T: Task/Topic model (domain knowledge) - future extension

Currently implements M.A fully, with extensibility for M.U, M.S, M.T.
"""

from typing import Optional
from dataclasses import dataclass
from ..personality.state import PersonalityState


@dataclass
class InfluenceContext:
    """
    Complete context available to RnB influence operators.
    
    Maps to RnB's Model M (Mental Model) components:
    - personality: M.A (Agent personality - traits, moods, affects)
    - user_context: M.U (User model - preferences, knowledge) [future]
    - session_context: M.S (Session history, conversation state) [future]
    - task_context: M.T (Domain knowledge, task specifics) [future]
    
    From RnB papers (Bouchet & Sansonnet):
    "The Model M is a mental model composed of an agent model M.A, 
    a user model M.U, a session model M.S, and topic models M.T"
    
    Attributes:
        personality: M.A - Current personality state (FFM traits, moods, affects)
        user_context: M.U - User model data (not yet implemented)
        session_context: M.S - Session/conversation context (not yet implemented)
        task_context: M.T - Domain/task-specific knowledge (not yet implemented)
    """
    
    # M.A: Agent personality (fully implemented)
    personality: PersonalityState
    
    # M.U: User model (future extension)
    user_context: Optional[dict] = None
    
    # M.S: Session model (future extension)
    session_context: Optional[dict] = None
    
    # M.T: Task/topic model (future extension)
    task_context: Optional[dict] = None
    
    @classmethod
    def from_personality(cls, personality: PersonalityState) -> "InfluenceContext":
        """
        Create context from personality state only.
        
        Convenience constructor for implementations where only
        M.A (agent personality) is actively managed.
        
        Args:
            personality: PersonalityState representing M.A
            
        Returns:
            InfluenceContext with M.A populated, other components None
        """
        return cls(personality=personality)
    
    def has_user_context(self) -> bool:
        """Check if M.U (user model) is available"""
        return self.user_context is not None
    
    def has_session_context(self) -> bool:
        """Check if M.S (session model) is available"""
        return self.session_context is not None
    
    def has_task_context(self) -> bool:
        """Check if M.T (task/topic model) is available"""
        return self.task_context is not None
    
    def get_agent_id(self) -> str:
        """
        Get agent identifier from M.A.
        
        Returns:
            Agent ID string
        """
        return self.personality.agent_id
    
    def __repr__(self) -> str:
        components = ["M.A(personality)"]
        if self.user_context:
            components.append("M.U(user)")
        if self.session_context:
            components.append("M.S(session)")
        if self.task_context:
            components.append("M.T(task)")
        
        return f"InfluenceContext({', '.join(components)})"