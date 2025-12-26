"""
Influence Operator Base Class - RnB Framework Behavioral Engine

This module implements the base class for RnB influence operators, which are
the core mechanism of the Behavioral Engine (B).

From RnB papers (Bouchet & Sansonnet):
"The behavioral engine EB modifies rational decisions through influence operators
that are activated based on the agent's personality state (traits, moods, affects).
Each operator implements a specific behavioral scheme from the FFM/NEO PI-R taxonomy."

Architecture mapping:
- Rational Engine (R): Produces task-focused plans/prompts
- Behavioral Engine (B): Applies influence operators to modify R's output
- Influence Operators: Individual behavioral schemes (e.g., cooperation, formality)
- Activation Matrix: Determines which operators fire for given personality state

Reference: "Influence of FFM/NEO PI-R personality traits on the rational process 
of autonomous agents" (Bouchet & Sansonnet, 2013)
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from .context import InfluenceContext

if TYPE_CHECKING:
    from .context import InfluenceContext

class InfluenceOperator(ABC):
    """
    Base class for RnB behavioral influence operators.
    
    In RnB framework:
    - Operators implement specific behavioral schemes (69 bipolar schemes in full RnB)
    - Each operator modifies rational output based on personality state
    - Activation controlled by personality parameters (traits/moods/affects)
    - Multiple operators can be active simultaneously (activation matrix)
    - Each operator has activation conditions (applies method)
    - When active, operators modify prompts to express personality
    - Priority determines application order
    
    Subclasses must implement:
    - applies(context): Return True if operator should activate - activation condition (maps to RnB activation matrix)
    - apply(base_prompt, context): Return modified prompt - behavioral modification (maps to RnB influence mechanism)
    - get_activation_priority(): Return priority (lower = applied first)
    
    Example from RnB papers:
    "If cooperation affect is low, the operator reduces response verbosity
    and helpfulness. If extraversion is high, the operator increases
    enthusiasm and expressiveness."
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        category: Optional[str] = None
    ):
        """
        Initialize influence operator.
        
        Args:
            name: Operator identifier (e.g., "cooperation_verbosity")
            description: Human-readable description of behavioral effect
            category: Operator category (trait_based, mood_based, affect_based, composite)
        """
        self.name = name
        self.description = description
        self.category = category or "uncategorized"
    
    @abstractmethod
    def applies(self, context: InfluenceContext) -> bool:
        """
        Determine if operator should activate for given context.
        
        Implements RnB's activation matrix logic: maps personality state
        (FFM traits, moods, affects) to operator activation decision.
        
        From RnB papers:
        "The activation matrix specifies for each personality profile which
        influence operators should be active. This prevents conflicting
        behavioral modifications."
        
        Args:
            context: Current InfluenceContext (primarily M.A personality state)
            
        Returns:
            True if operator should modify the prompt/plan, False otherwise
            
        Example:
            def applies(self, context):
                # Activate only if cooperation affect is extreme
                cooperation = context.personality.affects[AffectDimension.COOPERATION]
                return abs(cooperation) > 0.5
        """
        pass
    
    @abstractmethod
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        """
        Apply behavioral influence to rational output.
        
        Implements RnB's influence mechanism: modifies the rational agent's
        output (task-focused prompt/plan) according to personality state.
        
        From RnB papers:
        "Influence operators modify rational decisions at specific intervention
        points. The modification reflects personality traits (stable), moods
        (dynamic short-term), and affects (relationship-specific)."
        
        Args:
            base_prompt: Original rational prompt (task-focused, personality-neutral)
            context: Current InfluenceContext with personality state
            
        Returns:
            Modified prompt incorporating behavioral influence
            
        Note:
            - Should preserve task requirements from base_prompt
            - Should only add personality-appropriate behavioral modifications
            - Multiple operators may be chained (order matters)
            
        Example:
            def apply(self, base_prompt, context):
                cooperation = context.personality.affects[AffectDimension.COOPERATION]
                if cooperation < -0.3:
                    return base_prompt + "\\nBe brief and minimal."
                elif cooperation > 0.7:
                    return base_prompt + "\\nBe thorough and helpful."
                return base_prompt
        """
        pass
    
    def get_activation_priority(self) -> int:
        """
        Get operator priority for application order.
        
        When multiple operators are active, they are applied in priority order.
        Lower numbers = higher priority (applied first).
        
        Default priority: 100 (medium)
        
        Override in subclasses for specific ordering needs:
        - Critical constraints (safety, correctness): 0-50
        - Personality traits (stable): 50-100
            - Gloss-based operators: 50-60
            - Trait-based operators: 60-80
        - Moods (dynamic): 100-150
        - Affects (relationship): 150-200
        
        Returns:
            Priority value (0-1000)
        """
        return 100
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}', category='{self.category}')"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class CompositeInfluenceOperator(InfluenceOperator):
    """
    Composite operator that combines multiple influence operators.
    
    Example:
    A "professional communication" composite might combine:
    - High conscientiousness → structured responses
    - Low extraversion → measured tone
    - High agreeableness → collaborative language
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        operators: list[InfluenceOperator]
    ):
        """
        Initialize composite operator.
        
        Args:
            name: Composite operator identifier
            description: Description of combined behavioral effect
            operators: List of component operators to apply
        """
        super().__init__(name, description, category="composite")
        self.operators = operators
    
    def applies(self, context: InfluenceContext) -> bool:
        """
        Composite applies if any component operator applies.
        
        Override for different composition logic (e.g., all must apply).
        """
        return any(op.applies(context) for op in self.operators)
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        """
        Apply all active component operators in sequence.
        
        Operators applied in priority order.
        """
        modified_prompt = base_prompt
        
        # Sort by priority (lower = higher priority)
        active_ops = [
            op for op in sorted(self.operators, key=lambda x: x.get_activation_priority())
            if op.applies(context)
        ]
        
        # Apply each operator in order
        for operator in active_ops:
            modified_prompt = operator.apply(modified_prompt, context)
        
        return modified_prompt