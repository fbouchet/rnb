"""
Operator Registry - RnB Activation Matrix Implementation

This module implements the activation matrix and operator management system
from the RnB Behavioral Engine.

From RnB papers (Bouchet & Sansonnet):
"The activation matrix determines which influence operators should be active
for a given personality profile. This prevents conflicting behavioral 
modifications and ensures coherent personality expression."

The registry:
- Maintains collection of available influence operators
- Implements activation matrix logic (which operators fire when)
- Coordinates operator application in priority order
- Provides query interface for active operators

Reference: "Influence of FFM/NEO PI-R personality traits on the rational 
process of autonomous agents" (Bouchet & Sansonnet, 2013), Section 3.
"""

from typing import Optional, Callable
from .base import InfluenceOperator
from .context import InfluenceContext


class OperatorRegistry:
    """
    Central registry for RnB influence operators with activation matrix.
    
    Maps to RnB's Behavioral Engine coordinator that:
    1. Maintains set of available influence operators (behavioral schemes)
    2. Determines which operators activate for given personality state
    3. Applies active operators in priority order
    4. Manages operator interactions and conflicts
    
    From RnB papers:
    "The behavioral scheduler selects which operators to activate based on
    the activation matrix - a mapping from personality profiles to operator
    sets. Multiple operators can be active simultaneously."
    """
    
    def __init__(self):
        """Initialize empty operator registry"""
        self._operators: list[InfluenceOperator] = []
        self._operators_by_name: dict[str, InfluenceOperator] = {}
        self._operators_by_category: dict[str, list[InfluenceOperator]] = {}
        
        # Custom activation rules (overrides operator.applies())
        # Maps operator name -> custom activation function
        self._custom_activation_rules: dict[str, Callable[[InfluenceContext], bool]] = {}
    
    def register(
        self,
        operator: InfluenceOperator,
        custom_activation: Optional[Callable[[InfluenceContext], bool]] = None
    ) -> None:
        """
        Register an influence operator.
        
        From RnB papers:
        "Each operator is registered with the behavioral engine and assigned
        an activation condition based on the personality parameters it responds to."
        
        Args:
            operator: InfluenceOperator instance to register
            custom_activation: Optional custom activation function that overrides
                             operator.applies(). Useful for complex activation
                             matrix logic that spans multiple operators.
        
        Raises:
            ValueError: If operator with same name already registered
            
        Example:
            registry = OperatorRegistry()
            
            # Register with default activation
            registry.register(CooperationInfluence())
            
            # Register with custom activation logic
            def activate_only_for_tutors(context):
                return context.personality.agent_id.startswith("tutor_")
            
            registry.register(
                PedagogicalInfluence(),
                custom_activation=activate_only_for_tutors
            )
        """
        if operator.name in self._operators_by_name:
            raise ValueError(
                f"Operator '{operator.name}' already registered. "
                f"Use unregister() first or choose different name."
            )
        
        # Add to collections
        self._operators.append(operator)
        self._operators_by_name[operator.name] = operator
        
        # Index by category
        if operator.category not in self._operators_by_category:
            self._operators_by_category[operator.category] = []
        self._operators_by_category[operator.category].append(operator)
        
        # Store custom activation rule if provided
        if custom_activation:
            self._custom_activation_rules[operator.name] = custom_activation
    
    def unregister(self, operator_name: str) -> bool:
        """
        Remove an operator from registry.
        
        Args:
            operator_name: Name of operator to remove
            
        Returns:
            True if operator was removed, False if not found
        """
        if operator_name not in self._operators_by_name:
            return False
        
        operator = self._operators_by_name[operator_name]
        
        # Remove from all collections
        self._operators.remove(operator)
        del self._operators_by_name[operator_name]
        self._operators_by_category[operator.category].remove(operator)
        
        # Clean up empty categories
        if not self._operators_by_category[operator.category]:
            del self._operators_by_category[operator.category]
        
        # Remove custom activation rule if exists
        if operator_name in self._custom_activation_rules:
            del self._custom_activation_rules[operator_name]
        
        return True
    
    def get_operator(self, name: str) -> Optional[InfluenceOperator]:
        """
        Retrieve operator by name.
        
        Args:
            name: Operator name
            
        Returns:
            InfluenceOperator if found, None otherwise
        """
        return self._operators_by_name.get(name)
    
    def list_operators(
        self,
        category: Optional[str] = None,
        include_inactive: bool = True
    ) -> list[InfluenceOperator]:
        """
        List registered operators.
        
        Args:
            category: If specified, only return operators in this category
            include_inactive: If False, filter to currently active operators
                            (requires context, not implemented here)
            
        Returns:
            List of operators matching criteria
        """
        if category:
            return self._operators_by_category.get(category, []).copy()
        return self._operators.copy()
    
    def list_categories(self) -> list[str]:
        """
        List all operator categories in registry.
        
        Returns:
            List of category names
        """
        return list(self._operators_by_category.keys())
    
    def _check_activation(
        self,
        operator: InfluenceOperator,
        context: InfluenceContext
    ) -> bool:
        """
        Check if operator should activate for given context.
        
        Implements RnB activation matrix logic:
        1. If custom activation rule exists, use that
        2. Otherwise, use operator's applies() method
        
        Args:
            operator: Operator to check
            context: Current influence context
            
        Returns:
            True if operator should activate
        """
        # Custom activation rule takes precedence
        if operator.name in self._custom_activation_rules:
            return self._custom_activation_rules[operator.name](context)
        
        # Default: use operator's applies() method
        return operator.applies(context)
    
    def get_active_operators(
        self,
        context: InfluenceContext,
        category: Optional[str] = None
    ) -> list[InfluenceOperator]:
        """
        Get all operators that should activate for given context.
        
        This is the core activation matrix query: given personality state
        (and potentially user/session/task context), which operators should
        modify the rational output?
        
        From RnB papers:
        "The activation matrix is queried with the current personality state
        to determine the set of active influence operators. These operators
        are then applied in priority order to modify the rational plan."
        
        Args:
            context: Current InfluenceContext (M.A, M.U, M.S, M.T)
            category: If specified, only consider operators in this category
            
        Returns:
            List of active operators, sorted by priority (high priority first)
            
        Example:
            context = InfluenceContext.from_personality(agent_state)
            active_ops = registry.get_active_operators(context)
            
            for op in active_ops:
                print(f"{op.name} is active (priority {op.get_activation_priority()})")
        """
        # Get candidate operators
        if category:
            candidates = self._operators_by_category.get(category, [])
        else:
            candidates = self._operators
        
        # Filter to active operators
        active = [
            op for op in candidates
            if self._check_activation(op, context)
        ]
        
        # Sort by priority (lower number = higher priority = applied first)
        active.sort(key=lambda op: op.get_activation_priority())
        
        return active
    
    def apply_all(
        self,
        base_prompt: str,
        context: InfluenceContext,
        category: Optional[str] = None,
        dry_run: bool = False
    ) -> str:
        """
        Apply all active operators to prompt.
        
        This is the main interface for the RnB Behavioral Engine:
        takes rational output (base_prompt) and applies behavioral
        influences based on personality state.
        
        From RnB papers:
        "The behavioral engine modifies the rational agent's output by
        sequentially applying active influence operators. Each operator
        adds personality-appropriate behavioral modifications."
        
        Args:
            base_prompt: Original rational prompt (task-focused, personality-neutral)
            context: Current InfluenceContext with personality state
            category: If specified, only apply operators from this category
            dry_run: If True, return base_prompt unchanged but log what would happen
            
        Returns:
            Modified prompt with all behavioral influences applied
            
        Example:
            # Rational agent produces task-focused prompt
            rational_prompt = "Explain how photosynthesis works."
            
            # Behavioral engine applies personality influences
            context = InfluenceContext.from_personality(agent_state)
            behavioral_prompt = registry.apply_all(rational_prompt, context)
            
            # Result might be:
            # "Explain how photosynthesis works.
            #  Communication style: Be thorough and helpful.
            #  Tone: Be enthusiastic and expressive."
        """
        active_operators = self.get_active_operators(context, category)
        
        if dry_run:
            # Just return what would be applied
            print(f"Would apply {len(active_operators)} operators:")
            for op in active_operators:
                print(f"  - {op.name} (priority {op.get_activation_priority()})")
            return base_prompt
        
        # Apply each operator in sequence
        modified_prompt = base_prompt
        for operator in active_operators:
            modified_prompt = operator.apply(modified_prompt, context)
        
        return modified_prompt
    
    def get_activation_summary(self, context: InfluenceContext) -> dict:
        """
        Get summary of operator activation for given context.
        
        Useful for debugging, logging, and understanding which operators
        are influencing behavior for a given personality state.
        
        Args:
            context: Current InfluenceContext
            
        Returns:
            Dictionary with activation statistics and details
            
        Example return:
            {
                "total_operators": 15,
                "active_operators": 5,
                "activation_rate": 0.33,
                "active_by_category": {
                    "trait_based": 2,
                    "mood_based": 2,
                    "affect_based": 1
                },
                "active_operator_names": [
                    "cooperation_verbosity",
                    "extraversion_enthusiasm",
                    "mood_energy",
                    "mood_happiness",
                    "conscientiousness_structure"
                ]
            }
        """
        total = len(self._operators)
        active = self.get_active_operators(context)
        
        # Count by category
        active_by_category = {}
        for op in active:
            if op.category not in active_by_category:
                active_by_category[op.category] = 0
            active_by_category[op.category] += 1
        
        return {
            "total_operators": total,
            "active_operators": len(active),
            "activation_rate": len(active) / total if total > 0 else 0.0,
            "active_by_category": active_by_category,
            "active_operator_names": [op.name for op in active]
        }
    
    def clear(self) -> None:
        """
        Remove all operators from registry.
        
        Useful for testing or reinitializing the behavioral engine.
        """
        self._operators.clear()
        self._operators_by_name.clear()
        self._operators_by_category.clear()
        self._custom_activation_rules.clear()
    
    def __len__(self) -> int:
        """Return number of registered operators"""
        return len(self._operators)
    
    def __contains__(self, operator_name: str) -> bool:
        """Check if operator is registered"""
        return operator_name in self._operators_by_name
    
    def __repr__(self) -> str:
        return (
            f"OperatorRegistry("
            f"operators={len(self._operators)}, "
            f"categories={len(self._operators_by_category)})"
        )