"""
Gloss-Based Influence Operator

An InfluenceOperator that generates behavioral context from RnB scheme glosses.
This replaces hardcoded behavioral templates with psychologically-grounded
expressions derived from the WordNet/Goldberg personality resources.

Unlike the hardcoded trait operators (e.g., ConscientiousnessInfluence),
this operator:
1. Uses authentic behavioral definitions from the RnB resource
2. Works at scheme-level granularity (70 schemes vs 5 traits)
3. Automatically adapts to the agent's specific personality profile
4. Provides traceable, explainable behavioral modifications

The operator integrates with the standard InfluenceOperator interface,
so it can be registered with OperatorRegistry alongside other operators.

Reference: Bouchet & Sansonnet (2010), "Implementing WordNet Personality
Adjectives as Influences on Rational Agents"

Example:
    from rnb.influence import GlossBasedInfluence, OperatorRegistry
    from rnb.personality import PersonalityStateFactory
    
    # Create personality from adjectives
    factory = PersonalityStateFactory.from_default_resources()
    state = factory.from_adjectives("agent", ["romantic", "organized"])
    
    # Create gloss-based operator
    gloss_op = GlossBasedInfluence.from_default_resources()
    
    # Register and use
    registry = OperatorRegistry()
    registry.register(gloss_op)
    
    context = InfluenceContext.from_personality(state)
    modified_prompt = registry.apply_all("Explain recursion.", context)
"""

import logging
from typing import Optional

from .base import InfluenceOperator
from .context import InfluenceContext
from .gloss_engine import GlossInfluenceEngine, ContextStyle
from ..personality import PersonalityState, Trait


logger = logging.getLogger(__name__)


class GlossBasedInfluence(InfluenceOperator):
    """
    Influence operator that generates behavioral context from glosses.
    
    This operator queries the GlossInfluenceEngine to produce personality-
    appropriate behavioral context, which is then appended to the prompt.
    
    The operator activates when the personality state has at least one
    scheme above the activation threshold.
    
    Attributes:
        engine: GlossInfluenceEngine for gloss retrieval
        threshold: Activation threshold for schemes
        style: Context generation style
        max_glosses: Maximum glosses to include
        filter_trait: Optional trait filter
        priority: Operator priority (lower = applied earlier)
    
    Configuration options:
        - threshold: Higher = fewer, stronger personality expressions
        - style: descriptive, prescriptive, concise, or narrative
        - max_glosses: Limit prompt length impact
        - filter_trait: Focus on specific trait (e.g., just Openness)
    
    Example:
        # Full personality expression
        op = GlossBasedInfluence.from_default_resources(
            threshold=0.3,
            style="descriptive",
            max_glosses=8
        )
        
        # Focused on conscientiousness only
        op = GlossBasedInfluence.from_default_resources(
            filter_trait=Trait.CONSCIENTIOUSNESS,
            style="prescriptive"
        )
    """
    
    def __init__(
        self,
        engine: GlossInfluenceEngine,
        threshold: float = 0.3,
        style: ContextStyle | str = ContextStyle.DESCRIPTIVE,
        max_glosses: int = 8,
        filter_trait: Optional[str | Trait] = None,
        priority: int = 50,
        name: Optional[str] = None
    ):
        """
        Initialize the operator.
        
        Args:
            engine: GlossInfluenceEngine instance
            threshold: Scheme activation threshold
            style: Context generation style
            max_glosses: Maximum glosses to include
            filter_trait: Only use glosses from this trait
            priority: Operator priority (lower = applied earlier)
            name: Optional custom name
        """
        self.engine = engine
        self.threshold = threshold
        self.style = ContextStyle(style) if isinstance(style, str) else style
        self.max_glosses = max_glosses
        self.filter_trait = filter_trait
        self._priority = priority
        
        # Build name
        if name:
            self._name = name
        elif filter_trait:
            trait_name = filter_trait.value if isinstance(filter_trait, Trait) else filter_trait
            self._name = f"gloss_influence_{trait_name.lower()}"
        else:
            self._name = "gloss_influence"
        
        super().__init__(
            name=self._name,
            description=f"Gloss-based personality influence ({self.style.value} style)"
        )
    
    @classmethod
    def from_default_resources(cls, **kwargs) -> "GlossBasedInfluence":
        """
        Create operator using default resource locations.
        
        All kwargs are passed to the constructor.
        
        Returns:
            Configured GlossBasedInfluence
        """
        engine = GlossInfluenceEngine.from_default_resources()
        return cls(engine=engine, **kwargs)
    
    def applies(self, context: InfluenceContext) -> bool:
        """
        Check if operator should activate.
        
        Activates when at least one scheme exceeds the threshold.
        
        Args:
            context: Current influence context with personality state
            
        Returns:
            True if operator should apply
        """
        state = context.personality
        
        # Check if any scheme exceeds threshold
        active_schemes = self.engine.get_active_schemes(state, self.threshold)
        
        # Apply trait filter if specified
        if self.filter_trait and active_schemes:
            trait_name = self.filter_trait.value if isinstance(self.filter_trait, Trait) else self.filter_trait
            active_schemes = [
                (key, val, pole) for key, val, pole in active_schemes
                if key.startswith(trait_name + "_")
            ]
        
        return len(active_schemes) > 0
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        """
        Apply gloss-based behavioral context to the prompt.
        
        Generates behavioral context from active glosses and appends
        it to the base prompt.
        
        Args:
            base_prompt: Original prompt
            context: Current influence context
            
        Returns:
            Modified prompt with behavioral context
        """
        state = context.personality
        
        # Generate behavioral context
        behavioral_context = self.engine.generate_behavioral_context(
            state=state,
            style=self.style,
            threshold=self.threshold,
            max_total_glosses=self.max_glosses,
            filter_trait=self.filter_trait,
            include_header=True,
            include_footer=True
        )
        
        if not behavioral_context:
            return base_prompt
        
        # Combine prompt with behavioral context
        return f"{base_prompt}\n\n{behavioral_context}"
    
    def get_activation_priority(self) -> int:
        """
        Get operator priority.
        
        Default is 50, which is higher priority (applied earlier) than
        the hardcoded trait operators (60-80).
        
        Returns:
            Priority value (lower = higher priority)
        """
        return self._priority
    
    def get_active_glosses_preview(
        self, 
        state: PersonalityState
    ) -> list[str]:
        """
        Preview which glosses would be used for a personality state.
        
        Useful for debugging and understanding operator behavior.
        
        Args:
            state: Personality state to analyze
            
        Returns:
            List of gloss text strings that would be used
        """
        glosses = self.engine.get_active_glosses(
            state,
            threshold=self.threshold,
            filter_trait=self.filter_trait
        )
        return [g.text for g in glosses[:self.max_glosses]]


class TraitGlossInfluence(GlossBasedInfluence):
    """
    Convenience class for trait-specific gloss influence.
    
    Pre-configured to filter by a specific FFM trait.
    
    Example:
        # Openness-focused operator
        openness_op = TraitGlossInfluence(
            trait=Trait.OPENNESS,
            style="descriptive"
        )
        
        # Only activates and uses glosses from Openness schemes
    """
    
    def __init__(
        self,
        trait: Trait,
        threshold: float = 0.3,
        style: ContextStyle | str = ContextStyle.DESCRIPTIVE,
        max_glosses: int = 6,
        priority: int = 55
    ):
        """
        Initialize trait-specific operator.
        
        Args:
            trait: FFM trait to focus on
            threshold: Scheme activation threshold
            style: Context generation style
            max_glosses: Maximum glosses to include
            priority: Operator priority
        """
        engine = GlossInfluenceEngine.from_default_resources()
        
        super().__init__(
            engine=engine,
            threshold=threshold,
            style=style,
            max_glosses=max_glosses,
            filter_trait=trait,
            priority=priority,
            name=f"gloss_{trait.value.lower()}"
        )
        
        self.trait = trait


# Convenience instances for each trait
def create_openness_gloss_operator(**kwargs) -> TraitGlossInfluence:
    """Create gloss-based operator for Openness trait."""
    return TraitGlossInfluence(trait=Trait.OPENNESS, **kwargs)


def create_conscientiousness_gloss_operator(**kwargs) -> TraitGlossInfluence:
    """Create gloss-based operator for Conscientiousness trait."""
    return TraitGlossInfluence(trait=Trait.CONSCIENTIOUSNESS, **kwargs)


def create_extraversion_gloss_operator(**kwargs) -> TraitGlossInfluence:
    """Create gloss-based operator for Extraversion trait."""
    return TraitGlossInfluence(trait=Trait.EXTRAVERSION, **kwargs)


def create_agreeableness_gloss_operator(**kwargs) -> TraitGlossInfluence:
    """Create gloss-based operator for Agreeableness trait."""
    return TraitGlossInfluence(trait=Trait.AGREEABLENESS, **kwargs)


def create_neuroticism_gloss_operator(**kwargs) -> TraitGlossInfluence:
    """Create gloss-based operator for Neuroticism trait."""
    return TraitGlossInfluence(trait=Trait.NEUROTICISM, **kwargs)