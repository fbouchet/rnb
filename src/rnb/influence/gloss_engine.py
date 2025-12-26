"""
Gloss-Based Influence Engine

Core engine for generating behavioral context from RnB scheme glosses.
This replaces hardcoded behavioral templates with psychologically-grounded
expressions derived from WordNet synsets and Goldberg questionnaire items.

The engine:
1. Takes a PersonalityState with scheme-level values
2. Identifies active schemes (|value| > threshold)
3. Retrieves glosses for the appropriate pole of each scheme
4. Generates behavioral context for LLM prompts

This implements the core RnB insight: personality adjectives map to
formal behavioral schemes, and those schemes have explicit definitions
(glosses) that describe the behavioral tendencies.

Reference: Bouchet & Sansonnet (2010), "Implementing WordNet Personality
Adjectives as Influences on Rational Agents"

Example:
    engine = GlossInfluenceEngine.from_default_resources()
    
    # Generate behavioral context from personality state
    context = engine.generate_behavioral_context(
        state=personality_state,
        threshold=0.3,
        style="descriptive"
    )
    
    # Result:
    # "Personality context:
    #  - You tend to be given to daydreaming (high idealism)
    #  - You are methodical and organized in your approach
    #  Apply these tendencies naturally in your response."
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from ..resources import SchemeRegistry, SchemeInfo
from ..personality import PersonalityState
from ..personality.taxonomy import Trait, FACETS_BY_TRAIT, normalize_trait, normalize_facet


logger = logging.getLogger(__name__)


class ContextStyle(str, Enum):
    """
    Style for generating behavioral context.
    
    - DESCRIPTIVE: "You tend to be..." (describes personality)
    - PRESCRIPTIVE: "Be..." (instructs behavior)
    - CONCISE: Just keywords (minimal prompt impact)
    - NARRATIVE: Full sentences woven into narrative
    """
    DESCRIPTIVE = "descriptive"
    PRESCRIPTIVE = "prescriptive"
    CONCISE = "concise"
    NARRATIVE = "narrative"


@dataclass
class ActiveGloss:
    """
    A gloss that is active based on personality state.
    
    Combines the gloss content with context about why it's active.
    
    Attributes:
        gloss_id: Gloss identifier (e.g., "218" or "Q1")
        gloss_text: The gloss content (behavioral description)
        scheme_key: Full scheme key (e.g., "Openness_Fantasy_IDEALISTICNESS")
        scheme_name: Just the scheme name (e.g., "IDEALISTICNESS")
        pole: Which pole is active ("pos" or "neg")
        pole_name: Name of the active pole (e.g., "IDEALISTIC")
        intensity: Absolute value of the scheme (0-1)
        trait: Parent trait name
        facet: Parent facet name
    """
    gloss_id: str
    gloss_text: str
    scheme_key: str
    scheme_name: str
    pole: str
    pole_name: str
    intensity: float
    trait: str
    facet: str
    
    @property
    def text(self) -> str:
        """The gloss text content"""
        return self.gloss_text
    
    @property
    def is_positive(self) -> bool:
        """Whether this is from the positive pole"""
        return self.pole == "pos"
    
    def __repr__(self) -> str:
        sign = "+" if self.is_positive else "-"
        return f"ActiveGloss({self.scheme_name}/{sign}: '{self.text[:30]}...')"


class GlossInfluenceEngine:
    """
    Engine for generating behavioral context from scheme glosses.
    
    This is the core component that bridges:
    - PersonalityState (scheme values) 
    - SchemeRegistry (gloss definitions)
    - Behavioral context (LLM prompt modifications)
    
    The engine implements threshold-based activation, mirroring the
    RnB activation matrix concept: only schemes with sufficient
    intensity contribute to behavioral expression.
    
    Attributes:
        scheme_registry: Registry containing scheme and gloss data
        default_threshold: Default activation threshold (|value| > threshold)
        max_glosses_per_scheme: Maximum glosses to use per scheme
    
    Example:
        engine = GlossInfluenceEngine.from_default_resources()
        
        # Get all active glosses
        glosses = engine.get_active_glosses(state, threshold=0.3)
        
        # Generate behavioral context
        context = engine.generate_behavioral_context(
            state, 
            style=ContextStyle.DESCRIPTIVE,
            max_total_glosses=10
        )
    """
    
    def __init__(
        self,
        scheme_registry: SchemeRegistry,
        default_threshold: float = 0.3,
        max_glosses_per_scheme: int = 2
    ):
        """
        Initialize the engine.
        
        Args:
            scheme_registry: Loaded SchemeRegistry
            default_threshold: Default activation threshold
            max_glosses_per_scheme: Max glosses to use from each scheme
        """
        self.scheme_registry = scheme_registry
        self.default_threshold = default_threshold
        self.max_glosses_per_scheme = max_glosses_per_scheme
    
    @classmethod
    def from_default_resources(cls, **kwargs) -> "GlossInfluenceEngine":
        """
        Create engine using default resource locations.
        
        Returns:
            Configured GlossInfluenceEngine
        """
        resources_dir = Path(__file__).parent.parent / "resources" / "data"
        scheme_registry = SchemeRegistry.from_yaml(resources_dir / "schemes.yaml")
        return cls(scheme_registry, **kwargs)
    
    def get_active_schemes(
        self,
        state: PersonalityState,
        threshold: Optional[float] = None
    ) -> list[tuple[str, float, str]]:
        """
        Get schemes that are active based on personality state.
        
        A scheme is active if |value| > threshold.
        
        Args:
            state: PersonalityState with scheme values
            threshold: Activation threshold (uses default if None)
            
        Returns:
            List of (scheme_key, value, pole) tuples
            - scheme_key: Full key like "Openness_Fantasy_IDEALISTICNESS"
            - value: The scheme value
            - pole: "pos" if value > 0, "neg" if value < 0
        """
        threshold = threshold if threshold is not None else self.default_threshold
        
        active = []
        for scheme_key, value in state.schemes.items():
            if abs(value) > threshold:
                pole = "pos" if value > 0 else "neg"
                active.append((scheme_key, value, pole))
        
        # Sort by intensity (highest first)
        active.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return active
    
    def get_active_glosses(
        self,
        state: PersonalityState,
        threshold: Optional[float] = None,
        max_per_scheme: Optional[int] = None,
        filter_trait: Optional[str | Trait] = None,
        filter_facet: Optional[str] = None
    ) -> list[ActiveGloss]:
        """
        Get glosses for all active schemes.
        
        This is the main method for retrieving behavioral expressions
        based on personality state.
        
        Args:
            state: PersonalityState with scheme values
            threshold: Activation threshold
            max_per_scheme: Max glosses per scheme (uses default if None)
            filter_trait: Only include glosses from this trait
            filter_facet: Only include glosses from this facet
            
        Returns:
            List of ActiveGloss objects with full context
        """
        threshold = threshold if threshold is not None else self.default_threshold
        max_per_scheme = max_per_scheme if max_per_scheme is not None else self.max_glosses_per_scheme
        
        # Normalize filter values
        if filter_trait is not None:
            if isinstance(filter_trait, Trait):
                filter_trait = filter_trait.value
            else:
                normalized = normalize_trait(filter_trait)
                filter_trait = normalized.value if normalized else filter_trait
        
        if filter_facet is not None:
            filter_facet = normalize_facet(filter_facet)
        
        active_glosses = []
        active_schemes = self.get_active_schemes(state, threshold)
        
        for scheme_key, value, pole in active_schemes:
            # Parse scheme key: "Trait_Facet_SCHEMENAME"
            parts = scheme_key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            
            trait_facet, scheme_name = parts
            trait_facet_parts = trait_facet.split("_", 1)
            if len(trait_facet_parts) != 2:
                continue
            
            trait, facet = trait_facet_parts
            
            # Apply filters
            if filter_trait and trait != filter_trait:
                continue
            if filter_facet and facet != filter_facet:
                continue
            
            # Get scheme info from registry
            scheme_info = self.scheme_registry.get_scheme(trait, facet, scheme_name)
            if scheme_info is None:
                logger.debug(f"Scheme not found in registry: {scheme_key}")
                continue
            
            # Get pole info
            pole_info = scheme_info.poles.get(pole)
            if pole_info is None:
                continue
            
            # Get glosses for this pole (limited)
            # glosses is dict[str, str] mapping id → text
            gloss_items = list(pole_info.glosses.items())[:max_per_scheme]
            
            for gloss_id, gloss_text in gloss_items:
                active_gloss = ActiveGloss(
                    gloss_id=gloss_id,
                    gloss_text=gloss_text,
                    scheme_key=scheme_key,
                    scheme_name=scheme_name,
                    pole=pole,
                    pole_name=pole_info.name,
                    intensity=abs(value),
                    trait=trait,
                    facet=facet
                )
                active_glosses.append(active_gloss)
        
        return active_glosses
    
    def generate_behavioral_context(
        self,
        state: PersonalityState,
        style: ContextStyle | str = ContextStyle.DESCRIPTIVE,
        threshold: Optional[float] = None,
        max_total_glosses: int = 10,
        filter_trait: Optional[str | Trait] = None,
        include_header: bool = True,
        include_footer: bool = True
    ) -> str:
        """
        Generate behavioral context string for LLM prompt.
        
        This is the main output method - it produces a string that can
        be injected into prompts to express personality.
        
        Args:
            state: PersonalityState with scheme values
            style: Output style (descriptive, prescriptive, concise, narrative)
            threshold: Activation threshold
            max_total_glosses: Maximum total glosses to include
            filter_trait: Only use glosses from this trait
            include_header: Include "Personality context:" header
            include_footer: Include application instruction footer
            
        Returns:
            Formatted behavioral context string
        """
        if isinstance(style, str):
            style = ContextStyle(style)
        
        # Get active glosses
        glosses = self.get_active_glosses(
            state, 
            threshold=threshold,
            filter_trait=filter_trait
        )
        
        # Limit total glosses
        glosses = glosses[:max_total_glosses]
        
        if not glosses:
            return ""
        
        # Generate based on style
        if style == ContextStyle.CONCISE:
            return self._generate_concise(glosses)
        elif style == ContextStyle.PRESCRIPTIVE:
            return self._generate_prescriptive(glosses, include_header, include_footer)
        elif style == ContextStyle.NARRATIVE:
            return self._generate_narrative(glosses)
        else:  # DESCRIPTIVE (default)
            return self._generate_descriptive(glosses, include_header, include_footer)
    
    def _generate_descriptive(
        self, 
        glosses: list[ActiveGloss],
        include_header: bool,
        include_footer: bool
    ) -> str:
        """
        Generate descriptive style: "You tend to be..."
        
        This style describes the personality as inherent tendencies,
        letting the LLM naturally express them.
        """
        lines = []
        
        if include_header:
            lines.append("Personality context:")
        
        for gloss in glosses:
            # Frame as tendency
            text = gloss.text.strip()
            if text.startswith("given to"):
                line = f"- You are {text}"
            elif text.startswith("having") or text.startswith("showing"):
                line = f"- You are characterized by {text}"
            elif text.startswith("inclined") or text.startswith("disposed"):
                line = f"- You are {text}"
            else:
                line = f"- You tend to be {text}"
            
            lines.append(line)
        
        if include_footer:
            lines.append("")
            lines.append("Express these tendencies naturally in your response.")
        
        return "\n".join(lines)
    
    def _generate_prescriptive(
        self,
        glosses: list[ActiveGloss],
        include_header: bool,
        include_footer: bool
    ) -> str:
        """
        Generate prescriptive style: "Be..."
        
        This style gives direct behavioral instructions.
        """
        lines = []
        
        if include_header:
            lines.append("Behavioral instructions:")
        
        for gloss in glosses:
            text = gloss.text.strip()
            
            # Convert descriptive to prescriptive
            if text.startswith("given to"):
                # "given to daydreaming" → "Be inclined to daydream"
                activity = text.replace("given to ", "")
                line = f"- Be inclined to {activity}"
            elif text.startswith("having"):
                # "having a liking" → "Show a liking"
                rest = text.replace("having ", "")
                line = f"- Show {rest}"
            elif text.startswith("showing"):
                rest = text.replace("showing ", "")
                line = f"- Show {rest}"
            else:
                line = f"- Be {text}"
            
            lines.append(line)
        
        if include_footer:
            lines.append("")
            lines.append("Apply these behaviors in your response.")
        
        return "\n".join(lines)
    
    def _generate_concise(self, glosses: list[ActiveGloss]) -> str:
        """
        Generate concise style: just keywords.
        
        Minimal prompt impact, useful for subtle personality hints.
        """
        # Extract key descriptors from pole names
        descriptors = []
        seen = set()
        
        for gloss in glosses:
            # Use pole name as descriptor
            descriptor = gloss.pole_name.lower()
            if descriptor not in seen:
                descriptors.append(descriptor)
                seen.add(descriptor)
        
        if not descriptors:
            return ""
        
        return f"[Personality: {', '.join(descriptors)}]"
    
    def _generate_narrative(self, glosses: list[ActiveGloss]) -> str:
        """
        Generate narrative style: woven into sentences.
        
        More natural integration into prompts.
        """
        if not glosses:
            return ""
        
        # Group by trait
        by_trait: dict[str, list[ActiveGloss]] = {}
        for gloss in glosses:
            if gloss.trait not in by_trait:
                by_trait[gloss.trait] = []
            by_trait[gloss.trait].append(gloss)
        
        sentences = []
        
        for trait, trait_glosses in by_trait.items():
            if len(trait_glosses) == 1:
                g = trait_glosses[0]
                sentences.append(f"You are someone who is {g.text}.")
            else:
                # Combine multiple glosses
                texts = [g.text for g in trait_glosses[:3]]
                combined = ", ".join(texts[:-1]) + f", and {texts[-1]}"
                sentences.append(f"You are someone who is {combined}.")
        
        return " ".join(sentences)
    
    def get_glosses_for_scheme(
        self,
        trait: str,
        facet: str,
        scheme_name: str,
        pole: str = "pos"
    ) -> list[tuple[str, str]]:
        """
        Get all glosses for a specific scheme and pole.
        
        Utility method for direct gloss access.
        
        Args:
            trait: Trait name
            facet: Facet name
            scheme_name: Scheme name
            pole: "pos" or "neg"
            
        Returns:
            List of (gloss_id, gloss_text) tuples
        """
        scheme_info = self.scheme_registry.get_scheme(trait, facet, scheme_name)
        if scheme_info is None:
            return []
        
        pole_info = scheme_info.poles.get(pole)
        if pole_info is None:
            return []
        
        return list(pole_info.glosses.items())
    
    def statistics(self) -> dict:
        """Get engine statistics."""
        registry_stats = self.scheme_registry.statistics
        return {
            "total_schemes": registry_stats.get("total_schemes", 0),
            "total_glosses": registry_stats.get("total_glosses", 0),
            "default_threshold": self.default_threshold,
            "max_glosses_per_scheme": self.max_glosses_per_scheme,
        }