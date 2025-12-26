"""
RnB Personality Taxonomy

Defines the hierarchical structure of the FFM/NEO PI-R/BS taxonomy:
    Trait (5) → Facet (30) → Scheme (70)

This module provides:
- Canonical names and structure for the taxonomy
- Lookup functions for navigating the hierarchy
- Validation of taxonomy positions

The taxonomy is derived from the schemes.yaml resource but provides
a static, validated structure for runtime use.

Reference: Bouchet & Sansonnet (2013), "Influence of FFM/NEO PI-R 
personality traits on the rational process of autonomous agents"
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from ..resources import SchemeRegistry


class Trait(str, Enum):
    """
    Five Factor Model (FFM) personality traits.
    
    Also known as OCEAN or Big Five:
    - Openness to Experience
    - Conscientiousness  
    - Extraversion
    - Agreeableness
    - Neuroticism
    """
    OPENNESS = "Openness"
    CONSCIENTIOUSNESS = "Conscientiousness"
    EXTRAVERSION = "Extraversion"
    AGREEABLENESS = "Agreeableness"
    NEUROTICISM = "Neuroticism"


# Canonical facet names per trait (NEO PI-R structure)
# Note: Some names in the YAML have typos - we use corrected versions here
FACETS_BY_TRAIT: dict[Trait, list[str]] = {
    Trait.OPENNESS: [
        "Fantasy",
        "Aesthetics", 
        "Feelings",
        "Actions",
        "Ideas",
        "Values",
    ],
    Trait.CONSCIENTIOUSNESS: [
        "Competence",
        "Order",
        "Dutifulness",
        "Achievement-Striving",
        "Self-Discipline",
        "Deliberation",
    ],
    Trait.EXTRAVERSION: [
        "Warmth",
        "Gregariousness",
        "Assertiveness",
        "Activity",
        "Excitement-Seeking",
        "Positive-Emotions",
    ],
    Trait.AGREEABLENESS: [
        "Trust",
        "Straightforwardness",
        "Altruism",
        "Compliance",
        "Modesty",
        "Tender-Mindedness",
    ],
    Trait.NEUROTICISM: [
        "Anxiety",
        "Angry-Hostility",
        "Depression",
        "Self-Consciousness",
        "Impulsiveness",
        "Vulnerability",
    ],
}


# Mapping from variant facet names (as they appear in YAML) to canonical names
FACET_NAME_NORMALIZATION: dict[str, str] = {
    # Conscientiousness variants
    "Orderliness": "Order",
    "Achievement-striving": "Achievement-Striving",
    "Self-discipline": "Self-Discipline",
    # Extraversion variants  
    "Excitement-seeking": "Excitement-Seeking",
    "Positive-emotions": "Positive-Emotions",
    # Agreeableness variants
    "Tender-mindedness": "Tender-Mindedness",
    # Neuroticism variants
    "Angry-hostility": "Angry-Hostility",
    "Self-consciousness": "Self-Consciousness",
}


# Mapping from variant trait names to canonical Trait enum
# (lowercase for case-insensitive lookup)
TRAIT_NAME_NORMALIZATION: dict[str, Trait] = {
    "openness": Trait.OPENNESS,
    "conscientiousness": Trait.CONSCIENTIOUSNESS,
    "extraversion": Trait.EXTRAVERSION,
    "agreeableness": Trait.AGREEABLENESS,
    "neuroticism": Trait.NEUROTICISM,
}


@dataclass
class SchemeDefinition:
    """
    Definition of a behavioral scheme in the taxonomy.
    
    Attributes:
        name: Scheme name (e.g., "IDEALISTICNESS")
        trait: Parent trait
        facet: Parent facet (canonical name)
        positive_pole: Name of positive pole (e.g., "IDEALISTIC")
        negative_pole: Name of negative pole (e.g., "PRACTICAL")
        operator_hint: Operator category hint from resource
    """
    name: str
    trait: Trait
    facet: str
    positive_pole: str
    negative_pole: str
    operator_hint: str = ""
    
    @property
    def scheme_key(self) -> str:
        """Unique key for this scheme"""
        return f"{self.trait.value}_{self.facet}_{self.name}"
    
    @property
    def facet_key(self) -> str:
        """Key for parent facet"""
        return f"{self.trait.value}_{self.facet}"


class Taxonomy:
    """
    Complete RnB personality taxonomy.
    
    Provides navigation and lookup across the hierarchy:
        Trait (5) → Facet (30) → Scheme (70)
    
    The taxonomy can be built from the schemes resource or
    initialized with a static structure.
    
    Example:
        taxonomy = Taxonomy.from_scheme_registry(registry)
        
        # Get all schemes for a facet
        schemes = taxonomy.get_schemes_for_facet(Trait.OPENNESS, "Fantasy")
        
        # Get parent trait for a scheme
        trait = taxonomy.get_trait_for_scheme("IDEALISTICNESS")
    """
    
    def __init__(self):
        self._schemes: dict[str, SchemeDefinition] = {}
        self._schemes_by_facet: dict[str, list[str]] = {}
        self._schemes_by_trait: dict[Trait, list[str]] = {t: [] for t in Trait}
        self._facet_to_trait: dict[str, Trait] = {}
        
        # Build facet → trait mapping from canonical structure
        for trait, facets in FACETS_BY_TRAIT.items():
            for facet in facets:
                facet_key = f"{trait.value}_{facet}"
                self._facet_to_trait[facet_key] = trait
    
    @classmethod
    def from_scheme_registry(cls, registry: SchemeRegistry) -> "Taxonomy":
        """
        Build taxonomy from a loaded SchemeRegistry.
        
        Args:
            registry: Loaded SchemeRegistry from resources module
            
        Returns:
            Populated Taxonomy instance
        """
        taxonomy = cls()
        
        for scheme_info in registry.get_all_schemes():
            # Normalize trait name
            trait = normalize_trait(scheme_info.trait)
            if trait is None:
                continue  # Skip non-FFM schemes (e.g., "Others")
            
            # Normalize facet name
            facet = normalize_facet(scheme_info.facet)
            
            # Get pole names
            pos_pole = scheme_info.poles.get("pos")
            neg_pole = scheme_info.poles.get("neg")
            
            scheme_def = SchemeDefinition(
                name=scheme_info.name,
                trait=trait,
                facet=facet,
                positive_pole=pos_pole.name if pos_pole else "",
                negative_pole=neg_pole.name if neg_pole else "",
                operator_hint=scheme_info.operator_hint
            )
            
            taxonomy.add_scheme(scheme_def)
        
        return taxonomy
    
    def add_scheme(self, scheme: SchemeDefinition) -> None:
        """Add a scheme to the taxonomy"""
        key = scheme.scheme_key
        self._schemes[key] = scheme
        
        # Index by facet
        facet_key = scheme.facet_key
        if facet_key not in self._schemes_by_facet:
            self._schemes_by_facet[facet_key] = []
        self._schemes_by_facet[facet_key].append(key)
        
        # Index by trait
        self._schemes_by_trait[scheme.trait].append(key)
    
    def get_scheme(self, scheme_key: str) -> Optional[SchemeDefinition]:
        """Get scheme by its full key"""
        return self._schemes.get(scheme_key)
    
    def get_scheme_by_name(
        self, 
        trait: Trait, 
        facet: str, 
        name: str
    ) -> Optional[SchemeDefinition]:
        """Get scheme by trait, facet, and name"""
        key = f"{trait.value}_{facet}_{name}"
        return self._schemes.get(key)
    
    def get_schemes_for_facet(
        self, 
        trait: Trait, 
        facet: str
    ) -> list[SchemeDefinition]:
        """Get all schemes under a facet"""
        facet_key = f"{trait.value}_{facet}"
        scheme_keys = self._schemes_by_facet.get(facet_key, [])
        return [self._schemes[k] for k in scheme_keys]
    
    def get_schemes_for_trait(self, trait: Trait) -> list[SchemeDefinition]:
        """Get all schemes under a trait"""
        scheme_keys = self._schemes_by_trait.get(trait, [])
        return [self._schemes[k] for k in scheme_keys]
    
    def get_all_schemes(self) -> list[SchemeDefinition]:
        """Get all schemes in the taxonomy"""
        return list(self._schemes.values())
    
    def get_all_scheme_keys(self) -> list[str]:
        """Get all scheme keys"""
        return list(self._schemes.keys())
    
    def get_trait_for_facet(self, facet_key: str) -> Optional[Trait]:
        """Get the parent trait for a facet"""
        return self._facet_to_trait.get(facet_key)
    
    def get_facets_for_trait(self, trait: Trait) -> list[str]:
        """Get canonical facet names for a trait"""
        return FACETS_BY_TRAIT.get(trait, [])
    
    @property
    def num_traits(self) -> int:
        return len(Trait)
    
    @property
    def num_facets(self) -> int:
        return sum(len(facets) for facets in FACETS_BY_TRAIT.values())
    
    @property
    def num_schemes(self) -> int:
        return len(self._schemes)
    
    def __contains__(self, scheme_key: str) -> bool:
        return scheme_key in self._schemes


def normalize_trait(name: str) -> Optional[Trait]:
    """
    Normalize a trait name to the canonical Trait enum.
    
    Handles case variations and typos in the source data.
    
    Args:
        name: Trait name (may have variations)
        
    Returns:
        Trait enum or None if not a valid FFM trait
    """
    name_lower = name.lower()
    
    # Check normalization map first (handles typos)
    if name_lower in TRAIT_NAME_NORMALIZATION:
        return TRAIT_NAME_NORMALIZATION[name_lower]
    
    # Try direct enum lookup
    for trait in Trait:
        if trait.value.lower() == name_lower:
            return trait
    
    return None


def normalize_facet(name: str) -> str:
    """
    Normalize a facet name to canonical form.
    
    Handles variations in capitalization and naming.
    
    Args:
        name: Facet name (may have variations)
        
    Returns:
        Canonical facet name
    """
    # Check normalization map
    if name in FACET_NAME_NORMALIZATION:
        return FACET_NAME_NORMALIZATION[name]
    
    # Return as-is (already canonical or close enough)
    return name


def get_all_facet_keys() -> list[str]:
    """Get all canonical facet keys (trait_facet format)"""
    keys = []
    for trait, facets in FACETS_BY_TRAIT.items():
        for facet in facets:
            keys.append(f"{trait.value}_{facet}")
    return keys