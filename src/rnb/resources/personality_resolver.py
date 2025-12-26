"""
High-level Personality Resolver for RnB framework.

Combines adjective resolution with scheme lookup to provide
complete personality specifications from natural language input.

This is the main entry point for converting user-provided adjectives
into structured personality specifications that can be converted to
PersonalityState in subsequent processing steps.

Usage:
    resolver = PersonalityResolver.from_yaml(
        neopiradj_path="path/to/neopiradj.yaml",
        schemes_path="path/to/schemes.yaml"
    )
    
    spec = resolver.resolve(["romantic", "organized", "shy"])
    
    # Access resolved positions
    for r in spec.resolved:
        print(f"{r.adjective.word} → {r.position}")
        print(f"  Glosses: {[g.text for g in r.all_glosses_for_pole]}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .adjective_resolver import AdjectiveResolver
from .scheme_registry import SchemeRegistry
from .models import (
    AdjectiveEntry,
    TaxonomyPosition,
    SchemeInfo,
    GlossEntry,
)


logger = logging.getLogger(__name__)


@dataclass
class ResolvedAdjective:
    """
    Fully resolved adjective with complete scheme information.
    
    Links the full chain: adjective → gloss → scheme → all glosses
    
    Attributes:
        adjective: The AdjectiveEntry with word, synset, weight, etc.
        scheme: Full SchemeInfo (may be None if gloss not in schemes)
        position: Complete TaxonomyPosition including scheme
        all_glosses_for_pole: All glosses in this scheme's pole
    
    Example:
        resolved = spec.resolved[0]
        print(f"Word: {resolved.adjective.word}")
        print(f"Position: {resolved.position}")
        print(f"Scheme: {resolved.scheme.name if resolved.scheme else 'N/A'}")
        print(f"Related glosses: {len(resolved.all_glosses_for_pole)}")
    """
    adjective: AdjectiveEntry
    scheme: Optional[SchemeInfo]
    position: TaxonomyPosition
    all_glosses_for_pole: list[GlossEntry] = field(default_factory=list)
    
    @property
    def has_scheme(self) -> bool:
        """True if scheme information was found"""
        return self.scheme is not None
    
    @property
    def operator_hint(self) -> str:
        """Operator hint from scheme (empty string if no scheme)"""
        return self.scheme.operator_hint if self.scheme else ""
    
    @property
    def pole_name(self) -> str:
        """Name of the activated pole (e.g., 'IDEALISTIC')"""
        if self.scheme and self.position.pole in self.scheme.poles:
            return self.scheme.poles[self.position.pole].name
        return ""


@dataclass
class PersonalitySpecification:
    """
    Complete specification of a personality from adjectives.
    
    This is the output of the resolution pipeline, containing:
    - All successfully resolved adjectives with their positions
    - Adjectives that couldn't be found
    - Adjectives with ambiguous mappings
    
    Attributes:
        input_adjectives: Original list of adjectives provided
        resolved: List of fully resolved adjectives
        unresolved: Adjectives not found in the resource
        ambiguities: Adjectives mapping to multiple facets
    
    Example:
        spec = resolver.resolve(["romantic", "lazy", "unknown_word"])
        
        print(f"Resolved: {len(spec.resolved)}")
        print(f"Unresolved: {spec.unresolved}")  # ["unknown_word"]
        print(f"Ambiguous: {spec.ambiguities}")
        
        # Get all affected schemes
        for position in spec.affected_positions:
            print(f"  {position}")
    """
    input_adjectives: list[str]
    resolved: list[ResolvedAdjective] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    ambiguities: list[tuple[str, list[TaxonomyPosition]]] = field(default_factory=list)
    
    @property
    def affected_positions(self) -> set[TaxonomyPosition]:
        """
        All unique positions affected by this specification.
        
        Returns:
            Set of TaxonomyPosition objects
        """
        return {r.position for r in self.resolved}
    
    @property
    def affected_schemes(self) -> set[str]:
        """
        All unique scheme keys affected.
        
        Returns:
            Set of scheme keys (trait_facet_scheme format)
        """
        return {
            r.position.scheme_key 
            for r in self.resolved 
            if r.position.scheme
        }
    
    @property
    def affected_facets(self) -> set[str]:
        """
        All unique facets affected.
        
        Returns:
            Set of facet keys (trait_facet format)
        """
        return {r.position.facet_key for r in self.resolved}
    
    @property
    def affected_traits(self) -> set[str]:
        """
        All unique traits affected.
        
        Returns:
            Set of trait names
        """
        return {r.position.trait for r in self.resolved}
    
    @property
    def is_complete(self) -> bool:
        """True if all input adjectives were resolved"""
        return len(self.unresolved) == 0
    
    @property
    def has_ambiguities(self) -> bool:
        """True if any adjectives had ambiguous mappings"""
        return len(self.ambiguities) > 0
    
    def get_resolved_for_facet(
        self, 
        trait: str, 
        facet: str
    ) -> list[ResolvedAdjective]:
        """
        Get all resolved adjectives for a specific facet.
        
        Args:
            trait: Trait name
            facet: Facet name
            
        Returns:
            List of ResolvedAdjective objects for that facet
        """
        facet_key = f"{trait}_{facet}"
        return [
            r for r in self.resolved
            if r.position.facet_key == facet_key
        ]
    
    def get_resolved_for_trait(self, trait: str) -> list[ResolvedAdjective]:
        """
        Get all resolved adjectives for a specific trait.
        
        Args:
            trait: Trait name
            
        Returns:
            List of ResolvedAdjective objects for that trait
        """
        return [
            r for r in self.resolved
            if r.position.trait.lower() == trait.lower()
        ]
    
    def get_glosses_for_trait(self, trait: str) -> list[GlossEntry]:
        """
        Get all glosses activated for a trait.
        
        Collects glosses from all resolved adjectives under the trait.
        
        Args:
            trait: Trait name
            
        Returns:
            List of GlossEntry objects
        """
        glosses = []
        for r in self.get_resolved_for_trait(trait):
            glosses.extend(r.all_glosses_for_pole)
        return glosses
    
    def summary(self) -> str:
        """
        Generate human-readable summary of the specification.
        
        Returns:
            Multi-line string summarizing the resolution
        """
        lines = [
            f"Personality Specification",
            f"  Input: {self.input_adjectives}",
            f"  Resolved: {len(self.resolved)} mappings",
            f"  Unresolved: {self.unresolved}" if self.unresolved else "",
            f"  Affected traits: {sorted(self.affected_traits)}",
            f"  Affected facets: {len(self.affected_facets)}",
        ]
        
        if self.ambiguities:
            lines.append(f"  Ambiguities: {len(self.ambiguities)}")
            for adj, positions in self.ambiguities:
                lines.append(f"    '{adj}' → {[str(p) for p in positions]}")
        
        return "\n".join(line for line in lines if line)


class PersonalityResolver:
    """
    High-level resolver combining adjective resolution with scheme lookup.
    
    This is the main entry point for converting user-provided adjectives
    into a structured personality specification.
    
    The resolver:
    1. Looks up each adjective in the neopiradj resource
    2. Links found glosses to their schemes
    3. Collects all glosses for each activated scheme pole
    4. Tracks unresolved and ambiguous adjectives
    
    Example:
        resolver = PersonalityResolver.from_yaml(
            neopiradj_path="resources/neopiradj.yaml",
            schemes_path="resources/schemes.yaml"
        )
        
        spec = resolver.resolve(["romantic", "organized", "shy"])
        
        # Check results
        print(f"Found {len(spec.resolved)} mappings")
        for r in spec.resolved:
            print(f"  {r.adjective.word} → {r.position}")
    """
    
    def __init__(
        self, 
        adjective_resolver: AdjectiveResolver,
        scheme_registry: SchemeRegistry,
        warn_unresolved: bool = True
    ):
        """
        Initialize with pre-loaded resolvers.
        
        Args:
            adjective_resolver: Loaded AdjectiveResolver
            scheme_registry: Loaded SchemeRegistry
            warn_unresolved: If True, log warnings for unresolved adjectives
        """
        self.adjectives = adjective_resolver
        self.schemes = scheme_registry
        self._warn_unresolved = warn_unresolved
    
    @classmethod
    def from_yaml(
        cls, 
        neopiradj_path: str | Path, 
        schemes_path: str | Path,
        warn_unresolved: bool = True
    ) -> "PersonalityResolver":
        """
        Load both resources and create resolver.
        
        Args:
            neopiradj_path: Path to neopiradj.yaml
            schemes_path: Path to schemes.yaml
            warn_unresolved: If True, log warnings for unresolved adjectives
            
        Returns:
            Initialized PersonalityResolver
        """
        adj_resolver = AdjectiveResolver.from_yaml(
            neopiradj_path, 
            warn_unresolved=warn_unresolved
        )
        scheme_reg = SchemeRegistry.from_yaml(schemes_path)
        
        return cls(adj_resolver, scheme_reg, warn_unresolved=warn_unresolved)
    
    @classmethod
    def from_default_resources(
        cls,
        warn_unresolved: bool = True
    ) -> "PersonalityResolver":
        """
        Load from default resource locations.
        
        Expects resources in src/rnb/resources/data/
        
        Args:
            warn_unresolved: If True, log warnings for unresolved adjectives
            
        Returns:
            Initialized PersonalityResolver
        """
        # Determine path relative to this file
        resources_dir = Path(__file__).parent / "data"
        
        return cls.from_yaml(
            neopiradj_path=resources_dir / "neopiradj.yaml",
            schemes_path=resources_dir / "schemes.yaml",
            warn_unresolved=warn_unresolved
        )
    
    def resolve(
        self, 
        adjectives: list[str]
    ) -> PersonalitySpecification:
        """
        Resolve a list of adjectives to a complete personality specification.
        
        Args:
            adjectives: List of personality adjectives
            
        Returns:
            PersonalitySpecification with all resolved positions
        """
        resolved: list[ResolvedAdjective] = []
        unresolved: list[str] = []
        ambiguities: list[tuple[str, list[TaxonomyPosition]]] = []
        
        for adj in adjectives:
            # Step 1: Resolve adjective to gloss(es)
            resolution = self.adjectives.resolve(adj)
            
            if not resolution.found:
                unresolved.append(adj)
                if self._warn_unresolved:
                    logger.warning(f"Adjective '{adj}' not found in RnB resource")
                continue
            
            # Track ambiguities (multiple distinct facets)
            if resolution.ambiguous:
                positions = []
                for m in resolution.mappings:
                    full_pos = self.schemes.get_position_for_gloss(m.gloss_id)
                    if full_pos:
                        positions.append(full_pos)
                    else:
                        positions.append(m.position)
                ambiguities.append((adj, positions))
                logger.debug(f"Adjective '{adj}' is ambiguous: {positions}")
            
            # Step 2: For each mapping, get full scheme information
            for mapping in resolution.mappings:
                # Get complete position from scheme registry
                full_position = self.schemes.get_position_for_gloss(mapping.gloss_id)
                
                if full_position is None:
                    # Gloss exists in neopiradj but not in schemes
                    # This shouldn't happen with consistent resources, but handle it
                    logger.debug(
                        f"Gloss {mapping.gloss_id} not found in schemes, "
                        f"using partial position"
                    )
                    full_position = mapping.position
                    scheme = None
                    all_glosses = []
                else:
                    # Get scheme info and all glosses for this pole
                    scheme = self.schemes.get_scheme(
                        full_position.trait,
                        full_position.facet,
                        full_position.scheme
                    )
                    
                    all_glosses = self.schemes.get_glosses_for_scheme(
                        full_position.trait,
                        full_position.facet,
                        full_position.scheme,
                        full_position.pole
                    ) if scheme else []
                
                # Create updated adjective entry with full position
                updated_entry = AdjectiveEntry(
                    word=mapping.word,
                    synset=mapping.synset,
                    weight=mapping.weight,
                    gloss_id=mapping.gloss_id,
                    gloss_text=mapping.gloss_text,
                    position=full_position
                )
                
                resolved.append(ResolvedAdjective(
                    adjective=updated_entry,
                    scheme=scheme,
                    position=full_position,
                    all_glosses_for_pole=all_glosses
                ))
        
        spec = PersonalitySpecification(
            input_adjectives=adjectives,
            resolved=resolved,
            unresolved=unresolved,
            ambiguities=ambiguities
        )
        
        logger.info(
            f"Resolved {len(adjectives)} adjectives: "
            f"{len(resolved)} mappings, {len(unresolved)} unresolved"
        )
        
        return spec
    
    def resolve_single(self, adjective: str) -> PersonalitySpecification:
        """
        Resolve a single adjective.
        
        Convenience method for single-adjective resolution.
        
        Args:
            adjective: A personality adjective
            
        Returns:
            PersonalitySpecification for the single adjective
        """
        return self.resolve([adjective])
    
    def suggest_adjectives(
        self, 
        trait: str,
        facet: Optional[str] = None,
        valence: Optional[str] = None,
        limit: int = 10
    ) -> list[str]:
        """
        Suggest adjectives for a given trait/facet.
        
        Useful for helping users discover relevant adjectives.
        
        Args:
            trait: Trait name
            facet: Optional facet name to narrow suggestions
            valence: "+" or "-" to filter by pole
            limit: Maximum suggestions to return
            
        Returns:
            List of adjective words, sorted by weight
        """
        if facet:
            entries = self.adjectives.get_adjectives_for_facet(trait, facet, valence)
        else:
            # Get all adjectives for trait
            entries = []
            for facet_name in self.schemes.get_facets_for_trait(trait):
                entries.extend(
                    self.adjectives.get_adjectives_for_facet(trait, facet_name, valence)
                )
        
        # Sort by weight, deduplicate
        seen = set()
        unique_entries = []
        for e in sorted(entries, key=lambda x: x.weight, reverse=True):
            if e.word not in seen:
                seen.add(e.word)
                unique_entries.append(e)
        
        return [e.word for e in unique_entries[:limit]]
    
    @property
    def statistics(self) -> dict:
        """
        Return combined statistics from both resources.
        
        Returns:
            Dictionary with adjective and scheme statistics
        """
        return {
            "adjectives": self.adjectives.statistics,
            "schemes": self.schemes.statistics
        }