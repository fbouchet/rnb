"""
RnB Resource Handling

This module provides access to the RnB personality resources:
- neopiradj: Maps 1055 personality adjectives to FFM/NEO PI-R taxonomy
- schemes: Defines 70 bipolar behavioral schemes with 766 glosses

The module implements Step 1 of the RnB pipeline:
    User adjectives → AdjectiveResolver → SchemeRegistry → PersonalitySpecification

Usage:
    from rnb.resources import PersonalityResolver

    # Load from default locations
    resolver = PersonalityResolver.from_default_resources()

    # Or specify paths explicitly
    resolver = PersonalityResolver.from_yaml(
        neopiradj_path="path/to/neopiradj.yaml",
        schemes_path="path/to/schemes.yaml"
    )

    # Resolve adjectives to personality specification
    spec = resolver.resolve(["romantic", "organized", "shy"])

    # Inspect results
    print(spec.summary())
    for r in spec.resolved:
        print(f"{r.adjective.word} → {r.position}")
        print(f"  Scheme: {r.scheme.name if r.scheme else 'N/A'}")
        print(f"  Glosses: {len(r.all_glosses_for_pole)}")

Resource Statistics:
    - 1055 unique personality adjectives
    - 766 glosses (WordNet + Goldberg questionnaire items)
    - 69 bipolar behavioral schemes
    - 30 NEO PI-R facets (6 per trait)
    - 5 FFM traits (OCEAN)

References:
    - Bouchet & Sansonnet (2010), "Implementing WordNet Personality
      Adjectives as Influences on Rational Agents", IJCISM
    - Bouchet & Sansonnet (2013), "Influence of FFM/NEO PI-R personality
      traits on the rational process of autonomous agents"
"""

from .adjective_resolver import AdjectiveResolver
from .models import (
    AdjectiveEntry,
    AdjectiveResolution,
    GlossEntry,
    PoleInfo,
    SchemeInfo,
    # Core data structures
    TaxonomyPosition,
    # Enums
    TraitName,
)
from .modifier_lexicon import (
    ModifierCategory,
    ModifierEntry,
    ModifierLexicon,
    ModifierType,
)
from .personality_resolver import (
    PersonalityResolver,
    PersonalitySpecification,
    ResolvedAdjective,
)
from .phrase_parser import ModifiedAdjective, PhraseParser
from .scheme_registry import SchemeRegistry

__all__ = [
    # Enums
    "TraitName",
    # Data models
    "TaxonomyPosition",
    "GlossEntry",
    "AdjectiveEntry",
    "AdjectiveResolution",
    "SchemeInfo",
    "PoleInfo",
    # Resolution results
    "ResolvedAdjective",
    "PersonalitySpecification",
    # Resolvers
    "AdjectiveResolver",
    "SchemeRegistry",
    "PersonalityResolver",
    # Modifier lexicon
    "ModifierLexicon",
    "ModifierEntry",
    "ModifierType",
    "ModifierCategory",
    # Phrase parser
    "PhraseParser",
    "ModifiedAdjective",
]
