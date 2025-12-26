#!/usr/bin/env python
"""
Demo script showing RnB resources module usage.

Run from project root:
    PYTHONPATH=src python examples/demo_resources.py
"""

from rnb.console import rule, blank, say_agent, say_model, say_system
from rnb.logging import configure_logging
from rnb.resources import PersonalityResolver


def main() -> None:
    # Configure demo logging once.
    # Optional file logging: set env RNB_LOG_FILE=./rnb.log
    configure_logging(level="INFO", rich=True)

    rule("RnB Resources Module Demo", width=60)

    # Load resources from default location
    blank(1)
    say_system("1. Loading resources...")
    resolver = PersonalityResolver.from_default_resources(warn_unresolved=True)

    stats = resolver.statistics
    say_system(f"   Adjectives: {stats['adjectives']['unique_adjectives']}")
    say_system(f"   Schemes: {stats['schemes']['total_schemes']}")
    say_system(f"   Glosses: {stats['schemes']['total_glosses']}")

    # Resolve some adjectives
    blank(1)
    say_system("2. Resolving adjectives: ['romantic', 'organized', 'shy']")
    spec = resolver.resolve(["romantic", "organized", "shy"])

    blank(1)
    say_system("   Results:")
    say_system(f"   - Total mappings: {len(spec.resolved)}")
    say_system(f"   - Affected traits: {sorted(spec.affected_traits)}")
    say_system(f"   - Affected facets: {len(spec.affected_facets)}")

    # Show details for each resolved adjective
    blank(1)
    say_system("3. Detailed breakdown:")
    for r in spec.resolved:
        if r.has_scheme:  # Only show FFM mappings
            blank(1)
            say_system(f"   '{r.adjective.word}' (weight={r.adjective.weight})")
            say_system(f"   Position: {r.position}")
            say_system(f"   Scheme: {r.scheme.name} (op='{r.operator_hint}')")
            say_system(f"   Pole: {r.pole_name}")
            say_system(f"   Glosses in this pole: {len(r.all_glosses_for_pole)}")

            # Show first 2 glosses
            for g in r.all_glosses_for_pole[:2]:
                source = "Q" if g.source == "goldberg" else "WN"
                say_model(f"     [{source}] {g.text[:60]}...")

    # Test conflict scenario
    blank(1)
    say_system("4. Testing potential conflict: ['lazy', 'ambitious']")
    conflict_spec = resolver.resolve(["lazy", "ambitious"])

    say_system(f"   Resolved: {len(conflict_spec.resolved)} mappings")

    # Group by trait
    traits: dict[str, list] = {}
    for r in conflict_spec.resolved:
        if r.has_scheme:
            trait = r.position.trait
            traits.setdefault(trait, []).append(r)

    for trait, resolveds in traits.items():
        blank(1)
        say_system(f"   {trait}:")
        for r in resolveds:
            say_model(
                f"     - {r.adjective.word} → {r.position.facet}/{r.position.scheme}/{r.position.valence}"
            )

    # Search for adjectives
    blank(1)
    say_system("5. Searching for adjectives starting with 'friend':")
    matches = resolver.adjectives.search("friend", limit=5)
    say_agent(f"   {matches}")

    # Suggest adjectives for a facet
    blank(1)
    say_system("6. Suggesting adjectives for Openness/Fantasy (+):")
    suggestions = resolver.suggest_adjectives("Openness", facet="Fantasy", valence="+", limit=5)
    say_agent(f"   {suggestions}")

    blank(1)
    rule("Demo complete!", width=60)

if __name__ == "__main__":
    main()
