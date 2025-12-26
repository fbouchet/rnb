#!/usr/bin/env python
"""
Demo script showing RnB personality module usage.

Run from project root:
    PYTHONPATH=src python examples/demo_personality.py
"""

from rnb.console import rule, sep, blank, say_agent, say_model, say_system
from rnb.logging import configure_logging
from rnb.personality import (
    ARCHETYPES,
    AffectDimension,
    MoodDimension,
    PersonalityState,
    PersonalityStateFactory,
    Trait,
)

def main() -> None:
    # Configure demo logging once.
    # Optional file logging: set env RNB_LOG_FILE=./rnb.log
    configure_logging(level="INFO", rich=True)

    rule("RnB Personality Module Demo (Step 2)", width=60)

    # Load factory with default resources
    blank(1)
    say_system("1. Loading personality factory...")
    factory = PersonalityStateFactory.from_default_resources()
    say_system(f"   Taxonomy: {factory.taxonomy.num_schemes} schemes loaded")

    # ===== Create from adjectives =====
    blank(1)
    say_system("2. Creating state from adjectives: ['romantic', 'organized', 'shy']")
    state = factory.from_adjectives("agent-adjectives", ["romantic", "organized", "shy"])

    blank(1)
    say_system("   Result:")
    say_system(f"   - Schemes set: {state.num_schemes_set}")
    say_system(f"   - Source adjectives: {state.source_adjectives}")

    blank(1)
    say_system("   Trait values (aggregated from schemes):")
    for trait in Trait:
        value = state.get_trait(trait)
        if value != 0:
            bar = (
                "+" * int(abs(value) * 10)
                if value > 0
                else "-" * int(abs(value) * 10)
            )
            say_model(f"     {trait.value:20} {value:+.2f} [{bar}]")

    # Show specific schemes
    blank(1)
    say_system("   Sample scheme values:")
    for key, value in list(state.schemes.items())[:5]:
        say_model(f"     {key}: {value:+.2f}")

    # ===== Create from traits (coarse) =====
    blank(1)
    say_system("3. Creating state from traits (high-level):")
    say_system("   {Openness: 0.8, Conscientiousness: 0.6, Neuroticism: -0.5}")

    state2 = factory.from_traits(
        "agent-traits",
        {
            Trait.OPENNESS: 0.8,
            Trait.CONSCIENTIOUSNESS: 0.6,
            Trait.NEUROTICISM: -0.5,
        },
    )

    blank(1)
    say_system("   All Openness schemes set to 0.8:")
    openness_schemes = state2.get_schemes_for_trait(Trait.OPENNESS)
    say_system(f"   - {len(openness_schemes)} schemes under Openness")

    # ===== Create from mixed levels =====
    blank(1)
    say_system("4. Creating from mixed levels:")
    say_system("   - Trait default: Openness = 0.5")
    say_system("   - Facet override: Openness_Fantasy = 0.9")
    say_system("   - Adjective: 'organized'")

    state3 = factory.from_mixed(
        "agent-mixed",
        traits={Trait.OPENNESS: 0.5},
        facets={"Openness_Fantasy": 0.9},
        adjectives=["organized"],
    )

    blank(1)
    say_system("   Result:")
    say_system(
        f"   - Fantasy facet: {state3.get_facet('Openness', 'Fantasy'):.2f} (overridden)"
    )
    say_system(
        f"   - Aesthetics facet: {state3.get_facet('Openness', 'Aesthetics'):.2f} (from trait)"
    )
    say_system(
        f"   - Conscientiousness: {state3.get_trait(Trait.CONSCIENTIOUSNESS):.2f} (from adjective)"
    )

    # ===== Create from archetype =====
    blank(1)
    say_system("5. Creating from predefined archetype: 'creative_thinker'")

    state4 = factory.create_from_archetype("agent-archetype", "creative_thinker")

    blank(1)
    say_system("   Archetype traits:")
    for trait, expected in ARCHETYPES["creative_thinker"]["traits"].items():
        actual = state4.get_trait(trait)
        say_model(f"     {trait.value:20} expected={expected:+.2f}, actual={actual:+.2f}")

    # ===== Mood and affect operations =====
    blank(1)
    say_system("6. Dynamic mood and affect updates:")

    state5 = factory.from_adjectives("agent-dynamic", ["friendly"])
    say_system(f"   Initial happiness: {state5.get_mood(MoodDimension.HAPPINESS):.2f}")

    state5.set_mood(MoodDimension.HAPPINESS, 0.7)
    say_system(
        f"   After positive interaction: {state5.get_mood(MoodDimension.HAPPINESS):.2f}"
    )

    state5.update_mood(MoodDimension.HAPPINESS, -0.2)
    say_system(
        f"   After minor frustration: {state5.get_mood(MoodDimension.HAPPINESS):.2f}"
    )

    state5.set_affect(AffectDimension.TRUST, 0.6)
    say_system(f"   Trust in current user: {state5.get_affect(AffectDimension.TRUST):.2f}")

    # ===== Serialization =====
    blank(1)
    say_system("7. Serialization for storage:")

    data = state.to_dict()
    say_system(f"   Serialized keys: {list(data.keys())}")
    say_system(f"   Schemes in storage: {len(data['schemes'])}")

    restored = PersonalityState.from_dict(data)
    say_system(f"   Restored successfully: {restored.agent_id}")

    # ===== Summary =====
    blank(1)
    say_system("8. State summary:")
    # summary() might already contain multi-line text; keep as a single agent message
    say_agent(state.summary())

    blank(1)
    rule("Demo complete!", width=60)

    # Show available archetypes
    blank(1)
    say_system(f"Available archetypes: {list(ARCHETYPES.keys())}")


if __name__ == "__main__":
    main()
