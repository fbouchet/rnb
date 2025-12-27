#!/usr/bin/env python3
"""
Demo: Gloss-Based Influence System

This script demonstrates the gloss-based influence system, which replaces
hardcoded behavioral templates with psychologically-grounded expressions
derived from WordNet synsets and Goldberg questionnaire items.

Key concepts demonstrated:
1. Creating personality from adjectives
2. Retrieving active glosses based on personality state
3. Generating behavioral context in different styles
4. Using GlossBasedInfluence operator for prompt modification

Run with:
    poetry run python examples/demo_gloss_influence.py
"""

from rnb.console import blank, rule, say_agent, say_model, say_system
from rnb.influence import (
    ContextStyle,
    GlossBasedInfluence,
    GlossInfluenceEngine,
    InfluenceContext,
    TraitGlossInfluence,
)
from rnb.logging import configure_logging
from rnb.personality import PersonalityStateFactory, Trait


def demo_basic_gloss_retrieval():
    """Demonstrate basic gloss retrieval from personality state"""
    rule("1. Basic Gloss Retrieval")

    # Create personality from adjectives
    factory = PersonalityStateFactory.from_default_resources()
    state = factory.from_adjectives("demo-agent", ["romantic", "organized", "friendly"])

    say_system("Created personality from adjectives:")
    say_system(f"  Adjectives: {state.source_adjectives}")
    say_system(f"  Schemes set: {state.num_schemes_set}")
    blank(1)

    # Show scheme values
    say_system("Active scheme values:")
    for key, value in state.schemes.items():
        if abs(value) > 0.1:  # Only show non-trivial values
            pole = "+" if value > 0 else "-"
            say_model(f"  {key}: {value:+.2f} ({pole} pole)")
    blank(1)

    # Create engine and retrieve glosses
    engine = GlossInfluenceEngine.from_default_resources()
    glosses = engine.get_active_glosses(state, threshold=0.3)

    say_system(f"Active glosses retrieved ({len(glosses)} total):")
    for gloss in glosses[:5]:  # Show first 5
        say_model(f'  • [{gloss.scheme_name}/{gloss.pole}] "{gloss.text}"')
        say_model(
            f"    Trait: {gloss.trait}, Facet: {gloss.facet}, Intensity: {gloss.intensity:.2f}"
        )


def demo_context_styles():
    """Demonstrate different context generation styles"""
    rule("2. Context Generation Styles")

    factory = PersonalityStateFactory.from_default_resources()
    state = factory.from_adjectives("style-demo", ["creative", "methodical"])

    engine = GlossInfluenceEngine.from_default_resources()

    styles = [
        (ContextStyle.DESCRIPTIVE, "Descriptive - 'You tend to be...'"),
        (ContextStyle.PRESCRIPTIVE, "Prescriptive - 'Be...'"),
        (ContextStyle.CONCISE, "Concise - Just keywords"),
        (ContextStyle.NARRATIVE, "Narrative - Woven sentences"),
    ]

    for style, description in styles:
        blank(1)
        say_system(f"--- {description} ---")
        blank(1)
        context = engine.generate_behavioral_context(
            state, style=style, max_total_glosses=4
        )
        say_agent(context if context else "(No context generated)")


def demo_operator_usage():
    """Demonstrate GlossBasedInfluence operator"""
    rule("3. GlossBasedInfluence Operator")

    # Create personality
    factory = PersonalityStateFactory.from_default_resources()
    state = factory.from_adjectives("operator-demo", ["imaginative", "thorough"])

    say_system("Personality:")
    say_system(f"  Adjectives: {state.source_adjectives}")
    say_system(f"  Openness: {state.get_trait(Trait.OPENNESS):+.2f}")
    say_system(f"  Conscientiousness: {state.get_trait(Trait.CONSCIENTIOUSNESS):+.2f}")
    blank(1)

    # Create operator
    op = GlossBasedInfluence.from_default_resources(
        threshold=0.3, style="descriptive", max_glosses=6
    )

    # Create context
    context = InfluenceContext.from_personality(state)

    # Check activation
    say_system("Operator activation:")
    say_system(f"  Name: {op.name}")
    say_system(f"  Applies: {op.applies(context)}")
    say_system(f"  Priority: {op.get_activation_priority()}")
    blank(1)

    # Apply to prompt
    base_prompt = "Explain how recursion works in programming."
    modified_prompt = op.apply(base_prompt, context)

    say_system("Base prompt:")
    say_user_text = base_prompt  # keep variable name readable for debugging
    # We don't import say_user here; base prompt is not user input in this demo.
    say_model(f"  {say_user_text}")
    blank(1)

    say_system("Modified prompt:")
    say_agent(modified_prompt)


def demo_trait_specific_operators():
    """Demonstrate trait-specific operators"""
    rule("4. Trait-Specific Operators")

    # Create mixed personality
    factory = PersonalityStateFactory.from_default_resources()
    state = factory.from_traits(
        "trait-demo",
        traits={
            Trait.OPENNESS: 0.8,  # High
            Trait.CONSCIENTIOUSNESS: 0.6,  # Moderate-high
            Trait.EXTRAVERSION: 0.2,  # Low (won't activate)
        },
    )

    say_system("Personality traits:")
    for trait in Trait:
        value = state.get_trait(trait)
        say_model(f"  {trait.value}: {value:+.2f}")
    blank(1)

    context = InfluenceContext.from_personality(state)

    # Test each trait operator
    operators = [
        TraitGlossInfluence(trait=Trait.OPENNESS, threshold=0.3),
        TraitGlossInfluence(trait=Trait.CONSCIENTIOUSNESS, threshold=0.3),
        TraitGlossInfluence(trait=Trait.EXTRAVERSION, threshold=0.3),
    ]

    say_system("Trait operator activation:")
    for op in operators:
        applies = op.applies(context)
        status = "✓ Active" if applies else "✗ Inactive"
        say_model(f"  {op.name}: {status}")


def demo_archetype_influence():
    """Demonstrate influence generation from archetypes"""
    rule("5. Archetype-Based Influence")

    factory = PersonalityStateFactory.from_default_resources()
    engine = GlossInfluenceEngine.from_default_resources()

    archetypes = ["creative_thinker", "detail_oriented", "warm_social"]

    for archetype in archetypes:
        blank(1)
        say_system(f"--- Archetype: {archetype} ---")
        blank(1)

        try:
            state = factory.create_from_archetype(f"archetype-{archetype}", archetype)

            # Show trait values
            say_system("Trait values:")
            for trait in Trait:
                value = state.get_trait(trait)
                if abs(value) > 0.3:
                    say_model(f"  {trait.value}: {value:+.2f}")

            # Generate behavioral context
            context = engine.generate_behavioral_context(
                state, style=ContextStyle.CONCISE
            )
            blank(1)
            say_system("Behavioral context:")
            say_agent(context if context else "(No strong personality traits)")

        except ValueError as e:
            say_system(f"Archetype not found: {e}")


def demo_comparison_old_vs_new():
    """Compare old hardcoded vs new gloss-based approach"""
    rule("6. Old vs New Approach Comparison")

    factory = PersonalityStateFactory.from_default_resources()
    state = factory.from_adjectives(
        "comparison-demo", ["organized", "thorough", "careful"]
    )

    say_system(f"Personality: {state.source_adjectives}")
    blank(1)

    # Old approach (simulated hardcoded template)
    say_system("OLD: Hardcoded template")
    old_addition = """Be thorough and systematic in your response.
Include relevant details and caveats.
Structure your answer clearly."""
    say_model(old_addition)
    blank(1)

    # New approach (gloss-based)
    engine = GlossInfluenceEngine.from_default_resources()
    new_addition = engine.generate_behavioral_context(
        state,
        style=ContextStyle.DESCRIPTIVE,
        max_total_glosses=5,
        include_header=False,
        include_footer=False,
    )

    say_system("NEW: Gloss-based (from RnB resource)")
    say_agent(new_addition if new_addition else "(No glosses found)")
    blank(1)

    say_system("Key difference:")
    say_model("  • Old: Generic, hardcoded behavioral instructions")
    say_model(
        "  • New: Authentic expressions from WordNet/Goldberg psychology resources"
    )
    say_model("  • New: Automatically adapted to specific scheme values")
    say_model("  • New: Traceable to original personality adjectives")


def main():
    """Run all demonstrations"""
    # Configure demo logging once.
    # Optional file logging: set env RNB_LOG_FILE=./rnb.log
    configure_logging(level="INFO", rich=True)

    #  Big banner (system channel)
    blank(1)
    say_system(
        "╔══════════════════════════════════════════════════════════════════════╗"
    )
    say_system(
        "║           RnB Gloss-Based Influence System Demonstration             ║"
    )
    say_system(
        "╚══════════════════════════════════════════════════════════════════════╝"
    )
    blank(1)

    demo_basic_gloss_retrieval()
    demo_context_styles()
    demo_operator_usage()
    demo_trait_specific_operators()
    demo_archetype_influence()
    demo_comparison_old_vs_new()

    rule("Demo Complete")
    say_system("The gloss-based influence system provides psychologically-grounded")
    say_system(
        "behavioral expressions derived from the RnB WordNet/Goldberg resources,"
    )
    say_system("replacing hardcoded templates with authentic personality definitions.")


if __name__ == "__main__":
    main()
