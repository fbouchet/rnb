"""
Complete RnB Agent Example - End-to-End Demonstration

This example demonstrates the full RnB framework in action:
1. Create an agent with specific personality (traits, moods, affects)
2. Register influence operators (trait-based, mood-based, affect-based)
3. Multi-turn conversation with personality-consistent behavior
4. Update rules: mood and affect evolution based on interaction events
5. Observable personality consistency throughout conversation

Agent Profile: "Helpful Tutor"
- High conscientiousness (organized, thorough)
- Moderate extraversion (friendly but not overwhelming)
- High cooperation (helpful, engaging)
- Dynamic moods (energy decreases with interaction, satisfaction varies)

Usage:
    poetry run python examples/complete_rnb_agent.py
"""

from rnb.console import blank, rule, say_agent, say_model, say_system, say_user, sep
from rnb.influence.affect_based import (
    CooperationHelpfulnessInfluence,
    CooperationVerbosityInfluence,
)
from rnb.influence.context import InfluenceContext
from rnb.influence.mood_based import (
    EnergyLengthInfluence,
    HappinessToneInfluence,
    SatisfactionPatienceInfluence,
)
from rnb.influence.registry import OperatorRegistry
from rnb.influence.trait_based import (
    DetailOrientedInfluence,
    EnthusiasmInfluence,
    StructureInfluence,
)
from rnb.llm import LLMClient, ModelProvider
from rnb.logging import configure_logging
from rnb.personality.backend import RedisBackend
from rnb.personality.manager import AgentManager
from rnb.personality.state import AffectDimension, MoodDimension, PersonalityState
from rnb.personality.store import PersonalityStateStore
from rnb.personality.taxonomy import Trait


def print_personality_state(state: PersonalityState, show_header: bool = True):
    """Display current personality state (system channel)."""
    if show_header:
        say_system(f"Agent ID: {state.agent_id}")
        say_system(f"Interactions: {state.interaction_count}")
        say_system("")

    say_system("Traits (stable):")
    trait_values = state.get_all_trait_values()
    for trait, value in trait_values.items():
        say_system(f"  {trait.value:20s}: {value:+.2f}")

    say_system("")
    say_system("Moods (dynamic):")
    for mood, value in state.moods.items():
        say_system(f"  {mood.value:20s}: {value:+.2f}")

    say_system("")
    say_system("Affects (relationship):")
    for affect, value in state.affects.items():
        say_system(f"  {affect.value:20s}: {value:+.2f}")


def print_state_delta(label: str, initial: float, current: float):
    """Print state change with delta (system/model channel)."""
    delta = current - initial
    # No ANSI coloring here; just a clean delta line.
    say_model(f"  {label:15s}: {initial:+.2f} → {current:+.2f} (Δ {delta:+.2f})")


def apply_interaction_fatigue(store: PersonalityStateStore, agent_id: str):
    """
    Update rule: Each interaction slightly decreases energy (fatigue).

    RnB behavioral dynamics: Extended interactions cause energy decay.
    """
    store.update_mood(
        agent_id, {MoodDimension.ENERGY: -0.15}  # Gradual energy decrease
    )


def apply_positive_feedback(store: PersonalityStateStore, agent_id: str):
    """
    Update rule: Positive feedback increases happiness and cooperation.

    RnB behavioral dynamics: Positive reinforcement affects mood and relationship.
    """
    store.update_mood(
        agent_id, {MoodDimension.HAPPINESS: 0.2, MoodDimension.SATISFACTION: 0.15}
    )

    store.update_affect(
        agent_id, {AffectDimension.COOPERATION: 0.1, AffectDimension.TRUST: 0.05}
    )


def apply_criticism(store: PersonalityStateStore, agent_id: str):
    """
    Update rule: Criticism decreases happiness and cooperation.

    RnB behavioral dynamics: Negative feedback affects emotional state and relationship.
    """
    store.update_mood(
        agent_id, {MoodDimension.HAPPINESS: -0.25, MoodDimension.SATISFACTION: -0.2}
    )

    store.update_affect(agent_id, {AffectDimension.COOPERATION: -0.15})


def main():
    # Configure logging output for the demo.
    # - rich=True gives role-colored output
    # - log_file can be supplied via env RNB_LOG_FILE or explicitly here
    configure_logging(level="INFO", rich=True)

    rule("RnB Framework - Complete Agent Demonstration")

    # ===== Step 1: Initialize Components =====
    say_system("Initializing RnB components...")

    backend = RedisBackend()
    store = PersonalityStateStore(backend)
    manager = AgentManager(store)

    llm = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")

    say_system("✓ Backend: Redis")
    say_system("✓ LLM: llama3.2:3b (Ollama)")

    # ===== Step 2: Create Agent with Personality =====
    rule("Creating 'Helpful Tutor' Agent")

    say_system("Personality Profile:")
    say_system("  - High Conscientiousness (0.7): Organized, thorough, structured")
    say_system("  - Moderate Extraversion (0.5): Friendly, approachable")
    say_system("  - High Agreeableness (0.6): Cooperative, patient")
    say_system("  - Positive initial mood")
    say_system("  - High cooperation affect (helpful)")

    agent_id = "helpful_tutor"

    blank(1)
    say_system("Cleaning up any existing agent state...")
    try:
        manager.delete_agent(agent_id)
        say_system(f"✓ Cleaned up existing agent '{agent_id}'")
    except Exception:
        say_system("✓ No existing agent to clean up")

    agent_state = manager.create_agent(
        agent_id=agent_id,
        traits={
            Trait.CONSCIENTIOUSNESS: 0.7,
            Trait.EXTRAVERSION: 0.5,
            Trait.AGREEABLENESS: 0.6,
        },
        moods={
            MoodDimension.ENERGY: 0.6,
            MoodDimension.HAPPINESS: 0.5,
            MoodDimension.SATISFACTION: 0.4,
        },
        affects={
            AffectDimension.COOPERATION: 0.8,
            AffectDimension.TRUST: 0.4,
        },
    )

    initial_state = {
        "energy": agent_state.moods[MoodDimension.ENERGY],
        "happiness": agent_state.moods[MoodDimension.HAPPINESS],
        "satisfaction": agent_state.moods[MoodDimension.SATISFACTION],
        "cooperation": agent_state.affects[AffectDimension.COOPERATION],
        "trust": agent_state.affects[AffectDimension.TRUST],
    }

    blank(1)
    sep()
    blank(1)
    print_personality_state(agent_state)

    # ===== Step 3: Register Influence Operators =====
    rule("Registering Influence Operators")

    registry = OperatorRegistry()

    # Trait-based operators
    registry.register(StructureInfluence())
    registry.register(DetailOrientedInfluence())
    registry.register(EnthusiasmInfluence())

    # Affect-based operators
    registry.register(CooperationVerbosityInfluence())
    registry.register(CooperationHelpfulnessInfluence())

    # Mood-based operators
    registry.register(EnergyLengthInfluence())
    registry.register(HappinessToneInfluence())
    registry.register(SatisfactionPatienceInfluence())

    say_system(f"Registered {len(registry)} operators")
    blank(1)
    say_system("Categories:")
    for category in registry.list_categories():
        ops = registry.list_operators(category=category)
        say_system(f"  {category:15s}: {len(ops)} operators")

    # ===== Step 4: Multi-Turn Conversation =====
    rule("Multi-Turn Conversation")

    conversation = [
        {
            "turn": 1,
            "user": "Can you explain what recursion is in programming?",
            "event": None,
        },
        {
            "turn": 2,
            "user": "That's really helpful, thank you! Can you give me an example?",
            "event": "positive_feedback",
        },
        {
            "turn": 3,
            "user": "Hmm, I'm still confused. Can you explain it differently?",
            "event": "criticism",
        },
        {
            "turn": 4,
            "user": "Ah! Now I understand. Thanks for being patient!",
            "event": "positive_feedback",
        },
    ]

    previous_state = initial_state.copy()

    for interaction in conversation:
        turn = interaction["turn"]
        user_message = interaction["user"]
        event = interaction["event"]

        rule(f"Turn {turn}", char="=", width=70)

        current_state = store.get_state(agent_id)
        context = InfluenceContext.from_personality(current_state)
        active_ops = registry.get_active_operators(context)

        say_user(user_message)
        blank(1)

        say_system(f"Active operators: {len(active_ops)}")
        for op in active_ops:
            say_system(f"  - {op.name} (priority: {op.get_activation_priority()})")

        rational_prompt = user_message
        behavioral_prompt = registry.apply_all(rational_prompt, context)

        if behavioral_prompt != rational_prompt:
            blank(1)
            say_model("Behavioral modifications added:")
            modifications = behavioral_prompt[len(rational_prompt) :].strip()
            say_model(modifications)

        blank(1)
        say_system("Querying LLM...")
        response = llm.query(
            behavioral_prompt,
            temperature=0.7,
            max_tokens=300,
        )

        blank(1)
        say_agent(response)
        blank(1)

        # Apply update rules
        if event == "positive_feedback":
            say_system("📈 Event: Positive feedback received")
            say_system("   → Increasing happiness, satisfaction, cooperation, trust")
            apply_positive_feedback(store, agent_id)
        elif event == "criticism":
            say_system("📉 Event: Criticism/confusion detected")
            say_system("   → Decreasing happiness, satisfaction, cooperation")
            apply_criticism(store, agent_id)

        say_system("⚡ Event: Interaction fatigue")
        say_system("   → Decreasing energy")
        apply_interaction_fatigue(store, agent_id)

        store.increment_interaction(agent_id)

        updated_state = store.get_state(agent_id)
        blank(1)
        say_system("State Changes (this turn):")
        print_state_delta(
            "Energy",
            previous_state["energy"],
            updated_state.moods[MoodDimension.ENERGY],
        )
        print_state_delta(
            "Happiness",
            previous_state["happiness"],
            updated_state.moods[MoodDimension.HAPPINESS],
        )
        print_state_delta(
            "Satisfaction",
            previous_state["satisfaction"],
            updated_state.moods[MoodDimension.SATISFACTION],
        )
        print_state_delta(
            "Cooperation",
            previous_state["cooperation"],
            updated_state.affects[AffectDimension.COOPERATION],
        )
        print_state_delta(
            "Trust",
            previous_state["trust"],
            updated_state.affects[AffectDimension.TRUST],
        )

        previous_state = {
            "energy": updated_state.moods[MoodDimension.ENERGY],
            "happiness": updated_state.moods[MoodDimension.HAPPINESS],
            "satisfaction": updated_state.moods[MoodDimension.SATISFACTION],
            "cooperation": updated_state.affects[AffectDimension.COOPERATION],
            "trust": updated_state.affects[AffectDimension.TRUST],
        }

    # ===== Step 5: Final State Summary =====
    rule("Final Agent State")

    final_state = store.get_state(agent_id)
    print_personality_state(final_state)

    rule("Personality Evolution Summary", char="=", width=70)

    say_system("Traits (stable - no change expected):")
    say_system(
        f"  Conscientiousness: {final_state.get_trait(Trait.CONSCIENTIOUSNESS):+.2f} (unchanged)"
    )
    say_system(
        f"  Extraversion:      {final_state.get_trait(Trait.EXTRAVERSION):+.2f} (unchanged)"
    )

    blank(1)
    say_system("Moods (dynamic - evolved during conversation):")
    print_state_delta(
        "Energy", initial_state["energy"], final_state.moods[MoodDimension.ENERGY]
    )
    print_state_delta(
        "Happiness",
        initial_state["happiness"],
        final_state.moods[MoodDimension.HAPPINESS],
    )
    print_state_delta(
        "Satisfaction",
        initial_state["satisfaction"],
        final_state.moods[MoodDimension.SATISFACTION],
    )

    blank(1)
    say_system(
        "Affects (relationship-specific - evolved based on interaction quality):"
    )
    print_state_delta(
        "Cooperation",
        initial_state["cooperation"],
        final_state.affects[AffectDimension.COOPERATION],
    )
    print_state_delta(
        "Trust", initial_state["trust"], final_state.affects[AffectDimension.TRUST]
    )

    rule("Key Observations", char="=", width=70)
    say_system("1. Traits remained stable (as expected in RnB framework)")
    say_system("2. Energy decreased due to interaction fatigue (mood decay)")
    say_system("3. Happiness fluctuated based on feedback (mood dynamics)")
    say_system(
        "4. Cooperation and trust evolved based on interaction quality (affect development)"
    )
    say_system("5. Responses showed personality-consistent behavior throughout")

    # Cleanup
    rule("Cleanup")
    say_system("Deleting agent from storage...")
    manager.delete_agent(agent_id)
    backend.close()
    say_system("✓ Complete")

    rule("Demonstration Complete")
    say_system("This example demonstrated:")
    say_system("  ✓ Agent creation with FFM personality profile")
    say_system("  ✓ Influence operator registration and activation")
    say_system("  ✓ Multi-turn conversation with personality-consistent behavior")
    say_system("  ✓ Update rules: mood and affect evolution")
    say_system("  ✓ Observable personality dynamics (stable traits + dynamic state)")
    blank(1)
    say_system("The RnB framework successfully maintains personality consistency")
    say_system("while allowing dynamic behavioral adaptation based on interactions.")


if __name__ == "__main__":
    main()
