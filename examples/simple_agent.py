import os

from rnb.console import say_agent, say_model, say_system, say_user
from rnb.influence.operators import (
    CooperationInfluence,
    ExtraversionInfluence,
    MoodInfluence,
)
from rnb.llm.client import LLMClient, ModelProvider
from rnb.logging import configure_logging
from rnb.personality.state import AffectDimension, FFMTrait, MoodDimension
from rnb.personality.store import PersonalityStateStore


def main() -> None:
    # Configure logging once (console colors via Rich if available)
    # Optional file logging: set env RNB_LOG_FILE=./rnb.log
    configure_logging(level="INFO", rich=True)

    # Initialize components
    say_system("Initializing components...")
    store = PersonalityStateStore()

    llm = LLMClient(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create agent with specific personality
    agent_id = "tutor_001"
    say_system(f"Creating agent '{agent_id}' with personality profile")

    store.create_agent(
        agent_id=agent_id,
        traits={
            FFMTrait.EXTRAVERSION: 0.6,  # Moderately outgoing
            FFMTrait.CONSCIENTIOUSNESS: 0.8,  # Highly organized
            FFMTrait.AGREEABLENESS: 0.7,  # Very cooperative
        },
    )

    # Set initial affect (relationship with user)
    say_system("Setting initial affect (relationship state)")
    store.update_affect(
        agent_id,
        {
            AffectDimension.COOPERATION: 0.8,  # Very cooperative
            AffectDimension.TRUST: 0.5,  # Neutral trust
        },
    )

    # Set current mood
    say_system("Setting initial mood")
    store.update_mood(
        agent_id,
        {
            MoodDimension.ENERGY: 0.6,  # Good energy
            MoodDimension.HAPPINESS: 0.4,  # Slightly positive
        },
    )

    # Construct influenced prompt
    base_prompt = "Explain the concept of recursion in programming."
    say_user(base_prompt)

    # Apply influence operators
    state = store.get_state(agent_id)
    operators = [
        CooperationInfluence(),
        ExtraversionInfluence(),
        MoodInfluence(),
    ]

    influenced_prompt = base_prompt
    for operator in operators:
        if operator.applies(state):
            influenced_prompt = operator.apply(influenced_prompt, state)
            say_system(f"Applied influence: {operator.name}")

    say_model("")
    say_model("Final prompt after personality influences:")
    say_model(influenced_prompt)

    # Query LLM
    say_system("Querying LLM...")
    response = llm.query(influenced_prompt)

    say_agent(response)

    # Update state after interaction
    say_system("Updating agent state after interaction")
    store.increment_interaction(agent_id)
    store.update_mood(
        agent_id,
        {
            MoodDimension.SATISFACTION: 0.1,  # Positive interaction
        },
    )

    say_system("✓ Interaction complete")


if __name__ == "__main__":
    main()
