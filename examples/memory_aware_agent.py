"""
Memory-Aware Agent Example - Full RnB Model M Demonstration

Demonstrates the complete RnB Model M:
- M.A: Agent personality (traits, moods, affects)
- M.U: User model (preferences, expertise)
- M.S: Session model (conversation history)
- M.T: Task model (goal tracking)

Shows how memory complements personality:
- Personality controls HOW the agent behaves
- Memory provides WHAT the agent remembers

Usage:
    poetry run python examples/memory_aware_agent.py
"""

from rnb.personality.backend import RedisBackend
from rnb.personality.store import PersonalityStateStore
from rnb.personality.manager import AgentManager
from rnb.personality.state import MoodDimension, AffectDimension
from rnb.personality.taxonomy import Trait
from rnb.influence.context import InfluenceContext
from rnb.influence.registry import OperatorRegistry
from rnb.influence.trait_based import StructureInfluence, DetailOrientedInfluence
from rnb.influence.affect_based import CooperationVerbosityInfluence
from rnb.memory import MemoryStore, MemoryType
from rnb.llm import LLMClient, ModelProvider



from rnb.logging import configure_logging
from rnb.console import rule, sep, blank, say_user, say_agent, say_system


def main() -> None:
    """Run memory-aware agent demonstration"""

    # Demo logging configuration (colors in console if Rich is installed).
    # Optional file logging: set env RNB_LOG_FILE=./rnb.log
    configure_logging(level="INFO", rich=True)

    rule("RnB Model M - Memory-Aware Agent")

    # ===== Initialize Components =====
    say_system("Initializing components...")

    # Personality (M.A)
    backend = RedisBackend()
    store = PersonalityStateStore(backend)
    manager = AgentManager(store)

    # Memory (M.U, M.S, M.T)
    memory = MemoryStore(persist_directory="./data/chroma")

    # LLM
    llm = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")

    # Operators
    registry = OperatorRegistry()
    registry.register(StructureInfluence())
    registry.register(DetailOrientedInfluence())
    registry.register(CooperationVerbosityInfluence())

    say_system("✓ All components initialized")

    # ===== Cleanup any existing agent from previous runs =====
    agent_id = "programming_tutor"

    blank(1)
    say_system("Cleaning up any existing agent state...")
    try:
        manager.delete_agent(agent_id)
        memory.clear_memories(agent_id)
        say_system(f"✓ Cleaned up existing agent '{agent_id}'")
    except Exception:
        say_system("✓ No existing agent to clean up")

    # ===== Create Agent with Personality =====
    rule("Creating Programming Tutor Agent")

    say_system("Personality Profile (M.A):")
    say_system("  - High Conscientiousness (0.7): Organized, thorough")
    say_system("  - High Cooperation (0.8): Helpful, supportive")

    manager.create_agent(
        agent_id=agent_id,
        traits={Trait.CONSCIENTIOUSNESS: 0.7},
        moods={MoodDimension.ENERGY: 0.6, MoodDimension.HAPPINESS: 0.5},
        affects={AffectDimension.COOPERATION: 0.8, AffectDimension.TRUST: 0.5},
    )

    # ===== Initialize User Model (M.U) =====
    rule("Building User Model (M.U)")

    say_system("Learning about user from initial interaction...")

    # Store user preferences
    memory.add_user_memory(
        agent_id=agent_id,
        content="User prefers Python programming language for all examples",
        metadata={"category": "preference", "confidence": 0.9},
    )
    say_system("✓ Stored: User prefers Python")

    memory.add_user_memory(
        agent_id=agent_id,
        content="User has intermediate programming knowledge, learning data structures",
        metadata={"category": "expertise", "domain": "programming"},
    )
    say_system("✓ Stored: User expertise level")

    memory.add_user_memory(
        agent_id=agent_id,
        content="User likes concise explanations followed by practical examples",
        metadata={"category": "learning_style", "confidence": 0.8},
    )
    say_system("✓ Stored: User learning style")

    blank(1)
    say_system(
        f"User model initialized with {memory.count_memories(agent_id, MemoryType.USER)} memories"
    )

    # ===== Conversation with Memory =====
    rule("Multi-Turn Conversation (M.S + M.T)")

    session_id = "session_001"
    task_id = "learn_binary_search"

    # Initialize task (M.T)
    memory.add_task_memory(
        agent_id=agent_id,
        content="Help user learn and implement binary search algorithm",
        metadata={
            "task_id": task_id,
            "status": "in_progress",
            "steps": "explain_concept, show_example, help_implement",
            "current_step": "explain_concept",
        },
    )

    turns = [
        {
            "turn": 1,
            "user_query": "Can you explain binary search?",
            "task_update": {"current_step": "show_example", "steps_completed": "explain_concept"},
        },
        {
            "turn": 2,
            "user_query": "Can you show me an example?",
            "task_update": {
                "current_step": "help_implement",
                "steps_completed": "explain_concept, show_example",
            },
        },
    ]

    for turn_data in turns:
        turn = turn_data["turn"]
        user_query = turn_data["user_query"]

        sep()
        say_system(f"Turn {turn}")
        blank(1)

        say_user(user_query)
        blank(1)

        # ===== Retrieve Relevant Memories =====

        # M.U: User preferences
        user_prefs = memory.search_memories(
            agent_id=agent_id,
            query=user_query,
            memory_type=MemoryType.USER,
            n_results=3,
        )

        # M.S: Session history
        session_history = memory.get_session_memories(
            agent_id=agent_id,
            session_id=session_id,
            n_results=5,
        )

        # M.T: Task status
        task_status = memory.get_task_memories(
            agent_id=agent_id,
            status="in_progress",
            n_results=3,
        )

        say_system("Retrieved memories:")
        say_system(f"  - M.U (User): {len(user_prefs)} preferences")
        say_system(f"  - M.S (Session): {len(session_history)} history items")
        say_system(f"  - M.T (Task): {len(task_status)} active tasks")
        blank(1)

        # ===== Build Context-Aware Prompt =====

        # User context (M.U)
        user_context = "\n".join([f"- {mem['content']}" for mem in user_prefs[:2]])

        # Session context (M.S)
        history_context = ""
        if session_history:
            history_context = "Previous conversation:\n" + "\n".join(
                [
                    f"Turn {mem['metadata'].get('turn', '?')}: {mem['content'][:100]}..."
                    for mem in session_history[:2]
                ]
            )

        # Task context (M.T)
        task_context = ""
        if task_status:
            task_info = task_status[0]
            steps_done = task_info["metadata"].get("steps_completed", [])
            # NOTE: In your original code, steps_completed is sometimes a string.
            # If it's a string, joining will iterate characters; keep behavior but guard.
            if isinstance(steps_done, str):
                steps_done_text = steps_done
            else:
                steps_done_text = ", ".join(steps_done)
            task_context = (
                f"Current task: {task_info['content']}\nSteps completed: {steps_done_text}"
            )

        memory_augmented_query = f"""{user_query}

Context:
User preferences:
{user_context}

{history_context}

{task_context}

Respond according to user preferences and task progress."""

        # Apply personality influences (M.A)
        agent_state = store.get_state(agent_id)
        context = InfluenceContext.from_personality(agent_state)
        behavioral_prompt = registry.apply_all(memory_augmented_query, context)

        # Query LLM
        say_system("Querying LLM...")
        blank(1)
        response = llm.query(behavioral_prompt, temperature=0.7, max_tokens=250)

        say_agent(response)
        blank(1)

        # ===== Store Interaction in Session Memory (M.S) =====
        memory.add_session_memory(
            agent_id=agent_id,
            content=f"User asked: {user_query}\nAgent responded: {response[:150]}...",
            metadata={
                "session_id": session_id,
                "turn": turn,
                "topic": "binary_search",
            },
        )

        # ===== Update Task Progress (M.T) =====
        if turn_data.get("task_update"):
            memory.add_task_memory(
                agent_id=agent_id,
                content=f"Task progress updated at turn {turn}",
                metadata={
                    "task_id": task_id,
                    "status": "in_progress",
                    **turn_data["task_update"],
                },
            )

    # ===== Final Summary =====
    rule("Memory System Summary")

    say_system("Final Memory Counts:")
    say_system(f"  M.U (User Model):    {memory.count_memories(agent_id, MemoryType.USER)} memories")
    say_system(f"  M.S (Session Model): {memory.count_memories(agent_id, MemoryType.SESSION)} memories")
    say_system(f"  M.T (Task Model):    {memory.count_memories(agent_id, MemoryType.TASK)} memories")

    blank(1)
    say_system("Key Observations:")
    say_system("1. Personality (M.A) remained stable: High conscientiousness throughout")
    say_system("2. User model (M.U) captured: Python preference, expertise, learning style")
    say_system("3. Session history (M.S) tracked: Multi-turn conversation context")
    say_system("4. Task tracking (M.T) monitored: Progress through learning objectives")
    say_system("5. Memory + Personality: Agent provided Python examples (memory)")
    say_system("   with structured explanations (personality)")

    # Cleanup
    sep()
    say_system("Cleaning up...")
    manager.delete_agent(agent_id)
    memory.clear_memories(agent_id)
    backend.close()
    memory.close()
    say_system("✓ Complete")

    rule("Demonstration Complete")


if __name__ == "__main__":
    main()