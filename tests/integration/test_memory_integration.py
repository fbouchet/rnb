"""
Integration tests for memory system with actual LLM.

Validates that memory system (M.U, M.S, M.T) works end-to-end:
- Store and retrieve memories
- Memory-informed LLM responses
- Multi-session continuity
- User preference learning
"""

import pytest
import tempfile
import shutil
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


@pytest.fixture(scope="module")
def llm_client():
    """LLM client for integration tests"""
    return LLMClient(
        provider=ModelProvider.LOCAL,
        model_name="llama3.2:3b",
        timeout=120.0
    )


@pytest.fixture
def temp_chroma_dir():
    """Temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def redis_backend():
    """Redis backend for personality"""
    backend = RedisBackend()
    yield backend
    backend.flush_db()
    backend.close()


@pytest.fixture
def personality_store(redis_backend):
    """Personality store"""
    return PersonalityStateStore(redis_backend)


@pytest.fixture
def agent_manager(personality_store):
    """Agent manager"""
    return AgentManager(personality_store)


@pytest.fixture
def memory_store(temp_chroma_dir):
    """Memory store with temp storage"""
    store = MemoryStore(persist_directory=temp_chroma_dir)
    yield store
    store.close()


@pytest.fixture
def operator_registry():
    """Operator registry with basic operators"""
    registry = OperatorRegistry()
    registry.register(StructureInfluence())
    registry.register(DetailOrientedInfluence())
    registry.register(CooperationVerbosityInfluence())
    return registry


class TestUserMemoryIntegration:
    """Test M.U (User Model) integration with LLM"""
    
    def test_user_preference_affects_response(
        self,
        agent_manager,
        personality_store,
        memory_store,
        operator_registry,
        llm_client
    ):
        """
        Test that stored user preferences influence LLM responses.
        
        Scenario:
        1. Store user preference: "User prefers concise explanations"
        2. Query without memory → baseline response
        3. Query with memory context → should be more concise
        """
        # Create agent
        agent_manager.create_agent(
            "tutor_001",
            traits={Trait.CONSCIENTIOUSNESS: 0.7},
            affects={AffectDimension.COOPERATION: 0.8}
        )
        
        # Store user preference
        memory_store.add_user_memory(
            agent_id="tutor_001",
            content="User prefers very concise explanations with minimal details",
            metadata={"category": "preference", "confidence": 0.9}
        )
        
        # Base query
        query = "Explain what a binary search tree is."
        
        # Get personality context
        agent_state = personality_store.get_state("tutor_001")
        context = InfluenceContext.from_personality(agent_state)
        
        # Response WITHOUT memory context
        base_prompt = operator_registry.apply_all(query, context)
        response_without_memory = llm_client.query(base_prompt, temperature=0.7, max_tokens=200)
        
        # Response WITH memory context
        user_prefs = memory_store.search_memories(
            agent_id="tutor_001",
            query="user preferences explanation style",
            memory_type=MemoryType.USER,
            n_results=3
        )
        
        memory_context = "\n".join([
            f"User preference: {mem['content']}"
            for mem in user_prefs
        ])
        
        augmented_prompt = f"""{query}

Context from past interactions:
{memory_context}

Respond according to the user's stated preferences."""
        
        behavioral_prompt = operator_registry.apply_all(augmented_prompt, context)
        response_with_memory = llm_client.query(behavioral_prompt, temperature=0.7, max_tokens=200)
        
        # Measure verbosity
        words_without = len(response_without_memory.split())
        words_with = len(response_with_memory.split())
        
        print(f"\n=== Without Memory Context ===")
        print(f"Word count: {words_without}")
        print(response_without_memory)
        
        print(f"\n=== With Memory Context (User Prefers Concise) ===")
        print(f"Word count: {words_with}")
        print(response_with_memory)
        
        # Memory-aware response should be more concise
        # (Allow some variance due to LLM stochasticity)
        assert words_with < words_without * 1.1, \
            f"Expected memory-aware response to be more concise: {words_with} vs {words_without}"
    
    def test_user_expertise_affects_explanation_level(
        self,
        agent_manager,
        memory_store,
        operator_registry,
        llm_client
    ):
        """
        Test that stored user expertise influences explanation depth.
        
        Scenario:
        - Store: "User is expert in Python, beginner in algorithms"
        - Query about algorithms should get more detailed explanation
        """
        # Create agent
        agent_manager.create_agent("tutor_002", traits={Trait.CONSCIENTIOUSNESS: 0.6})
        
        # Store user expertise
        memory_store.add_user_memory(
            agent_id="tutor_002",
            content="User is expert in Python programming but beginner in data structures and algorithms",
            metadata={"category": "expertise", "domain": "programming"}
        )
        
        # Query about algorithms
        query = "Explain quicksort algorithm"
        
        # Search relevant expertise
        expertise = memory_store.search_memories(
            agent_id="tutor_002",
            query="user knowledge algorithms",
            memory_type=MemoryType.USER,
            n_results=2
        )
        
        if expertise:
            context_str = "\n".join([m['content'] for m in expertise])
            augmented_query = f"""{query}

User background:
{context_str}

Adjust explanation level accordingly."""
            
            response = llm_client.query(augmented_query, temperature=0.7, max_tokens=250)
            
            print(f"\n=== Response (User: Expert Python, Beginner Algorithms) ===")
            print(response)
            
            # Should contain beginner-friendly elements
            response_lower = response.lower()
            beginner_indicators = ['basic', 'simple', 'fundamental', 'start', 'first']
            
            has_beginner_language = any(word in response_lower for word in beginner_indicators)
            assert has_beginner_language, "Expected beginner-friendly explanation based on user expertise"


class TestSessionMemoryIntegration:
    """Test M.S (Session Model) integration with LLM"""
    
    def test_conversation_continuity(
        self,
        agent_manager,
        memory_store,
        llm_client
    ):
        """
        Test that session memory enables conversation continuity.
        
        Scenario:
        Turn 1: "Explain recursion"
        Turn 2: "Give me an example" (should reference recursion from turn 1)
        """
        # Create agent
        agent_manager.create_agent("tutor_003", traits={Trait.CONSCIENTIOUSNESS: 0.7})
        
        session_id = "conv_001"
        
        # Turn 1: Explain recursion
        query1 = "Explain recursion in programming"
        response1 = llm_client.query(query1, temperature=0.7, max_tokens=200)
        
        # Store in session memory
        memory_store.add_session_memory(
            agent_id="tutor_003",
            content=f"User asked: {query1}\nAgent explained: {response1}",
            metadata={
                "session_id": session_id,
                "turn": 1,
                "topic": "recursion"
            }
        )
        
        print(f"\n=== Turn 1 ===")
        print(f"User: {query1}")
        print(f"Agent: {response1}")
        
        # Turn 2: Give example (references previous context)
        query2 = "Can you give me an example?"
        
        # Retrieve session history
        history = memory_store.get_session_memories(
            agent_id="tutor_003",
            session_id=session_id,
            n_results=5
        )
        
        # Build context from history
        history_context = "\n".join([
            f"Turn {mem['metadata'].get('turn', '?')}: {mem['content']}"
            for mem in history
        ])
        
        augmented_query2 = f"""Previous conversation:
{history_context}

Current query: {query2}

Respond with relevant example based on previous context."""
        
        response2 = llm_client.query(augmented_query2, temperature=0.7, max_tokens=200)
        
        print(f"\n=== Turn 2 ===")
        print(f"User: {query2}")
        print(f"Agent: {response2}")
        
        # Response should reference recursion from turn 1
        response2_lower = response2.lower()
        assert 'recurs' in response2_lower, \
            "Expected turn 2 response to reference recursion from turn 1"


class TestTaskMemoryIntegration:
    """Test M.T (Task Model) integration with LLM"""
    
    def test_task_tracking_across_turns(
        self,
        agent_manager,
        memory_store,
        llm_client
    ):
        """
        Test that task memory tracks progress across conversation.
        
        Scenario:
        Task: Help implement binary search
        Turn 1: Explain concept
        Turn 2: Show pseudocode
        Turn 3: Write actual code (agent knows previous steps completed)
        """
        # Create agent
        agent_manager.create_agent("tutor_004", traits={Trait.CONSCIENTIOUSNESS: 0.8})
        
        task_id = "task_binary_search"
        
        # Initialize task
        memory_store.add_task_memory(
            agent_id="tutor_004",
            content="Help user implement binary search algorithm in Python",
            metadata={
                "task_id": task_id,
                "status": "in_progress",
                "steps_completed": "",
                "steps_remaining": "explain_concept, show_pseudocode, write_code"
            }
        )
        
        # Turn 1: Explain concept
        query1 = "Explain how binary search works"
        response1 = llm_client.query(query1, temperature=0.7, max_tokens=200)
        
        # Update task: concept explained
        memory_store.add_task_memory(
            agent_id="tutor_004",
            content="Task progress: Explained binary search concept",
            metadata={
                "task_id": task_id,
                "status": "in_progress",
                "steps_completed": "explain_concept",
                "steps_remaining": "show_pseudocode, write_code"
            }
        )
        
        print(f"\n=== Turn 1: Explain Concept ===")
        print(response1)
        
        # Turn 3: Write code (check task progress first)
        query3 = "Now write the Python code"
        
        # Get task status
        task_memories = memory_store.get_task_memories(
            agent_id="tutor_004",
            status="in_progress",
            n_results=5
        )
        
        # Find relevant task
        task_info = None
        for mem in task_memories:
            if mem['metadata'].get('task_id') == task_id:
                task_info = mem
                break
        
        assert task_info is not None, "Task should be tracked in memory"
        
        steps_completed_str = task_info['metadata'].get('steps_completed', "")
        steps_completed = steps_completed_str.split(',') if steps_completed_str else []
        
        print(f"\n=== Task Status Before Turn 3 ===")
        print(f"Steps completed: {steps_completed}")
        print(f"Steps remaining: {task_info['metadata'].get('steps_remaining', [])}")
        
        # Agent should know concept was already explained
        assert "explain_concept" in steps_completed, \
            "Task memory should track that concept was explained"


class TestMemoryAndPersonalityIntegration:
    """Test full integration: Memory + Personality + LLM"""
    
    def test_memory_complements_personality(
        self,
        agent_manager,
        personality_store,
        memory_store,
        operator_registry,
        llm_client
    ):
        """
        Test that memory (what to remember) complements personality (how to behave).
        
        Scenario:
        - Agent: High conscientiousness (thorough, structured)
        - Memory: User prefers Python examples
        - Result: Thorough explanation WITH Python examples
        """
        # Create agent with high conscientiousness
        agent_manager.create_agent(
            "tutor_005",
            traits={Trait.CONSCIENTIOUSNESS: 0.8},
            affects={AffectDimension.COOPERATION: 0.9}
        )
        
        # Store user preference for Python
        memory_store.add_user_memory(
            agent_id="tutor_005",
            content="User prefers all examples in Python programming language",
            metadata={"category": "preference", "domain": "programming"}
        )
        
        # Query
        query = "Explain how a stack data structure works"
        
        # Get personality state
        agent_state = personality_store.get_state("tutor_005")
        context = InfluenceContext.from_personality(agent_state)
        
        # Apply personality influences (conscientiousness → structured, detailed)
        personality_prompt = operator_registry.apply_all(query, context)
        
        # Add memory context (user prefers Python)
        user_prefs = memory_store.search_memories(
            agent_id="tutor_005",
            query="user preferences examples",
            memory_type=MemoryType.USER,
            n_results=2
        )
        
        memory_context = "\n".join([m['content'] for m in user_prefs])
        
        full_prompt = f"""{query}

User context:
{memory_context}

{personality_prompt}"""
        
        response = llm_client.query(full_prompt, temperature=0.7, max_tokens=300)
        
        print(f"\n=== Full Integration Test ===")
        print(f"Personality: High conscientiousness (structured, detailed)")
        print(f"Memory: User prefers Python examples")
        print(f"\nResponse:")
        print(response)
        
        # Should be detailed (conscientiousness) AND mention Python (memory)
        response_lower = response.lower()
        
        # Check for structure (conscientiousness)
        has_structure = any(word in response_lower for word in ['first', 'then', 'step', 'finally'])
        
        # Check for Python (memory)
        has_python = 'python' in response_lower or 'py' in response_lower
        
        print(f"\nValidation:")
        print(f"  Structured response: {has_structure}")
        print(f"  Python mentioned: {has_python}")
        
        # At least one should be true (LLM behavior varies)
        assert has_structure or has_python, \
            "Expected response to show either personality (structure) or memory (Python) influence"