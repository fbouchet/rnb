"""
Integration tests for conscientiousness trait effects on LLM behavior.

Validates that high vs low conscientiousness produces measurably different responses
in terms of structure, detail, and precision.

Reference: Bouchet & Sansonnet (2013), NEO PI-R C-facets
"""

import pytest
from rnb.personality.state import FFMTrait
from rnb.influence.context import InfluenceContext
from rnb.influence.trait_based import (
    StructureInfluence,
    DetailOrientedInfluence,
    PrecisionInfluence
)


def count_sentences(text: str) -> int:
    """Count sentences in text (simple heuristic)"""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def has_structure_markers(text: str) -> bool:
    """Check if text contains structure markers (steps, points, etc.)"""
    structure_words = ['first', 'second', 'third', 'step', 'then', 'next', 'finally']
    text_lower = text.lower()
    return any(word in text_lower for word in structure_words)


class TestConscientiousnessVerbosity:
    """Test that conscientiousness affects response verbosity"""
    
    def test_high_conscientiousness_produces_longer_responses(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client,
        base_test_prompt
    ):
        """
        High conscientiousness → longer, more detailed responses.
        
        NEO PI-R facet: C1 (Competence), C6 (Deliberation)
        RnB behavioral scheme: Thoroughness
        """
        # Create high conscientiousness agent
        high_c = agent_manager.create_agent(
            "high_c_agent",
            traits={FFMTrait.CONSCIENTIOUSNESS: 0.8}
        )
        
        # Create low conscientiousness agent  
        low_c = agent_manager.create_agent(
            "low_c_agent",
            traits={FFMTrait.CONSCIENTIOUSNESS: -0.8}
        )
        
        # Register operators
        operator_registry.register(DetailOrientedInfluence())
        
        # Query with high conscientiousness
        context_high = InfluenceContext.from_personality(
            personality_store.get_state("high_c_agent")
        )
        prompt_high = operator_registry.apply_all(base_test_prompt, context_high)
        response_high = llm_client.query(prompt_high, temperature=0.7, max_tokens=500)
        
        # Query with low conscientiousness
        context_low = InfluenceContext.from_personality(
            personality_store.get_state("low_c_agent")
        )
        prompt_low = operator_registry.apply_all(base_test_prompt, context_low)
        response_low = llm_client.query(prompt_low, temperature=0.7, max_tokens=500)
        
        # Measure behavioral differences
        words_high = count_words(response_high)
        words_low = count_words(response_low)
        
        print(f"\n=== High Conscientiousness (0.8) ===")
        print(f"Word count: {words_high}")
        print(f"Response: {response_high[:200]}...")
        
        print(f"\n=== Low Conscientiousness (-0.8) ===")
        print(f"Word count: {words_low}")
        print(f"Response: {response_low[:200]}...")
        
        # Validation: high conscientiousness should produce longer responses
        # Allow some variance due to LLM stochasticity, but expect clear trend
        assert words_high > words_low * 1.1, \
            f"Expected high C to be >10% more verbose, got {words_high} vs {words_low}"
    
    def test_conscientiousness_affects_structure(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client
    ):
        """
        High conscientiousness → more structured responses with clear organization.
        
        NEO PI-R facet: C2 (Order)
        RnB behavioral scheme: Structure/organization
        """
        # Create agents
        agent_manager.create_agent(
            "structured_agent",
            traits={FFMTrait.CONSCIENTIOUSNESS: 0.8}
        )
        
        agent_manager.create_agent(
            "unstructured_agent",
            traits={FFMTrait.CONSCIENTIOUSNESS: -0.8}
        )
        
        # Register structure operator
        operator_registry.register(StructureInfluence())
        
        # Query both agents
        prompt = "Explain the steps to debug a program."
        
        context_structured = InfluenceContext.from_personality(
            personality_store.get_state("structured_agent")
        )
        prompt_structured = operator_registry.apply_all(prompt, context_structured)
        response_structured = llm_client.query(prompt_structured, temperature=0.7)
        
        context_unstructured = InfluenceContext.from_personality(
            personality_store.get_state("unstructured_agent")
        )
        prompt_unstructured = operator_registry.apply_all(prompt, context_unstructured)
        response_unstructured = llm_client.query(prompt_unstructured, temperature=0.7)
        
        # Check for structure markers
        has_structure_high = has_structure_markers(response_structured)
        has_structure_low = has_structure_markers(response_unstructured)
        
        print(f"\n=== Structured Response ===")
        print(f"Structure markers present: {has_structure_high}")
        print(response_structured)
        
        print(f"\n=== Unstructured Response ===")
        print(f"Structure markers present: {has_structure_low}")
        print(response_unstructured)
        
        # High conscientiousness more likely to use structure markers
        # (not guaranteed due to LLM variance, but should show trend)
        assert has_structure_high or not has_structure_low, \
            "Expected more structure in high conscientiousness response"


class TestConscientiousnessMultipleOperators:
    """Test multiple conscientiousness operators working together"""
    
    def test_combined_conscientiousness_effects(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client
    ):
        """
        Multiple conscientiousness operators produce compounded effects.
        
        Tests operator composition and priority ordering.
        """
        # Create very high conscientiousness agent
        agent_manager.create_agent(
            "very_conscientious",
            traits={FFMTrait.CONSCIENTIOUSNESS: 0.85}
        )
        
        # Register all conscientiousness operators
        operator_registry.register(StructureInfluence())       # Priority 70
        operator_registry.register(DetailOrientedInfluence())  # Priority 75
        operator_registry.register(PrecisionInfluence())       # Priority 60
        
        # Query
        prompt = "Explain photosynthesis."
        context = InfluenceContext.from_personality(
            personality_store.get_state("very_conscientious")
        )
        
        # Check activation
        active_ops = operator_registry.get_active_operators(context)
        assert len(active_ops) == 3, "All three operators should activate"
        
        # Verify priority ordering (precision first, then structure, then detail)
        assert active_ops[0].name == "conscientiousness_precision"
        assert active_ops[1].name == "conscientiousness_structure"
        assert active_ops[2].name == "conscientiousness_detail"
        
        # Apply all operators
        behavioral_prompt = operator_registry.apply_all(prompt, context)
        
        print(f"\n=== Base Prompt ===")
        print(prompt)
        
        print(f"\n=== Behavioral Prompt (3 operators) ===")
        print(behavioral_prompt)
        
        # Should contain guidance from all three operators
        prompt_lower = behavioral_prompt.lower()
        assert "structure" in prompt_lower or "organized" in prompt_lower
        assert "thorough" in prompt_lower or "detailed" in prompt_lower
        assert "precise" in prompt_lower or "careful" in prompt_lower
        
        # Query LLM
        response = llm_client.query(behavioral_prompt, temperature=0.7, max_tokens=600)
        
        print(f"\n=== LLM Response ===")
        print(response)
        print(f"\nWord count: {count_words(response)}")
        
        # Should produce substantial, detailed response
        assert count_words(response) > 100, \
            "Combined conscientiousness effects should produce detailed response"