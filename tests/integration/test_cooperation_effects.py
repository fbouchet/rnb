"""
Integration tests for cooperation affect effects on LLM behavior.

Validates that high vs low cooperation produces measurably different responses
in terms of helpfulness and verbosity.

Reference: Bouchet & Sansonnet (2013), RnB affect dimensions
"""

import pytest
from rnb.personality.state import AffectDimension
from rnb.influence.context import InfluenceContext
from rnb.influence.affect_based import (
    CooperationVerbosityInfluence,
    CooperationHelpfulnessInfluence
)


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


class TestCooperationVerbosity:
    """Test that cooperation affects response detail level"""
    
    def test_cooperation_affects_verbosity(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client
    ):
        """
        High cooperation → more detailed, helpful responses.
        Low cooperation → brief, minimal responses.
        
        RnB behavioral scheme: Verbosity control via cooperation
        """
        # Create agents with different cooperation levels
        agent_manager.create_agent(
            "high_coop",
            affects={AffectDimension.COOPERATION: 0.9}
        )
        
        agent_manager.create_agent(
            "low_coop",
            affects={AffectDimension.COOPERATION: -0.5}
        )
        
        # Register cooperation operator
        operator_registry.register(CooperationVerbosityInfluence())
        
        # Query both agents
        prompt = "What is a binary search tree?"
        
        # High cooperation
        context_high = InfluenceContext.from_personality(
            personality_store.get_state("high_coop")
        )
        prompt_high = operator_registry.apply_all(prompt, context_high)
        response_high = llm_client.query(prompt_high, temperature=0.7, max_tokens=400)
        
        # Low cooperation
        context_low = InfluenceContext.from_personality(
            personality_store.get_state("low_coop")
        )
        prompt_low = operator_registry.apply_all(prompt, context_low)
        response_low = llm_client.query(prompt_low, temperature=0.7, max_tokens=400)
        
        words_high = count_words(response_high)
        words_low = count_words(response_low)
        
        print(f"\n=== High Cooperation (0.9) ===")
        print(f"Prompt modifications: {prompt_high[len(prompt):]}")
        print(f"Word count: {words_high}")
        print(f"Response: {response_high}")
        
        print(f"\n=== Low Cooperation (-0.5) ===")
        print(f"Prompt modifications: {prompt_low[len(prompt):]}")
        print(f"Word count: {words_low}")
        print(f"Response: {response_low}")
        
        # Validation: high cooperation should be more verbose
        assert words_high > words_low * 1.2, \
            f"High cooperation should be >20% more verbose: {words_high} vs {words_low}"


class TestCooperationHelpfulness:
    """Test that cooperation affects proactive helpfulness"""
    
    def test_cooperation_affects_helpfulness(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client
    ):
        """
        High cooperation → proactive, offers additional information.
        Low cooperation → minimal, addresses only what was asked.
        
        RnB behavioral scheme: Helpfulness level
        """
        # Create agents
        agent_manager.create_agent(
            "helpful_agent",
            affects={AffectDimension.COOPERATION: 0.8}
        )
        
        agent_manager.create_agent(
            "minimal_agent",
            affects={AffectDimension.COOPERATION: -0.4}
        )
        
        # Register helpfulness operator
        operator_registry.register(CooperationHelpfulnessInfluence())
        
        # Query requiring potential additional context
        prompt = "How do I sort a list in Python?"
        
        # Helpful agent
        context_helpful = InfluenceContext.from_personality(
            personality_store.get_state("helpful_agent")
        )
        prompt_helpful = operator_registry.apply_all(prompt, context_helpful)
        response_helpful = llm_client.query(prompt_helpful, temperature=0.7)
        
        # Minimal agent
        context_minimal = InfluenceContext.from_personality(
            personality_store.get_state("minimal_agent")
        )
        prompt_minimal = operator_registry.apply_all(prompt, context_minimal)
        response_minimal = llm_client.query(prompt_minimal, temperature=0.7)
        
        print(f"\n=== Helpful Agent (High Cooperation) ===")
        print(response_helpful)
        
        print(f"\n=== Minimal Agent (Low Cooperation) ===")
        print(response_minimal)
        
        # Helpful agent more likely to mention related topics
        # (e.g., sort() vs sorted(), key parameter, reverse, etc.)
        words_helpful = count_words(response_helpful)
        words_minimal = count_words(response_minimal)
        
        assert words_helpful > words_minimal, \
            f"Helpful agent should provide more information: {words_helpful} vs {words_minimal}"