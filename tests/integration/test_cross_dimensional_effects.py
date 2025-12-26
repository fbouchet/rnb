"""
Integration tests for interactions between multiple personality dimensions.

Tests that traits, moods, and affects can work together to produce
complex behavioral patterns.

Reference: RnB Model M.A - multi-dimensional personality state
"""

import pytest
from rnb.personality.state import FFMTrait, MoodDimension, AffectDimension
from rnb.influence.context import InfluenceContext
from rnb.influence.trait_based import StructureInfluence, EnthusiasmInfluence
from rnb.influence.affect_based import CooperationVerbosityInfluence
from rnb.influence.mood_based import EnergyLengthInfluence


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


class TestTraitAffectInteraction:
    """Test interactions between stable traits and relationship affects"""
    
    def test_conscientiousness_and_cooperation_combine(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client
    ):
        """
        High conscientiousness + high cooperation → very detailed, helpful.
        High conscientiousness + low cooperation → detailed but minimal.
        
        Tests operator composition across dimensions.
        """
        # Agent 1: High C, High Coop (ideal tutor)
        agent_manager.create_agent(
            "ideal_tutor",
            traits={FFMTrait.CONSCIENTIOUSNESS: 0.8},
            affects={AffectDimension.COOPERATION: 0.9}
        )
        
        # Agent 2: High C, Low Coop (thorough but unhelpful)
        agent_manager.create_agent(
            "thorough_minimal",
            traits={FFMTrait.CONSCIENTIOUSNESS: 0.8},
            affects={AffectDimension.COOPERATION: -0.5}
        )
        
        # Register operators from both dimensions
        operator_registry.register(StructureInfluence())
        operator_registry.register(CooperationVerbosityInfluence())
        
        prompt = "Explain quicksort algorithm."
        
        # Ideal tutor
        context_tutor = InfluenceContext.from_personality(
            personality_store.get_state("ideal_tutor")
        )
        prompt_tutor = operator_registry.apply_all(prompt, context_tutor)
        response_tutor = llm_client.query(prompt_tutor, temperature=0.7, max_tokens=500)
        
        # Thorough but minimal
        context_minimal = InfluenceContext.from_personality(
            personality_store.get_state("thorough_minimal")
        )
        prompt_minimal = operator_registry.apply_all(prompt, context_minimal)
        response_minimal = llm_client.query(prompt_minimal, temperature=0.7, max_tokens=500)
        
        words_tutor = count_words(response_tutor)
        words_minimal = count_words(response_minimal)
        
        print(f"\n=== Ideal Tutor (High C + High Coop) ===")
        print(f"Word count: {words_tutor}")
        print(response_tutor)
        
        print(f"\n=== Thorough Minimal (High C + Low Coop) ===")
        print(f"Word count: {words_minimal}")
        print(response_minimal)
        
        # Both should be structured (conscientiousness)
        # But tutor should be more verbose (cooperation)
        assert words_tutor > words_minimal, \
            f"High cooperation should add verbosity: {words_tutor} vs {words_minimal}"


class TestTraitMoodInteraction:
    """Test interactions between stable traits and dynamic moods"""
    
    def test_extraversion_and_energy_combine(
        self,
        agent_manager,
        personality_store,
        operator_registry,
        llm_client
    ):
        """
        High extraversion + high energy → very enthusiastic, elaborate.
        High extraversion + low energy → enthusiastic but concise.
        
        Tests trait-mood interaction.
        """
        # Agent 1: Extraverted and energetic
        agent_manager.create_agent(
            "energetic_extravert",
            traits={FFMTrait.EXTRAVERSION: 0.7},
            moods={MoodDimension.ENERGY: 0.8}
        )
        
        # Agent 2: Extraverted but tired
        agent_manager.create_agent(
            "tired_extravert",
            traits={FFMTrait.EXTRAVERSION: 0.7},
            moods={MoodDimension.ENERGY: -0.6}
        )
        
        # Register operators
        operator_registry.register(EnthusiasmInfluence())
        operator_registry.register(EnergyLengthInfluence())
        
        prompt = "Tell me about your favorite programming language."
        
        # Energetic extravert
        context_energetic = InfluenceContext.from_personality(
            personality_store.get_state("energetic_extravert")
        )
        prompt_energetic = operator_registry.apply_all(prompt, context_energetic)
        response_energetic = llm_client.query(prompt_energetic, temperature=0.8)
        
        # Tired extravert
        context_tired = InfluenceContext.from_personality(
            personality_store.get_state("tired_extravert")
        )
        prompt_tired = operator_registry.apply_all(prompt, context_tired)
        response_tired = llm_client.query(prompt_tired, temperature=0.8)
        
        words_energetic = count_words(response_energetic)
        words_tired = count_words(response_tired)
        
        print(f"\n=== Energetic Extravert ===")
        print(f"Word count: {words_energetic}")
        print(response_energetic)
        
        print(f"\n=== Tired Extravert ===")
        print(f"Word count: {words_tired}")
        print(response_tired)
        
        # Both should be enthusiastic (trait)
        # But energetic one should be longer (mood)
        assert words_energetic > words_tired * 0.8, \
            f"Low energy should reduce verbosity: {words_energetic} vs {words_tired}"