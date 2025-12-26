"""
Unit tests for Gloss-Based Influence System

Tests cover:
1. GlossInfluenceEngine - gloss retrieval and context generation
2. GlossBasedInfluence operator - activation and prompt modification
3. TraitGlossInfluence - trait-specific operators
4. Integration with PersonalityState from adjectives

These tests verify that the gloss-based system correctly:
- Retrieves glosses based on personality state values
- Respects activation thresholds
- Generates appropriate behavioral context in various styles
- Works end-to-end from adjectives to modified prompts
"""

import pytest
from pathlib import Path

from rnb.influence import (
    GlossInfluenceEngine,
    ContextStyle,
    ActiveGloss,
    GlossBasedInfluence,
    TraitGlossInfluence,
    InfluenceContext,
)
from rnb.personality import (
    PersonalityState,
    Trait,
)
from rnb.personality.factory import PersonalityStateFactory
from rnb.resources import SchemeRegistry


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def resources_dir() -> Path:
    """Path to resources directory"""
    return Path(__file__).parent.parent.parent / "src" / "rnb" / "resources" / "data"


@pytest.fixture
def scheme_registry(resources_dir) -> SchemeRegistry:
    """Loaded SchemeRegistry"""
    return SchemeRegistry.from_yaml(resources_dir / "schemes.yaml")


@pytest.fixture
def engine(scheme_registry) -> GlossInfluenceEngine:
    """Configured GlossInfluenceEngine"""
    return GlossInfluenceEngine(scheme_registry)


@pytest.fixture
def factory(resources_dir) -> PersonalityStateFactory:
    """Configured PersonalityStateFactory"""
    return PersonalityStateFactory.from_default_resources()


@pytest.fixture
def high_openness_state() -> PersonalityState:
    """PersonalityState with high Openness schemes"""
    state = PersonalityState(agent_id="test-high-openness")
    # Set some Openness schemes to high positive values
    state.set_scheme("Openness_Fantasy_IDEALISTICNESS", 0.8)
    state.set_scheme("Openness_Aesthetics_AESTHETIC-SENSITIVENESS", 0.7)
    return state


@pytest.fixture
def mixed_state() -> PersonalityState:
    """PersonalityState with mixed scheme values"""
    state = PersonalityState(agent_id="test-mixed")
    # High positive Openness
    state.set_scheme("Openness_Fantasy_IDEALISTICNESS", 0.7)
    # High negative Conscientiousness (disorganized)
    state.set_scheme("Conscientiousness_Order_METHODICALNESS", -0.6)
    # Low (inactive) Extraversion
    state.set_scheme("Extraversion_Warmth_FRIENDLINESS", 0.2)
    return state


@pytest.fixture
def neutral_state() -> PersonalityState:
    """PersonalityState with all near-zero values"""
    state = PersonalityState(agent_id="test-neutral")
    state.set_scheme("Openness_Fantasy_IDEALISTICNESS", 0.1)
    state.set_scheme("Conscientiousness_Order_METHODICALNESS", -0.1)
    return state


# ============================================================
# GlossInfluenceEngine Tests
# ============================================================

class TestGlossInfluenceEngineInit:
    """Tests for engine initialization"""
    
    def test_init_with_registry(self, scheme_registry):
        """Test initialization with scheme registry"""
        engine = GlossInfluenceEngine(scheme_registry)
        assert engine.scheme_registry is scheme_registry
        assert engine.default_threshold == 0.3
    
    def test_init_custom_threshold(self, scheme_registry):
        """Test initialization with custom threshold"""
        engine = GlossInfluenceEngine(scheme_registry, default_threshold=0.5)
        assert engine.default_threshold == 0.5
    
    def test_from_default_resources(self):
        """Test creation from default resources"""
        engine = GlossInfluenceEngine.from_default_resources()
        assert engine.scheme_registry is not None
        stats = engine.statistics()
        assert stats["total_schemes"] > 0


class TestGetActiveSchemes:
    """Tests for get_active_schemes method"""
    
    def test_active_schemes_above_threshold(self, engine, high_openness_state):
        """Test that schemes above threshold are returned"""
        active = engine.get_active_schemes(high_openness_state, threshold=0.3)
        
        assert len(active) > 0
        # All returned schemes should exceed threshold
        for scheme_key, value, pole in active:
            assert abs(value) > 0.3
    
    def test_no_active_schemes_below_threshold(self, engine, neutral_state):
        """Test that near-zero schemes are not returned"""
        active = engine.get_active_schemes(neutral_state, threshold=0.3)
        assert len(active) == 0
    
    def test_pole_selection_positive(self, engine, high_openness_state):
        """Test that positive values yield 'pos' pole"""
        active = engine.get_active_schemes(high_openness_state)
        
        for scheme_key, value, pole in active:
            if value > 0:
                assert pole == "pos"
    
    def test_pole_selection_negative(self, engine, mixed_state):
        """Test that negative values yield 'neg' pole"""
        active = engine.get_active_schemes(mixed_state)
        
        for scheme_key, value, pole in active:
            if value < 0:
                assert pole == "neg"
    
    def test_sorted_by_intensity(self, engine, mixed_state):
        """Test that results are sorted by intensity (descending)"""
        active = engine.get_active_schemes(mixed_state)
        
        if len(active) > 1:
            intensities = [abs(v) for _, v, _ in active]
            assert intensities == sorted(intensities, reverse=True)


class TestGetActiveGlosses:
    """Tests for get_active_glosses method"""
    
    def test_returns_active_gloss_objects(self, engine, high_openness_state):
        """Test that ActiveGloss objects are returned"""
        glosses = engine.get_active_glosses(high_openness_state)
        
        assert len(glosses) > 0
        for gloss in glosses:
            assert isinstance(gloss, ActiveGloss)
            assert gloss.text  # Has content
            assert gloss.scheme_name  # Has scheme info
    
    def test_gloss_has_correct_metadata(self, engine, high_openness_state):
        """Test that ActiveGloss has correct metadata"""
        glosses = engine.get_active_glosses(high_openness_state)
        
        if glosses:
            gloss = glosses[0]
            assert gloss.trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
            assert gloss.facet  # Has facet
            assert gloss.pole in ["pos", "neg"]
            assert 0 < gloss.intensity <= 1
    
    def test_filter_by_trait(self, engine, mixed_state):
        """Test filtering glosses by trait"""
        glosses = engine.get_active_glosses(
            mixed_state,
            filter_trait=Trait.OPENNESS
        )
        
        for gloss in glosses:
            assert gloss.trait == "Openness"
    
    def test_max_per_scheme(self, engine, high_openness_state):
        """Test limiting glosses per scheme"""
        glosses = engine.get_active_glosses(
            high_openness_state,
            max_per_scheme=1
        )
        
        # Count glosses per scheme
        scheme_counts = {}
        for gloss in glosses:
            scheme_counts[gloss.scheme_key] = scheme_counts.get(gloss.scheme_key, 0) + 1
        
        for count in scheme_counts.values():
            assert count <= 1
    
    def test_empty_for_neutral_state(self, engine, neutral_state):
        """Test that neutral state returns no glosses"""
        glosses = engine.get_active_glosses(neutral_state, threshold=0.3)
        assert len(glosses) == 0


class TestGenerateBehavioralContext:
    """Tests for generate_behavioral_context method"""
    
    def test_descriptive_style(self, engine, high_openness_state):
        """Test descriptive style output"""
        context = engine.generate_behavioral_context(
            high_openness_state,
            style=ContextStyle.DESCRIPTIVE
        )
        
        assert "Personality context:" in context
        assert "You" in context  # Descriptive uses "You tend to be..."
        assert "tendencies" in context or "naturally" in context
    
    def test_prescriptive_style(self, engine, high_openness_state):
        """Test prescriptive style output"""
        context = engine.generate_behavioral_context(
            high_openness_state,
            style=ContextStyle.PRESCRIPTIVE
        )
        
        assert "Behavioral instructions:" in context
        assert "Be" in context or "Show" in context
    
    def test_concise_style(self, engine, high_openness_state):
        """Test concise style output"""
        context = engine.generate_behavioral_context(
            high_openness_state,
            style=ContextStyle.CONCISE
        )
        
        assert context.startswith("[Personality:")
        assert context.endswith("]")
    
    def test_narrative_style(self, engine, high_openness_state):
        """Test narrative style output"""
        context = engine.generate_behavioral_context(
            high_openness_state,
            style=ContextStyle.NARRATIVE
        )
        
        assert "someone who" in context
    
    def test_empty_for_neutral_state(self, engine, neutral_state):
        """Test that neutral state returns empty context"""
        context = engine.generate_behavioral_context(neutral_state)
        assert context == ""
    
    def test_max_total_glosses(self, engine, high_openness_state):
        """Test limiting total glosses"""
        context = engine.generate_behavioral_context(
            high_openness_state,
            max_total_glosses=2
        )
        
        # Should have limited content
        lines = [l for l in context.split("\n") if l.startswith("- ")]
        assert len(lines) <= 2
    
    def test_string_style_accepted(self, engine, high_openness_state):
        """Test that string style names work"""
        context = engine.generate_behavioral_context(
            high_openness_state,
            style="concise"
        )
        assert "[Personality:" in context


class TestGetGlossesForScheme:
    """Tests for direct gloss retrieval"""
    
    def test_get_glosses_for_known_scheme(self, engine):
        """Test retrieving glosses for a known scheme"""
        glosses = engine.get_glosses_for_scheme(
            trait="Openness",
            facet="Fantasy",
            scheme_name="IDEALISTICNESS",
            pole="pos"
        )
        
        # Should have at least one gloss
        assert len(glosses) > 0
        # Each item is (gloss_id, gloss_text) tuple
        assert isinstance(glosses[0], tuple)
        assert isinstance(glosses[0][0], str)  # id
        assert isinstance(glosses[0][1], str)  # text
    
    def test_get_glosses_unknown_scheme(self, engine):
        """Test retrieving glosses for unknown scheme"""
        glosses = engine.get_glosses_for_scheme(
            trait="Openness",
            facet="Fantasy",
            scheme_name="NONEXISTENT",
            pole="pos"
        )
        
        assert len(glosses) == 0


# ============================================================
# GlossBasedInfluence Operator Tests
# ============================================================

class TestGlossBasedInfluenceInit:
    """Tests for operator initialization"""
    
    def test_from_default_resources(self):
        """Test creation from default resources"""
        op = GlossBasedInfluence.from_default_resources()
        assert op is not None
        assert op.name == "gloss_influence"
    
    def test_custom_parameters(self, engine):
        """Test initialization with custom parameters"""
        op = GlossBasedInfluence(
            engine=engine,
            threshold=0.5,
            style=ContextStyle.PRESCRIPTIVE,
            max_glosses=5,
            priority=60
        )
        
        assert op.threshold == 0.5
        assert op.style == ContextStyle.PRESCRIPTIVE
        assert op.max_glosses == 5
        assert op.get_activation_priority() == 60
    
    def test_trait_filter_in_name(self, engine):
        """Test that trait filter appears in name"""
        op = GlossBasedInfluence(
            engine=engine,
            filter_trait=Trait.OPENNESS
        )
        
        assert "openness" in op.name.lower()


class TestGlossBasedInfluenceApplies:
    """Tests for operator activation"""
    
    def test_applies_when_schemes_active(self, engine, high_openness_state):
        """Test operator applies when schemes exceed threshold"""
        op = GlossBasedInfluence(engine=engine, threshold=0.3)
        context = InfluenceContext.from_personality(high_openness_state)
        
        assert op.applies(context) is True
    
    def test_not_applies_when_neutral(self, engine, neutral_state):
        """Test operator doesn't apply for neutral state"""
        op = GlossBasedInfluence(engine=engine, threshold=0.3)
        context = InfluenceContext.from_personality(neutral_state)
        
        assert op.applies(context) is False
    
    def test_applies_respects_trait_filter(self, engine, mixed_state):
        """Test trait filter affects activation"""
        # Openness is high, should apply
        op_openness = GlossBasedInfluence(
            engine=engine,
            filter_trait=Trait.OPENNESS,
            threshold=0.3
        )
        context = InfluenceContext.from_personality(mixed_state)
        assert op_openness.applies(context) is True
        
        # Extraversion is low (0.2), should not apply with 0.3 threshold
        op_extraversion = GlossBasedInfluence(
            engine=engine,
            filter_trait=Trait.EXTRAVERSION,
            threshold=0.3
        )
        assert op_extraversion.applies(context) is False


class TestGlossBasedInfluenceApply:
    """Tests for prompt modification"""
    
    def test_apply_appends_context(self, engine, high_openness_state):
        """Test that apply appends behavioral context"""
        op = GlossBasedInfluence(engine=engine)
        context = InfluenceContext.from_personality(high_openness_state)
        
        base_prompt = "Explain recursion."
        modified = op.apply(base_prompt, context)
        
        assert base_prompt in modified
        assert len(modified) > len(base_prompt)
        assert "Personality context:" in modified
    
    def test_apply_returns_unchanged_for_neutral(self, engine, neutral_state):
        """Test that neutral state doesn't modify prompt"""
        op = GlossBasedInfluence(engine=engine)
        context = InfluenceContext.from_personality(neutral_state)
        
        base_prompt = "Explain recursion."
        modified = op.apply(base_prompt, context)
        
        # Should be unchanged (no behavioral context)
        assert modified == base_prompt
    
    def test_apply_respects_style(self, engine, high_openness_state):
        """Test that style parameter is respected"""
        op = GlossBasedInfluence(engine=engine, style=ContextStyle.CONCISE)
        context = InfluenceContext.from_personality(high_openness_state)
        
        modified = op.apply("Test prompt", context)
        
        assert "[Personality:" in modified


class TestGlossBasedInfluencePreview:
    """Tests for preview functionality"""
    
    def test_preview_returns_gloss_texts(self, engine, high_openness_state):
        """Test preview shows gloss texts"""
        op = GlossBasedInfluence(engine=engine)
        
        previews = op.get_active_glosses_preview(high_openness_state)
        
        assert isinstance(previews, list)
        assert all(isinstance(p, str) for p in previews)


# ============================================================
# TraitGlossInfluence Tests
# ============================================================

class TestTraitGlossInfluence:
    """Tests for trait-specific operator"""
    
    def test_openness_operator(self, high_openness_state):
        """Test Openness-specific operator"""
        op = TraitGlossInfluence(trait=Trait.OPENNESS)
        context = InfluenceContext.from_personality(high_openness_state)
        
        assert op.applies(context) is True
        assert "openness" in op.name.lower()
    
    def test_conscientiousness_operator(self, mixed_state):
        """Test Conscientiousness-specific operator"""
        op = TraitGlossInfluence(trait=Trait.CONSCIENTIOUSNESS)
        context = InfluenceContext.from_personality(mixed_state)
        
        # mixed_state has Conscientiousness at -0.6
        assert op.applies(context) is True


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_adjectives_to_behavioral_context(self, factory):
        """Test full pipeline: adjectives → state → glosses → context"""
        # Create state from adjectives
        state = factory.from_adjectives(
            "integration-test",
            ["romantic", "organized", "friendly"]
        )
        
        # Generate behavioral context
        engine = GlossInfluenceEngine.from_default_resources()
        context = engine.generate_behavioral_context(state, style="descriptive")
        
        # Should have generated some context
        assert len(context) > 0
    
    def test_operator_full_pipeline(self, factory):
        """Test operator with adjective-derived state"""
        state = factory.from_adjectives(
            "integration-test-2",
            ["creative", "imaginative"]
        )
        
        op = GlossBasedInfluence.from_default_resources()
        context = InfluenceContext.from_personality(state)
        
        modified = op.apply("Write a poem about spring.", context)
        
        # Should have added personality context
        assert "Write a poem about spring." in modified
        assert len(modified) > len("Write a poem about spring.")
    
    def test_archetype_to_behavioral_context(self, factory):
        """Test with archetype-derived state"""
        state = factory.create_from_archetype("archetype-test", "creative_thinker")
        
        engine = GlossInfluenceEngine.from_default_resources()
        context = engine.generate_behavioral_context(state)
        
        # creative_thinker has high Openness
        assert len(context) > 0