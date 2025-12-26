"""
Unit tests for RnB personality module.

Tests cover:
- PersonalityState (scheme-level storage, aggregation)
- Taxonomy (structure, navigation)
- PersonalityStateFactory (creation from various inputs)
"""

import pytest
from datetime import datetime
from pathlib import Path

from rnb.personality import (
    PersonalityState,
    MoodDimension,
    AffectDimension,
    Trait,
    Taxonomy,
    SchemeDefinition,
    FACETS_BY_TRAIT,
    normalize_trait,
    normalize_facet,
    PersonalityStateFactory,
    ArchetypeRegistry,
    get_default_archetypes,
)
from rnb.resources import SchemeRegistry, PersonalityResolver


# ===== Fixtures =====

@pytest.fixture
def resources_dir() -> Path:
    """Path to resource data files"""
    return Path(__file__).parent.parent.parent.parent / "src" / "rnb" / "resources" / "data"


@pytest.fixture
def scheme_registry(resources_dir) -> SchemeRegistry:
    """Loaded SchemeRegistry"""
    return SchemeRegistry.from_yaml(resources_dir / "schemes.yaml")


@pytest.fixture
def taxonomy(scheme_registry) -> Taxonomy:
    """Loaded Taxonomy"""
    return Taxonomy.from_scheme_registry(scheme_registry)


@pytest.fixture
def personality_resolver(resources_dir) -> PersonalityResolver:
    """Loaded PersonalityResolver"""
    return PersonalityResolver.from_yaml(
        neopiradj_path=resources_dir / "neopiradj.yaml",
        schemes_path=resources_dir / "schemes.yaml",
        warn_unresolved=False
    )


@pytest.fixture
def factory(personality_resolver, taxonomy, resources_dir) -> PersonalityStateFactory:
    """Configured PersonalityStateFactory"""
    from rnb.personality import ArchetypeRegistry
    
    archetypes_path = resources_dir / "archetypes.yaml"
    archetypes = None
    if archetypes_path.exists():
        archetypes = ArchetypeRegistry.from_yaml(archetypes_path)
    
    return PersonalityStateFactory(personality_resolver, taxonomy, archetypes)


# ===== Taxonomy Tests =====

class TestTrait:
    """Tests for Trait enum"""
    
    def test_all_traits_present(self):
        """Test that all 5 FFM traits are defined"""
        assert len(Trait) == 5
        assert Trait.OPENNESS in Trait
        assert Trait.CONSCIENTIOUSNESS in Trait
        assert Trait.EXTRAVERSION in Trait
        assert Trait.AGREEABLENESS in Trait
        assert Trait.NEUROTICISM in Trait
    
    def test_trait_values(self):
        """Test trait string values"""
        assert Trait.OPENNESS.value == "Openness"
        assert Trait.CONSCIENTIOUSNESS.value == "Conscientiousness"


class TestNormalization:
    """Tests for name normalization functions"""
    
    def test_normalize_trait_exact(self):
        """Test exact trait name normalization"""
        assert normalize_trait("Openness") == Trait.OPENNESS
        assert normalize_trait("Conscientiousness") == Trait.CONSCIENTIOUSNESS
    
    def test_normalize_trait_case_insensitive(self):
        """Test case-insensitive trait normalization"""
        assert normalize_trait("openness") == Trait.OPENNESS
        assert normalize_trait("OPENNESS") == Trait.OPENNESS
    
    def test_normalize_trait_invalid(self):
        """Test that invalid trait names return None"""
        assert normalize_trait("Invalid") is None
        assert normalize_trait("Others") is None
    
    def test_normalize_facet_exact(self):
        """Test exact facet name normalization"""
        assert normalize_facet("Fantasy") == "Fantasy"
        assert normalize_facet("Order") == "Order"
    
    def test_normalize_facet_variants(self):
        """Test facet name variant normalization"""
        assert normalize_facet("Orderliness") == "Order"
        assert normalize_facet("Achievement-striving") == "Achievement-Striving"


class TestTaxonomy:
    """Tests for Taxonomy class"""
    
    def test_load_from_registry(self, scheme_registry):
        """Test loading taxonomy from SchemeRegistry"""
        taxonomy = Taxonomy.from_scheme_registry(scheme_registry)
        assert taxonomy is not None
        assert taxonomy.num_schemes > 0
    
    def test_scheme_count(self, taxonomy):
        """Test that taxonomy has schemes loaded"""
        # With corrected YAMLs, should have 70 schemes
        # The exact count may vary based on YAML content
        assert taxonomy.num_schemes > 0
        # If fully loaded, should have around 70 schemes
        # (5 traits * ~14 schemes per trait average)
    
    def test_get_scheme(self, taxonomy):
        """Test getting a specific scheme"""
        scheme = taxonomy.get_scheme_by_name(
            Trait.OPENNESS, "Fantasy", "IDEALISTICNESS"
        )
        assert scheme is not None
        assert scheme.name == "IDEALISTICNESS"
        assert scheme.trait == Trait.OPENNESS
        assert scheme.facet == "Fantasy"
    
    def test_get_schemes_for_facet(self, taxonomy):
        """Test getting all schemes for a facet"""
        schemes = taxonomy.get_schemes_for_facet(Trait.OPENNESS, "Fantasy")
        assert len(schemes) > 0
        assert all(s.facet == "Fantasy" for s in schemes)
    
    def test_get_schemes_for_trait(self, taxonomy):
        """Test getting all schemes for a trait"""
        schemes = taxonomy.get_schemes_for_trait(Trait.OPENNESS)
        assert len(schemes) > 0
        assert all(s.trait == Trait.OPENNESS for s in schemes)
    
    def test_scheme_in_taxonomy(self, taxonomy):
        """Test __contains__ method"""
        assert "Openness_Fantasy_IDEALISTICNESS" in taxonomy


class TestSchemeDefinition:
    """Tests for SchemeDefinition"""
    
    def test_scheme_key(self):
        """Test scheme_key property"""
        scheme = SchemeDefinition(
            name="IDEALISTICNESS",
            trait=Trait.OPENNESS,
            facet="Fantasy",
            positive_pole="IDEALISTIC",
            negative_pole="PRACTICAL"
        )
        assert scheme.scheme_key == "Openness_Fantasy_IDEALISTICNESS"
    
    def test_facet_key(self):
        """Test facet_key property"""
        scheme = SchemeDefinition(
            name="IDEALISTICNESS",
            trait=Trait.OPENNESS,
            facet="Fantasy",
            positive_pole="IDEALISTIC",
            negative_pole="PRACTICAL"
        )
        assert scheme.facet_key == "Openness_Fantasy"


# ===== PersonalityState Tests =====

class TestPersonalityState:
    """Tests for PersonalityState"""
    
    def test_creation_empty(self):
        """Test creating empty state"""
        state = PersonalityState(agent_id="test-agent")
        assert state.agent_id == "test-agent"
        assert len(state.schemes) == 0
        assert state.num_schemes_set == 0
    
    def test_set_and_get_scheme(self):
        """Test setting and getting scheme values"""
        state = PersonalityState(agent_id="test")
        state.set_scheme("Openness_Fantasy_IDEALISTICNESS", 0.8)
        
        assert state.get_scheme("Openness_Fantasy_IDEALISTICNESS") == 0.8
        assert state.get_scheme("NonExistent", default=0.5) == 0.5
    
    def test_scheme_value_validation(self):
        """Test that scheme values must be in [-1, 1]"""
        state = PersonalityState(agent_id="test")
        
        with pytest.raises(ValueError):
            state.set_scheme("Test_Scheme", 1.5)
        
        with pytest.raises(ValueError):
            state.set_scheme("Test_Scheme", -1.5)
    
    def test_get_schemes_for_facet(self):
        """Test getting all schemes for a facet"""
        state = PersonalityState(agent_id="test")
        state.set_scheme("Openness_Fantasy_SCHEME1", 0.5)
        state.set_scheme("Openness_Fantasy_SCHEME2", 0.7)
        state.set_scheme("Openness_Aesthetics_SCHEME3", 0.3)
        
        fantasy_schemes = state.get_schemes_for_facet("Openness", "Fantasy")
        assert len(fantasy_schemes) == 2
        assert "Openness_Fantasy_SCHEME1" in fantasy_schemes
    
    def test_get_schemes_for_trait(self):
        """Test getting all schemes for a trait"""
        state = PersonalityState(agent_id="test")
        state.set_scheme("Openness_Fantasy_SCHEME1", 0.5)
        state.set_scheme("Conscientiousness_Order_SCHEME2", 0.7)
        
        openness_schemes = state.get_schemes_for_trait("Openness")
        assert len(openness_schemes) == 1
        assert "Openness_Fantasy_SCHEME1" in openness_schemes
    
    def test_facet_aggregation(self):
        """Test that facet values are aggregated from schemes"""
        state = PersonalityState(agent_id="test")
        state.set_scheme("Openness_Fantasy_SCHEME1", 0.6)
        state.set_scheme("Openness_Fantasy_SCHEME2", 0.8)
        
        # Should be mean of 0.6 and 0.8 = 0.7
        facet_value = state.get_facet("Openness", "Fantasy")
        assert abs(facet_value - 0.7) < 0.01
    
    def test_trait_aggregation(self):
        """Test that trait values are aggregated from all schemes"""
        state = PersonalityState(agent_id="test")
        state.set_scheme("Openness_Fantasy_SCHEME1", 0.4)
        state.set_scheme("Openness_Aesthetics_SCHEME2", 0.8)
        
        # Should be mean of 0.4 and 0.8 = 0.6
        trait_value = state.get_trait(Trait.OPENNESS)
        assert abs(trait_value - 0.6) < 0.01
    
    def test_mood_operations(self):
        """Test mood get/set/update operations"""
        state = PersonalityState(agent_id="test")
        
        # Initial value should be 0
        assert state.get_mood(MoodDimension.HAPPINESS) == 0.0
        
        # Set mood
        state.set_mood(MoodDimension.HAPPINESS, 0.5)
        assert state.get_mood(MoodDimension.HAPPINESS) == 0.5
        
        # Update mood (additive)
        state.update_mood(MoodDimension.HAPPINESS, 0.3)
        assert abs(state.get_mood(MoodDimension.HAPPINESS) - 0.8) < 0.01
        
        # Should clamp to 1.0
        state.update_mood(MoodDimension.HAPPINESS, 0.5)
        assert state.get_mood(MoodDimension.HAPPINESS) == 1.0
    
    def test_affect_operations(self):
        """Test affect get/set/update operations"""
        state = PersonalityState(agent_id="test")
        
        state.set_affect(AffectDimension.COOPERATION, 0.7)
        assert state.get_affect(AffectDimension.COOPERATION) == 0.7
        
        state.update_affect(AffectDimension.COOPERATION, -0.2)
        assert abs(state.get_affect(AffectDimension.COOPERATION) - 0.5) < 0.01
    
    def test_serialization(self):
        """Test to_dict and from_dict"""
        state = PersonalityState(agent_id="test")
        state.set_scheme("Openness_Fantasy_IDEALISTICNESS", 0.8)
        state.set_mood(MoodDimension.ENERGY, 0.5)
        state.source_adjectives = ["romantic", "creative"]
        
        # Serialize
        data = state.to_dict()
        assert data["agent_id"] == "test"
        assert data["schemes"]["Openness_Fantasy_IDEALISTICNESS"] == 0.8
        assert data["moods"]["energy"] == 0.5
        assert data["source_adjectives"] == ["romantic", "creative"]
        
        # Deserialize
        restored = PersonalityState.from_dict(data)
        assert restored.agent_id == "test"
        assert restored.get_scheme("Openness_Fantasy_IDEALISTICNESS") == 0.8
        assert restored.get_mood(MoodDimension.ENERGY) == 0.5
        assert restored.source_adjectives == ["romantic", "creative"]
    
    def test_clone(self):
        """Test cloning state"""
        state = PersonalityState(agent_id="original")
        state.set_scheme("Test_Scheme", 0.5)
        state.set_mood(MoodDimension.HAPPINESS, 0.7)
        
        # Clone with new ID
        cloned = state.clone("cloned-agent")
        assert cloned.agent_id == "cloned-agent"
        assert cloned.get_scheme("Test_Scheme") == 0.5
        
        # Modifications to clone don't affect original
        cloned.set_scheme("Test_Scheme", 0.9)
        assert state.get_scheme("Test_Scheme") == 0.5


# ===== PersonalityStateFactory Tests =====

class TestPersonalityStateFactory:
    """Tests for PersonalityStateFactory"""
    
    def test_from_adjectives(self, factory):
        """Test creating state from adjectives"""
        state = factory.from_adjectives("test-agent", ["romantic", "organized"])
        
        assert state.agent_id == "test-agent"
        assert state.num_schemes_set > 0
        assert state.source_adjectives == ["romantic", "organized"]
    
    def test_from_adjectives_unknown(self, factory):
        """Test that unknown adjectives are handled gracefully"""
        state = factory.from_adjectives(
            "test-agent", 
            ["romantic", "xyzunknown"]
        )
        
        # Should still create state from known adjective
        assert state.num_schemes_set > 0
    
    def test_from_traits(self, factory):
        """Test creating state from trait values"""
        state = factory.from_traits(
            "test-agent",
            {Trait.OPENNESS: 0.8, Trait.CONSCIENTIOUSNESS: 0.6}
        )
        
        assert state.agent_id == "test-agent"
        
        # All schemes under Openness should be 0.8
        openness_schemes = state.get_schemes_for_trait(Trait.OPENNESS)
        for scheme_key, value in openness_schemes.items():
            assert value == 0.8
        
        # Aggregated trait value should match
        assert abs(state.get_trait(Trait.OPENNESS) - 0.8) < 0.01
    
    def test_from_facets(self, factory):
        """Test creating state from facet values"""
        state = factory.from_facets(
            "test-agent",
            {"Openness_Fantasy": 0.9, "Conscientiousness_Order": 0.7}
        )
        
        # All schemes under Fantasy should be 0.9
        fantasy_schemes = state.get_schemes_for_facet("Openness", "Fantasy")
        for scheme_key, value in fantasy_schemes.items():
            assert value == 0.9
    
    def test_from_schemes(self, factory):
        """Test creating state from scheme values"""
        state = factory.from_schemes(
            "test-agent",
            {
                "Openness_Fantasy_IDEALISTICNESS": 0.8,
                "Openness_Fantasy_SPIRITUALISTNESS": 0.3,
            }
        )
        
        assert state.get_scheme("Openness_Fantasy_IDEALISTICNESS") == 0.8
        assert state.get_scheme("Openness_Fantasy_SPIRITUALISTNESS") == 0.3
    
    def test_from_mixed(self, factory):
        """Test creating state from mixed levels"""
        state = factory.from_mixed(
            "test-agent",
            traits={Trait.OPENNESS: 0.5},  # Default for all O schemes
            facets={"Openness_Fantasy": 0.8},  # Override Fantasy facet
            schemes={"Openness_Fantasy_IDEALISTICNESS": 0.95},  # Override specific
        )
        
        # Specific scheme override should take precedence
        assert state.get_scheme("Openness_Fantasy_IDEALISTICNESS") == 0.95
        
        # Other Fantasy schemes should be 0.8 (facet level)
        spiritualist = state.get_scheme("Openness_Fantasy_SPIRITUALISTNESS", default=None)
        if spiritualist is not None:  # May not exist depending on taxonomy
            assert spiritualist == 0.8
    
    def test_from_archetype(self, factory):
        """Test creating state from predefined archetype"""
        state = factory.create_from_archetype("test-agent", "helpful_assistant")
        
        assert state.agent_id == "test-agent"
        assert state.num_schemes_set > 0
        
        # Archetype should set schemes based on its trait values
        # The actual values depend on which schemes are loaded
    
    def test_from_archetype_unknown(self, factory):
        """Test that unknown archetypes raise error"""
        with pytest.raises(ValueError):
            factory.create_from_archetype("test-agent", "nonexistent_archetype")
    
    def test_create_neutral(self, factory):
        """Test creating neutral state"""
        state = factory.create_neutral("test-agent")
        
        assert state.agent_id == "test-agent"
        assert state.num_schemes_set == 0
        assert state.get_trait(Trait.OPENNESS) == 0.0


class TestArchetypes:
    """Tests for predefined archetypes"""
    
    def test_archetypes_loaded(self, factory):
        """Test that archetypes are loaded from YAML"""
        assert factory.archetypes is not None
        assert len(factory.archetypes) > 0
    
    def test_archetype_names(self, factory):
        """Test that expected archetypes exist"""
        names = factory.list_archetypes()
        assert "helpful_assistant" in names
        assert "creative_thinker" in names
    
    def test_archetype_structure(self, factory):
        """Test archetype structure"""
        archetype = factory.archetypes.get("helpful_assistant")
        assert archetype is not None
        assert archetype.description != ""
        assert len(archetype.traits) == 5  # All 5 traits
        
        # All trait values should be in valid range
        for trait_name, value in archetype.traits.items():
            assert -1.0 <= value <= 1.0
    
    def test_get_default_archetypes(self):
        """Test backwards-compatible get_default_archetypes function"""
        archetypes = get_default_archetypes()
        assert len(archetypes) > 0
        assert "helpful_assistant" in archetypes
        assert "traits" in archetypes["helpful_assistant"]


# ===== Integration Tests =====

class TestIntegration:
    """Integration tests for personality module"""
    
    def test_adjective_to_trait_pipeline(self, factory):
        """Test full pipeline from adjectives to trait values"""
        # Create from adjectives
        state = factory.from_adjectives("test", ["romantic", "lazy"])
        
        # Should have schemes set
        assert state.num_schemes_set > 0
        
        # Can get aggregated values
        openness = state.get_trait(Trait.OPENNESS)
        conscientiousness = state.get_trait(Trait.CONSCIENTIOUSNESS)
        
        # "romantic" → high Openness/Fantasy
        # "lazy" → low Conscientiousness
        assert openness != 0.0 or conscientiousness != 0.0
    
    def test_trait_propagation(self, factory):
        """Test that trait values propagate correctly to schemes"""
        state = factory.from_traits("test", {Trait.OPENNESS: 0.7})
        
        # All Openness schemes should be 0.7
        for scheme_key, value in state.get_schemes_for_trait(Trait.OPENNESS).items():
            assert value == 0.7
        
        # Other traits should be 0.0 (not set)
        conscientiousness_schemes = state.get_schemes_for_trait(Trait.CONSCIENTIOUSNESS)
        assert len(conscientiousness_schemes) == 0
    
    def test_state_persistence_roundtrip(self, factory):
        """Test that state survives serialization roundtrip"""
        original = factory.from_adjectives("test", ["romantic", "organized"])
        original.set_mood(MoodDimension.HAPPINESS, 0.6)
        original.set_affect(AffectDimension.TRUST, 0.4)
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = PersonalityState.from_dict(data)
        
        # Verify all data preserved
        assert restored.agent_id == original.agent_id
        assert restored.schemes == original.schemes
        assert restored.moods == original.moods
        assert restored.affects == original.affects
        assert restored.source_adjectives == original.source_adjectives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])