"""
Unit tests for RnB resources module.

Tests cover:
- Model data structures
- AdjectiveResolver functionality
- SchemeRegistry functionality
- PersonalityResolver integration
"""

import pytest
from pathlib import Path

from rnb.resources import (
    TraitName,
    TaxonomyPosition,
    GlossEntry,
    AdjectiveEntry,
    AdjectiveResolution,
    SchemeInfo,
    PoleInfo,
    AdjectiveResolver,
    SchemeRegistry,
    PersonalityResolver,
    PersonalitySpecification,
    ResolvedAdjective,
)


# ===== Fixtures =====

@pytest.fixture
def resources_dir() -> Path:
    """Path to resource data files"""
    return Path(__file__).parent.parent.parent.parent / "src" / "rnb" / "resources" / "data"


@pytest.fixture
def neopiradj_path(resources_dir) -> Path:
    """Path to neopiradj.yaml"""
    return resources_dir / "neopiradj.yaml"


@pytest.fixture
def schemes_path(resources_dir) -> Path:
    """Path to schemes.yaml"""
    return resources_dir / "schemes.yaml"


@pytest.fixture
def adjective_resolver(neopiradj_path) -> AdjectiveResolver:
    """Loaded AdjectiveResolver"""
    return AdjectiveResolver.from_yaml(neopiradj_path, warn_unresolved=False)


@pytest.fixture
def scheme_registry(schemes_path) -> SchemeRegistry:
    """Loaded SchemeRegistry"""
    return SchemeRegistry.from_yaml(schemes_path)


@pytest.fixture
def personality_resolver(neopiradj_path, schemes_path) -> PersonalityResolver:
    """Loaded PersonalityResolver"""
    return PersonalityResolver.from_yaml(
        neopiradj_path, 
        schemes_path,
        warn_unresolved=False
    )


# ===== Model Tests =====

class TestTaxonomyPosition:
    """Tests for TaxonomyPosition data class"""
    
    def test_creation(self):
        """Test basic position creation"""
        pos = TaxonomyPosition(
            trait="Openness",
            facet="Fantasy",
            scheme="IDEALISTICNESS",
            pole="pos"
        )
        assert pos.trait == "Openness"
        assert pos.facet == "Fantasy"
        assert pos.scheme == "IDEALISTICNESS"
        assert pos.pole == "pos"
    
    def test_facet_key(self):
        """Test facet_key property"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        assert pos.facet_key == "Openness_Fantasy"
    
    def test_scheme_key(self):
        """Test scheme_key property"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        assert pos.scheme_key == "Openness_Fantasy_IDEALISTICNESS_pos"
    
    def test_valence_positive(self):
        """Test valence property for positive pole"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        assert pos.valence == "+"
    
    def test_valence_negative(self):
        """Test valence property for negative pole"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "neg")
        assert pos.valence == "-"
    
    def test_str_representation(self):
        """Test string representation"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        assert str(pos) == "Openness/Fantasy/IDEALISTICNESS/+"
    
    def test_immutability(self):
        """Test that position is immutable (frozen)"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        with pytest.raises(AttributeError):
            pos.trait = "Conscientiousness"
    
    def test_hashable(self):
        """Test that position can be used in sets and as dict keys"""
        pos1 = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        pos2 = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        pos3 = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "neg")
        
        # Can use in set
        positions = {pos1, pos2, pos3}
        assert len(positions) == 2  # pos1 and pos2 are equal
        
        # Can use as dict key
        mapping = {pos1: "test"}
        assert mapping[pos2] == "test"  # pos2 equals pos1


class TestGlossEntry:
    """Tests for GlossEntry data class"""
    
    def test_wordnet_source(self):
        """Test source detection for WordNet glosses"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        gloss = GlossEntry(id="218", text="test gloss", position=pos)
        assert gloss.source == "wordnet"
    
    def test_goldberg_source(self):
        """Test source detection for Goldberg questionnaire items"""
        pos = TaxonomyPosition("Openness", "Fantasy", "IDEALISTICNESS", "pos")
        gloss = GlossEntry(id="Q1", text="test item", position=pos)
        assert gloss.source == "goldberg"


class TestAdjectiveResolution:
    """Tests for AdjectiveResolution data class"""
    
    def test_found_with_mappings(self):
        """Test found property with mappings"""
        pos = TaxonomyPosition("Openness", "Fantasy", "", "pos")
        entry = AdjectiveEntry(
            word="romantic", synset="WildEyed", weight=5,
            gloss_id="218", gloss_text="test", position=pos
        )
        resolution = AdjectiveResolution(
            input_word="romantic",
            normalized_word="romantic",
            mappings=[entry]
        )
        assert resolution.found is True
    
    def test_found_without_mappings(self):
        """Test found property without mappings"""
        resolution = AdjectiveResolution(
            input_word="unknown",
            normalized_word="unknown",
            mappings=[]
        )
        assert resolution.found is False
    
    def test_ambiguous_single_facet(self):
        """Test ambiguous property with single facet"""
        pos = TaxonomyPosition("Openness", "Fantasy", "", "pos")
        entry1 = AdjectiveEntry(
            word="test", synset="Syn1", weight=5,
            gloss_id="1", gloss_text="test1", position=pos
        )
        entry2 = AdjectiveEntry(
            word="test", synset="Syn2", weight=4,
            gloss_id="2", gloss_text="test2", position=pos
        )
        resolution = AdjectiveResolution(
            input_word="test",
            normalized_word="test",
            mappings=[entry1, entry2]
        )
        # Same facet, not ambiguous
        assert resolution.ambiguous is False
    
    def test_ambiguous_multiple_facets(self):
        """Test ambiguous property with multiple facets"""
        pos1 = TaxonomyPosition("Openness", "Fantasy", "", "pos")
        pos2 = TaxonomyPosition("Openness", "Feelings", "", "pos")
        entry1 = AdjectiveEntry(
            word="test", synset="Syn1", weight=5,
            gloss_id="1", gloss_text="test1", position=pos1
        )
        entry2 = AdjectiveEntry(
            word="test", synset="Syn2", weight=4,
            gloss_id="2", gloss_text="test2", position=pos2
        )
        resolution = AdjectiveResolution(
            input_word="test",
            normalized_word="test",
            mappings=[entry1, entry2]
        )
        # Different facets, ambiguous
        assert resolution.ambiguous is True


# ===== AdjectiveResolver Tests =====

class TestAdjectiveResolver:
    """Tests for AdjectiveResolver"""
    
    def test_load_from_yaml(self, neopiradj_path):
        """Test loading from YAML file"""
        resolver = AdjectiveResolver.from_yaml(neopiradj_path)
        assert resolver is not None
        stats = resolver.statistics
        assert stats["unique_adjectives"] > 0
    
    def test_resolve_known_adjective(self, adjective_resolver):
        """Test resolving a known adjective"""
        result = adjective_resolver.resolve("romantic")
        assert result.found is True
        assert result.normalized_word == "romantic"
        assert len(result.mappings) > 0
    
    def test_resolve_unknown_adjective(self, adjective_resolver):
        """Test resolving an unknown adjective"""
        result = adjective_resolver.resolve("xyznonexistent")
        assert result.found is False
        assert len(result.mappings) == 0
    
    def test_resolve_case_insensitive(self, adjective_resolver):
        """Test that resolution is case-insensitive"""
        result1 = adjective_resolver.resolve("romantic")
        result2 = adjective_resolver.resolve("ROMANTIC")
        result3 = adjective_resolver.resolve("Romantic")
        
        assert result1.found == result2.found == result3.found
        assert len(result1.mappings) == len(result2.mappings) == len(result3.mappings)
    
    def test_resolve_with_whitespace(self, adjective_resolver):
        """Test that resolution handles whitespace"""
        result1 = adjective_resolver.resolve("romantic")
        result2 = adjective_resolver.resolve("  romantic  ")
        
        assert result1.found == result2.found
        assert len(result1.mappings) == len(result2.mappings)
    
    def test_resolve_many(self, adjective_resolver):
        """Test resolving multiple adjectives"""
        results = adjective_resolver.resolve_many(["romantic", "organized", "unknown"])
        assert len(results) == 3
        assert results[0].found is True  # romantic
        assert results[1].found is True  # organized
        assert results[2].found is False  # unknown
    
    def test_exists(self, adjective_resolver):
        """Test exists method"""
        assert adjective_resolver.exists("romantic") is True
        assert adjective_resolver.exists("xyznonexistent") is False
    
    def test_search_prefix(self, adjective_resolver):
        """Test prefix search"""
        matches = adjective_resolver.search("friend")
        assert len(matches) > 0
        assert all(m.startswith("friend") for m in matches)
    
    def test_search_with_limit(self, adjective_resolver):
        """Test search with limit"""
        matches = adjective_resolver.search("a", limit=5)
        assert len(matches) <= 5
    
    def test_get_all_adjectives(self, adjective_resolver):
        """Test getting all adjectives"""
        all_adj = adjective_resolver.get_all_adjectives()
        assert len(all_adj) > 0
        assert all_adj == sorted(all_adj)  # Should be sorted
    
    def test_statistics(self, adjective_resolver):
        """Test statistics property"""
        stats = adjective_resolver.statistics
        assert "unique_adjectives" in stats
        assert "total_mappings" in stats
        assert stats["unique_adjectives"] > 0
        assert stats["total_mappings"] >= stats["unique_adjectives"]


# ===== SchemeRegistry Tests =====

class TestSchemeRegistry:
    """Tests for SchemeRegistry"""
    
    def test_load_from_yaml(self, schemes_path):
        """Test loading from YAML file"""
        registry = SchemeRegistry.from_yaml(schemes_path)
        assert registry is not None
        stats = registry.statistics
        assert stats["total_schemes"] > 0
    
    def test_get_scheme(self, scheme_registry):
        """Test getting a specific scheme"""
        scheme = scheme_registry.get_scheme("Openness", "Fantasy", "IDEALISTICNESS")
        assert scheme is not None
        assert scheme.name == "IDEALISTICNESS"
        assert scheme.trait == "Openness"
        assert scheme.facet == "Fantasy"
    
    def test_get_scheme_not_found(self, scheme_registry):
        """Test getting non-existent scheme"""
        scheme = scheme_registry.get_scheme("Openness", "Fantasy", "NONEXISTENT")
        assert scheme is None
    
    def test_get_position_for_gloss(self, scheme_registry):
        """Test gloss to position lookup"""
        # Gloss 218 should be in Openness/Fantasy/IDEALISTICNESS
        position = scheme_registry.get_position_for_gloss("218")
        assert position is not None
        assert position.trait == "Openness"
        assert position.facet == "Fantasy"
        assert position.scheme == "IDEALISTICNESS"
    
    def test_get_position_for_goldberg_gloss(self, scheme_registry):
        """Test Goldberg questionnaire item lookup"""
        position = scheme_registry.get_position_for_gloss("Q1")
        assert position is not None
    
    def test_get_glosses_for_scheme(self, scheme_registry):
        """Test getting glosses for a scheme"""
        glosses = scheme_registry.get_glosses_for_scheme(
            "Openness", "Fantasy", "IDEALISTICNESS"
        )
        assert len(glosses) > 0
        assert all(isinstance(g, GlossEntry) for g in glosses)
    
    def test_get_glosses_for_scheme_filtered_pole(self, scheme_registry):
        """Test getting glosses filtered by pole"""
        pos_glosses = scheme_registry.get_glosses_for_scheme(
            "Openness", "Fantasy", "IDEALISTICNESS", pole="pos"
        )
        neg_glosses = scheme_registry.get_glosses_for_scheme(
            "Openness", "Fantasy", "IDEALISTICNESS", pole="neg"
        )
        
        assert len(pos_glosses) > 0
        assert len(neg_glosses) > 0
        assert all(g.position.pole == "pos" for g in pos_glosses)
        assert all(g.position.pole == "neg" for g in neg_glosses)
    
    def test_get_schemes_for_facet(self, scheme_registry):
        """Test getting all schemes for a facet"""
        schemes = scheme_registry.get_schemes_for_facet("Openness", "Fantasy")
        assert len(schemes) > 0
        assert all(s.facet == "Fantasy" for s in schemes)
    
    def test_get_schemes_for_trait(self, scheme_registry):
        """Test getting all schemes for a trait"""
        schemes = scheme_registry.get_schemes_for_trait("Openness")
        assert len(schemes) > 0
        assert all(s.trait == "Openness" for s in schemes)
    
    def test_get_operator_hints(self, scheme_registry):
        """Test getting operator hints"""
        hints = scheme_registry.get_operator_hints()
        assert len(hints) > 0
        assert isinstance(hints, dict)
    
    def test_statistics(self, scheme_registry):
        """Test statistics property"""
        stats = scheme_registry.statistics
        assert "total_schemes" in stats
        assert "total_glosses" in stats
        # Resource has ~70 schemes (papers document 69, slight variation possible)
        assert 65 <= stats["total_schemes"] <= 75
        assert stats["total_glosses"] > 0


# ===== PersonalityResolver Tests =====

class TestPersonalityResolver:
    """Tests for PersonalityResolver"""
    
    def test_load_from_yaml(self, neopiradj_path, schemes_path):
        """Test loading from YAML files"""
        resolver = PersonalityResolver.from_yaml(neopiradj_path, schemes_path)
        assert resolver is not None
    
    def test_resolve_single_adjective(self, personality_resolver):
        """Test resolving a single adjective"""
        spec = personality_resolver.resolve(["romantic"])
        
        assert spec.is_complete
        assert len(spec.resolved) > 0
        assert len(spec.unresolved) == 0
    
    def test_resolve_multiple_adjectives(self, personality_resolver):
        """Test resolving multiple adjectives"""
        spec = personality_resolver.resolve(["romantic", "organized", "friendly"])
        
        assert spec.is_complete
        assert len(spec.resolved) >= 3  # At least one per adjective
    
    def test_resolve_with_unknown(self, personality_resolver):
        """Test resolving with unknown adjective"""
        spec = personality_resolver.resolve(["romantic", "xyzunknown"])
        
        assert not spec.is_complete
        assert len(spec.resolved) > 0
        assert "xyzunknown" in spec.unresolved
    
    def test_resolved_has_scheme_info(self, personality_resolver):
        """Test that resolved adjectives have scheme information when in FFM"""
        spec = personality_resolver.resolve(["romantic"])
        
        assert len(spec.resolved) > 0
        
        # Some adjectives map to non-FFM categories (Others, Discarded)
        # which don't have schemes. Check that at least one has scheme info.
        has_scheme_count = sum(1 for r in spec.resolved if r.has_scheme)
        no_scheme_count = sum(1 for r in spec.resolved if not r.has_scheme)
        
        # "romantic" should have at least one FFM mapping with scheme
        assert has_scheme_count > 0, "Expected at least one resolved with scheme info"
        
        # Verify scheme info is complete when present
        for r in spec.resolved:
            if r.has_scheme:
                assert r.position.scheme != ""
                assert r.scheme.name != ""
    
    def test_resolved_has_glosses(self, personality_resolver):
        """Test that resolved adjectives have related glosses"""
        spec = personality_resolver.resolve(["romantic"])
        
        assert len(spec.resolved) > 0
        # At least one resolved should have glosses
        has_glosses = any(len(r.all_glosses_for_pole) > 0 for r in spec.resolved)
        assert has_glosses
    
    def test_affected_properties(self, personality_resolver):
        """Test affected_* properties"""
        spec = personality_resolver.resolve(["romantic", "organized"])
        
        assert len(spec.affected_traits) > 0
        assert len(spec.affected_facets) > 0
        assert len(spec.affected_positions) > 0
    
    def test_get_resolved_for_trait(self, personality_resolver):
        """Test filtering resolved by trait"""
        spec = personality_resolver.resolve(["romantic", "organized"])
        
        openness_resolved = spec.get_resolved_for_trait("Openness")
        # romantic should map to Openness
        assert len(openness_resolved) > 0
    
    def test_summary(self, personality_resolver):
        """Test summary generation"""
        spec = personality_resolver.resolve(["romantic", "unknown_word"])
        
        summary = spec.summary()
        assert isinstance(summary, str)
        assert "romantic" in summary or "Personality" in summary
    
    def test_suggest_adjectives(self, personality_resolver):
        """Test adjective suggestions"""
        suggestions = personality_resolver.suggest_adjectives("Openness")
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_statistics(self, personality_resolver):
        """Test combined statistics"""
        stats = personality_resolver.statistics
        assert "adjectives" in stats
        assert "schemes" in stats


# ===== Integration Tests =====

class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_resolution_pipeline(self, personality_resolver):
        """Test the complete adjective → scheme → glosses pipeline"""
        # Resolve adjective
        spec = personality_resolver.resolve(["romantic"])
        
        assert spec.is_complete
        assert len(spec.resolved) > 0
        
        # Find a resolved entry with scheme info (FFM mapping)
        ffm_resolved = [r for r in spec.resolved if r.has_scheme]
        assert len(ffm_resolved) > 0, "Expected at least one FFM mapping"
        
        resolved = ffm_resolved[0]
        
        # Has adjective info
        assert resolved.adjective.word == "romantic"
        assert resolved.adjective.weight > 0
        
        # Has position info (with scheme for FFM mappings)
        assert resolved.position.trait in ["Openness", "Conscientiousness", 
                                            "Extraversion", "Agreeableness", 
                                            "Neuroticism"]
        assert resolved.position.scheme != ""
        
        # Has scheme info
        assert resolved.scheme is not None
        assert resolved.scheme.name != ""
        
        # Has glosses
        assert len(resolved.all_glosses_for_pole) > 0
    
    def test_gloss_linkage_consistency(self, adjective_resolver, scheme_registry):
        """Test that gloss IDs link correctly between resources"""
        # Get an adjective with known gloss
        result = adjective_resolver.resolve("romantic")
        assert result.found
        
        # Get the gloss ID
        gloss_id = result.mappings[0].gloss_id
        
        # Look up in scheme registry
        position = scheme_registry.get_position_for_gloss(gloss_id)
        
        # Should find it and match trait/facet
        assert position is not None
        assert position.trait == result.mappings[0].position.trait
        assert position.facet == result.mappings[0].position.facet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])