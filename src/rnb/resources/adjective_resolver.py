"""
Adjective Resolver for RnB personality resources.

Resolves personality adjectives to their positions in the
FFM/NEO PI-R taxonomy using the neopiradj resource.

The neopiradj resource maps 1055 personality adjectives to:
- WordNet synsets (senses)
- Glosses (behavioral definitions)
- Taxonomy positions (trait/facet/valence)
- Salience weights (1-9 scale)

Reference: Bouchet & Sansonnet, "Implementing WordNet Personality
Adjectives as Influences on Rational Agents" (IJCISM 2010)
"""

import logging
from pathlib import Path

import yaml

from .models import (
    AdjectiveEntry,
    AdjectiveResolution,
    TaxonomyPosition,
    TraitName,
)

logger = logging.getLogger(__name__)


class AdjectiveResolver:
    """
    Resolves personality adjectives to their RnB taxonomy positions.

    Uses the neopiradj resource which maps 1055 personality adjectives
    to their WordNet synsets and positions in the FFM/NEO PI-R taxonomy.

    Example:
        resolver = AdjectiveResolver.from_yaml("neopiradj.yaml")

        result = resolver.resolve("romantic")
        print(result.found)  # True
        print(result.mappings[0].position)  # Openness/Fantasy/.../+
        print(result.mappings[0].weight)  # 5

        # Search for adjectives
        matches = resolver.search("friend")  # ["friendly", "friendless", ...]
    """

    def __init__(self, neopiradj_data: dict, warn_unresolved: bool = True):
        """
        Initialize with parsed YAML data.

        Use from_yaml() class method for file loading.

        Args:
            neopiradj_data: Parsed YAML dictionary
            warn_unresolved: If True, log warnings for unresolved adjectives
        """
        self._raw_data = neopiradj_data
        self._warn_unresolved = warn_unresolved
        self._adjective_index: dict[str, list[AdjectiveEntry]] = {}
        self._build_index()

    @classmethod
    def from_yaml(
        cls, path: Path | str, warn_unresolved: bool = True
    ) -> "AdjectiveResolver":
        """
        Load resolver from YAML file.

        Args:
            path: Path to neopiradj.yaml file
            warn_unresolved: If True, log warnings for unresolved adjectives

        Returns:
            Initialized AdjectiveResolver

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If file is not valid YAML
        """
        path = Path(path)
        logger.info(f"Loading adjective resource from {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        resolver = cls(data, warn_unresolved=warn_unresolved)
        logger.info(
            f"Loaded {resolver.statistics['unique_adjectives']} unique adjectives"
        )

        return resolver

    def _build_index(self) -> None:
        """
        Build inverted index: adjective_word → list[AdjectiveEntry]

        Iterates through the full taxonomy to find all adjective occurrences.
        The index enables O(1) lookup by adjective word.
        """
        traits_data = self._raw_data.get("traits", {})

        for trait_name, facets in traits_data.items():
            normalized_trait = self._normalize_trait(trait_name)

            if not isinstance(facets, dict):
                continue

            for facet_name, valences in facets.items():
                if not isinstance(valences, dict):
                    continue

                for valence, glosses in valences.items():
                    # Only process + and - valences
                    if valence not in ("+", "-"):
                        continue

                    if not isinstance(glosses, dict):
                        continue

                    pole = "pos" if valence == "+" else "neg"

                    for gloss_id, gloss_data in glosses.items():
                        if not isinstance(gloss_data, dict):
                            continue

                        gloss_text = gloss_data.get("gloss", "")
                        adjectives = gloss_data.get("adjectives", [])

                        if not isinstance(adjectives, list):
                            continue

                        for adj_data in adjectives:
                            if not isinstance(adj_data, dict):
                                continue

                            word = adj_data.get("word", "")
                            if not word:
                                continue

                            word_lower = word.lower().strip()

                            # Create position (scheme will be filled by SchemeRegistry)
                            # For now, we use empty string for scheme
                            position = TaxonomyPosition(
                                trait=normalized_trait,
                                facet=facet_name,
                                scheme="",  # Will be resolved via gloss_id link
                                pole=pole,
                            )

                            entry = AdjectiveEntry(
                                word=word_lower,
                                synset=adj_data.get("synset", ""),
                                weight=int(adj_data.get("weight", 1)),
                                gloss_id=str(gloss_id),
                                gloss_text=gloss_text,
                                position=position,
                            )

                            if word_lower not in self._adjective_index:
                                self._adjective_index[word_lower] = []
                            self._adjective_index[word_lower].append(entry)

    def _normalize_trait(self, trait_name: str) -> str:
        """
        Normalize trait name to standard capitalized form.

        Handles variations like "OPENNESS", "openness", "Openness".
        """
        name_lower = trait_name.lower()
        for trait in TraitName:
            if trait.value.lower() == name_lower:
                return trait.value
        # Fallback: capitalize first letter
        return trait_name.capitalize()

    def resolve(self, adjective: str) -> AdjectiveResolution:
        """
        Resolve a single adjective to all its taxonomy positions.

        Args:
            adjective: A personality adjective (e.g., "romantic", "lazy")

        Returns:
            AdjectiveResolution with all mappings found (may be empty)
        """
        normalized = adjective.lower().strip()
        mappings = self._adjective_index.get(normalized, [])

        resolution = AdjectiveResolution(
            input_word=adjective,
            normalized_word=normalized,
            mappings=list(mappings),  # Copy to avoid mutation
        )

        if not resolution.found and self._warn_unresolved:
            logger.warning(f"Adjective '{adjective}' not found in RnB resource")

        if resolution.ambiguous:
            facets = {m.position.facet_key for m in resolution.mappings}
            logger.debug(
                f"Adjective '{adjective}' is ambiguous, maps to facets: {facets}"
            )

        return resolution

    def resolve_many(self, adjectives: list[str]) -> list[AdjectiveResolution]:
        """
        Resolve multiple adjectives.

        Args:
            adjectives: List of personality adjectives

        Returns:
            List of resolutions in same order as input
        """
        return [self.resolve(adj) for adj in adjectives]

    def exists(self, adjective: str) -> bool:
        """
        Check if an adjective exists in the resource.

        Args:
            adjective: Adjective to check

        Returns:
            True if adjective has at least one mapping
        """
        normalized = adjective.lower().strip()
        return normalized in self._adjective_index

    def search(self, prefix: str, limit: int = 10) -> list[str]:
        """
        Search for adjectives by prefix (useful for autocomplete).

        Args:
            prefix: Starting characters to match
            limit: Maximum results to return

        Returns:
            List of matching adjective words, sorted alphabetically
        """
        prefix_lower = prefix.lower().strip()
        matches = [
            word
            for word in self._adjective_index.keys()
            if word.startswith(prefix_lower)
        ]
        return sorted(matches)[:limit]

    def get_all_adjectives(self) -> list[str]:
        """
        Return all known adjectives.

        Returns:
            Sorted list of all adjective words in the resource
        """
        return sorted(self._adjective_index.keys())

    def get_adjectives_for_facet(
        self, trait: str, facet: str, valence: str | None = None
    ) -> list[AdjectiveEntry]:
        """
        Get all adjectives mapping to a specific facet.

        Useful for exploring what adjectives express a given facet.

        Args:
            trait: Trait name (e.g., "Openness")
            facet: Facet name (e.g., "Fantasy")
            valence: "+" or "-" to filter by pole, None for both

        Returns:
            List of AdjectiveEntry objects for that facet
        """
        results = []
        target_pole = None
        if valence == "+":
            target_pole = "pos"
        elif valence == "-":
            target_pole = "neg"

        for entries in self._adjective_index.values():
            for entry in entries:
                if (
                    entry.position.trait.lower() == trait.lower()
                    and entry.position.facet.lower() == facet.lower()
                ):
                    if target_pole is None or entry.position.pole == target_pole:
                        results.append(entry)

        return results

    def get_adjectives_by_weight(self, min_weight: int = 1) -> list[AdjectiveEntry]:
        """
        Get all adjectives with weight >= min_weight.

        Useful for finding the most salient personality adjectives.

        Args:
            min_weight: Minimum weight threshold (1-9)

        Returns:
            List of AdjectiveEntry objects, sorted by weight descending
        """
        results = []
        for entries in self._adjective_index.values():
            for entry in entries:
                if entry.weight >= min_weight:
                    results.append(entry)

        return sorted(results, key=lambda e: e.weight, reverse=True)

    @property
    def statistics(self) -> dict:
        """
        Return statistics about the loaded resource.

        Returns:
            Dictionary with counts and averages
        """
        total_mappings = sum(len(v) for v in self._adjective_index.values())
        weights = [
            entry.weight
            for entries in self._adjective_index.values()
            for entry in entries
        ]

        return {
            "unique_adjectives": len(self._adjective_index),
            "total_mappings": total_mappings,
            "avg_mappings_per_adjective": (
                total_mappings / len(self._adjective_index)
                if self._adjective_index
                else 0
            ),
            "weight_distribution": {w: weights.count(w) for w in range(1, 10)},
            "max_weight_adjectives": [
                word
                for word, entries in self._adjective_index.items()
                if any(e.weight >= 8 for e in entries)
            ],
        }
