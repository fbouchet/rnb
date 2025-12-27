"""
Scheme Registry for RnB behavioral schemes.

Provides access to the 69 bipolar behavioral schemes that define
personality-driven behaviors in the RnB framework.

Each scheme:
- Belongs to a facet (30 facets, ~2.3 schemes per facet)
- Has positive and negative poles
- Contains glosses defining the behavioral semantics
- Has an operator hint suggesting implementation approach

Reference: Bouchet & Sansonnet (2013), "Influence of FFM/NEO PI-R
personality traits on the rational process of autonomous agents"
"""

import logging
from pathlib import Path
from typing import Literal

import yaml

from .models import (
    GlossEntry,
    PoleInfo,
    SchemeInfo,
    TaxonomyPosition,
    TraitName,
)

logger = logging.getLogger(__name__)


class SchemeRegistry:
    """
    Registry of behavioral schemes from the RnB schemes resource.

    Provides lookup from gloss IDs to schemes, and access to
    scheme glosses for behavioral expression.

    The schemes resource contains 69 bipolar schemes organized
    under the 30 NEO PI-R facets, with 766 total glosses.

    Example:
        registry = SchemeRegistry.from_yaml("schemes.yaml")

        # Get a specific scheme
        scheme = registry.get_scheme("Openness", "Fantasy", "IDEALISTICNESS")
        print(scheme.poles["pos"].name)  # "IDEALISTIC"

        # Find scheme for a gloss
        position = registry.get_position_for_gloss("218")
        print(position)  # Openness/Fantasy/IDEALISTICNESS/+

        # Get all glosses for a scheme's positive pole
        glosses = registry.get_glosses_for_scheme(
            "Openness", "Fantasy", "IDEALISTICNESS", pole="pos"
        )
    """

    def __init__(self, schemes_data: dict):
        """
        Initialize with parsed YAML data.

        Use from_yaml() class method for file loading.

        Args:
            schemes_data: Parsed YAML dictionary
        """
        self._raw_data = schemes_data
        self._schemes: dict[str, SchemeInfo] = {}  # scheme_key → SchemeInfo
        self._gloss_to_position: dict[str, TaxonomyPosition] = {}  # gloss_id → position
        self._build_registry()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SchemeRegistry":
        """
        Load registry from YAML file.

        Args:
            path: Path to schemes.yaml file

        Returns:
            Initialized SchemeRegistry

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If file is not valid YAML
        """
        path = Path(path)
        logger.info(f"Loading schemes resource from {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        registry = cls(data)
        stats = registry.statistics
        logger.info(
            f"Loaded {stats['total_schemes']} schemes with {stats['total_glosses']} glosses"
        )

        return registry

    def _build_registry(self) -> None:
        """
        Build scheme registry and gloss-to-position index.

        Iterates through the full taxonomy to build:
        1. _schemes: Fast lookup of SchemeInfo by key
        2. _gloss_to_position: Maps gloss IDs to their full positions
        """
        traits_data = self._raw_data.get("traits", {})

        for trait_name, facets in traits_data.items():
            normalized_trait = self._normalize_trait(trait_name)

            if not isinstance(facets, dict):
                continue

            for facet_name, schemes in facets.items():
                if not isinstance(schemes, dict):
                    continue

                for scheme_name, scheme_data in schemes.items():
                    if not isinstance(scheme_data, dict):
                        continue

                    # Extract operator hint
                    operator_hint = scheme_data.get("op", "")

                    # Build poles
                    poles: dict[str, PoleInfo] = {}
                    poles_data = scheme_data.get("poles", {})

                    if not isinstance(poles_data, dict):
                        continue

                    for pole_key, pole_data in poles_data.items():
                        if pole_key not in ("pos", "neg"):
                            continue

                        if not isinstance(pole_data, dict):
                            continue

                        pole_name = pole_data.get("name", pole_key.upper())
                        glosses_raw = pole_data.get("glosses", {})

                        # Normalize glosses to dict[str, str]
                        glosses: dict[str, str] = {}
                        if isinstance(glosses_raw, dict):
                            for gid, gtext in glosses_raw.items():
                                gid_str = str(gid)
                                glosses[gid_str] = str(gtext)

                                # Index this gloss to its full position
                                position = TaxonomyPosition(
                                    trait=normalized_trait,
                                    facet=facet_name,
                                    scheme=scheme_name,
                                    pole=pole_key,
                                )
                                self._gloss_to_position[gid_str] = position

                        poles[pole_key] = PoleInfo(name=pole_name, glosses=glosses)

                    # Create scheme info
                    scheme_info = SchemeInfo(
                        name=scheme_name,
                        operator_hint=operator_hint,
                        trait=normalized_trait,
                        facet=facet_name,
                        poles=poles,
                    )

                    scheme_key = f"{normalized_trait}_{facet_name}_{scheme_name}"
                    self._schemes[scheme_key] = scheme_info

    def _normalize_trait(self, trait_name: str) -> str:
        """
        Normalize trait name to standard capitalized form.
        """
        name_lower = trait_name.lower()
        for trait in TraitName:
            if trait.value.lower() == name_lower:
                return trait.value
        return trait_name.capitalize()

    def get_scheme(self, trait: str, facet: str, scheme_name: str) -> SchemeInfo | None:
        """
        Get a specific scheme by its position.

        Args:
            trait: Trait name (e.g., "Openness")
            facet: Facet name (e.g., "Fantasy")
            scheme_name: Scheme name (e.g., "IDEALISTICNESS")

        Returns:
            SchemeInfo or None if not found
        """
        # Try exact match first
        key = f"{trait}_{facet}_{scheme_name}"
        if key in self._schemes:
            return self._schemes[key]

        # Try case-insensitive match
        key_lower = key.lower()
        for k, v in self._schemes.items():
            if k.lower() == key_lower:
                return v

        return None

    def get_scheme_by_key(self, scheme_key: str) -> SchemeInfo | None:
        """
        Get scheme by its full key.

        Args:
            scheme_key: Key like "Openness_Fantasy_IDEALISTICNESS"

        Returns:
            SchemeInfo or None
        """
        return self._schemes.get(scheme_key)

    def get_position_for_gloss(self, gloss_id: str) -> TaxonomyPosition | None:
        """
        Find which scheme a gloss belongs to.

        This is the critical link between neopiradj (adjective→gloss)
        and schemes (gloss→scheme). It completes the partial position
        from AdjectiveResolver with the scheme information.

        Args:
            gloss_id: Gloss identifier (e.g., "218" or "Q1")

        Returns:
            Complete TaxonomyPosition or None if gloss not found
        """
        return self._gloss_to_position.get(str(gloss_id))

    def get_glosses_for_scheme(
        self,
        trait: str,
        facet: str,
        scheme_name: str,
        pole: Literal["pos", "neg"] | None = None,
    ) -> list[GlossEntry]:
        """
        Get all glosses for a scheme (optionally filtered by pole).

        Args:
            trait, facet, scheme_name: Position identifiers
            pole: "pos" or "neg" to filter, None for both poles

        Returns:
            List of GlossEntry objects with full position info
        """
        scheme = self.get_scheme(trait, facet, scheme_name)
        if not scheme:
            return []

        results = []
        poles_to_check = [pole] if pole else ["pos", "neg"]

        for p in poles_to_check:
            if p not in scheme.poles:
                continue

            pole_info = scheme.poles[p]
            for gloss_id, gloss_text in pole_info.glosses.items():
                position = TaxonomyPosition(
                    trait=trait, facet=facet, scheme=scheme_name, pole=p
                )
                results.append(
                    GlossEntry(id=gloss_id, text=gloss_text, position=position)
                )

        return results

    def get_schemes_for_facet(self, trait: str, facet: str) -> list[SchemeInfo]:
        """
        Get all schemes under a specific facet.

        Args:
            trait: Trait name
            facet: Facet name

        Returns:
            List of SchemeInfo objects
        """
        prefix = f"{trait}_{facet}_"
        prefix_lower = prefix.lower()

        return [
            scheme
            for key, scheme in self._schemes.items()
            if key.lower().startswith(prefix_lower)
        ]

    def get_schemes_for_trait(self, trait: str) -> list[SchemeInfo]:
        """
        Get all schemes under a specific trait.

        Args:
            trait: Trait name

        Returns:
            List of SchemeInfo objects
        """
        trait_lower = trait.lower()
        return [
            scheme
            for scheme in self._schemes.values()
            if scheme.trait.lower() == trait_lower
        ]

    def get_all_schemes(self) -> list[SchemeInfo]:
        """
        Return all schemes in the registry.

        Returns:
            List of all SchemeInfo objects
        """
        return list(self._schemes.values())

    def get_operator_hints(self) -> dict[str, list[str]]:
        """
        Get unique operator hints and which schemes use them.

        Useful for understanding the operator categories in the resource.

        Returns:
            Dictionary mapping operator_hint to list of scheme names
        """
        hints: dict[str, list[str]] = {}
        for scheme in self._schemes.values():
            hint = scheme.operator_hint
            if hint not in hints:
                hints[hint] = []
            hints[hint].append(scheme.name)
        return hints

    def get_schemes_by_operator_hint(self, operator_hint: str) -> list[SchemeInfo]:
        """
        Get all schemes with a specific operator hint.

        Args:
            operator_hint: Operator hint to match (e.g., "cooperation")

        Returns:
            List of matching SchemeInfo objects
        """
        return [
            scheme
            for scheme in self._schemes.values()
            if scheme.operator_hint == operator_hint
        ]

    def get_facets_for_trait(self, trait: str) -> list[str]:
        """
        Get all facet names under a trait.

        Args:
            trait: Trait name

        Returns:
            Sorted list of facet names
        """
        trait_lower = trait.lower()
        facets = {
            scheme.facet
            for scheme in self._schemes.values()
            if scheme.trait.lower() == trait_lower
        }
        return sorted(facets)

    @property
    def statistics(self) -> dict:
        """
        Return statistics about the loaded resource.

        Returns:
            Dictionary with counts and breakdowns
        """
        total_glosses = sum(
            len(pole.glosses)
            for scheme in self._schemes.values()
            for pole in scheme.poles.values()
        )

        # Count glosses by source
        wordnet_glosses = sum(
            1 for gid in self._gloss_to_position.keys() if not gid.startswith("Q")
        )
        goldberg_glosses = sum(
            1 for gid in self._gloss_to_position.keys() if gid.startswith("Q")
        )

        # Schemes per trait
        schemes_per_trait = {}
        for trait in TraitName:
            count = len(self.get_schemes_for_trait(trait.value))
            schemes_per_trait[trait.value] = count

        return {
            "total_schemes": len(self._schemes),
            "total_glosses": total_glosses,
            "wordnet_glosses": wordnet_glosses,
            "goldberg_glosses": goldberg_glosses,
            "unique_operator_hints": len(
                set(s.operator_hint for s in self._schemes.values())
            ),
            "schemes_per_trait": schemes_per_trait,
            "avg_schemes_per_facet": len(self._schemes) / 30,  # 30 facets
        }
