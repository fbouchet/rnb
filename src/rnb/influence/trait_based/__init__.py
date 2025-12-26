"""
Trait-based Influence Operators - FFM/NEO PI-R Implementation

This module implements influence operators based on the Five Factor Model (FFM)
of personality, also known as the Big Five or NEO PI-R.

From RnB papers (Bouchet & Sansonnet):
"Personality traits are stable dispositions that consistently influence behavior.
The FFM provides a comprehensive taxonomy: Openness, Conscientiousness, 
Extraversion, Agreeableness, Neuroticism. Each trait decomposes into 6 facets
(NEO PI-R), and each facet maps to specific behavioral schemes."

Trait characteristics:
- Stable over time (change slowly, if at all)
- Context-independent (apply across situations)
- Bipolar (high vs low values produce opposite behaviors)
- Hierarchical (traits → facets → behavioral schemes)

Reference: Costa & McCrae (1992), NEO PI-R; Bouchet & Sansonnet (2013)
"""

from .conscientiousness import (
    StructureInfluence,
    DetailOrientedInfluence,
    PrecisionInfluence
)
from .extraversion import (
    EnthusiasmInfluence,
    ExpressionInfluence,
    SocialEnergyInfluence
)

__all__ = [
    # Conscientiousness operators
    "StructureInfluence",
    "DetailOrientedInfluence", 
    "PrecisionInfluence",
    # Extraversion operators
    "EnthusiasmInfluence",
    "ExpressionInfluence",
    "SocialEnergyInfluence",
]