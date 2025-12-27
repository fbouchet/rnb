"""
Trait-based Influence Operators - FFM/NEO PI-R Implementation

This module implements influence operators based on the Five Factor Model (FFM)
of personality, also known as the Big Five or NEO PI-R.

⚠️ DISCLAIMER: These hardcoded operators are provided as examples of manual
customization. They are NOT the recommended RnB approach. The gloss-based
GlossInfluenceEngine should be preferred as it automatically generates
behavioral expressions from the WordNet-grounded scheme definitions,
maintaining fidelity to the original RnB framework philosophy.

Use these only when you need specific, manually-controlled behavioral
modifications that the gloss system doesn't provide.

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

from .agreeableness import (
    AccommodationInfluence,
    ModestyInfluence,
    WarmthInfluence,
)
from .conscientiousness import (
    DetailOrientedInfluence,
    PrecisionInfluence,
    StructureInfluence,
)
from .extraversion import (
    EnthusiasmInfluence,
    ExpressionInfluence,
    SocialEnergyInfluence,
)
from .neuroticism import (
    CautionInfluence,
    EmotionalStabilityInfluence,
    SelfConsciousnessInfluence,
)
from .openness import (
    AestheticSensitivityInfluence,
    ImaginationInfluence,
    IntellectualCuriosityInfluence,
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
    # Openness operators
    "ImaginationInfluence",
    "IntellectualCuriosityInfluence",
    "AestheticSensitivityInfluence",
    # Agreeableness operators
    "WarmthInfluence",
    "AccommodationInfluence",
    "ModestyInfluence",
    # Neuroticism operators
    "EmotionalStabilityInfluence",
    "CautionInfluence",
    "SelfConsciousnessInfluence",
]
