"""
RnB Influence System - Behavioral Engine Implementation

This module implements the Behavioral Engine (B) from the RnB framework,
which modifies rational agent outputs through personality-driven influence operators.

Key components:
- InfluenceContext: RnB Model M context (M.A, M.U, M.S, M.T)
- InfluenceOperator: Base class for behavioral schemes
- OperatorRegistry: Activation matrix management
- Concrete operators: Trait/mood/affect-based behavioral modifications

Reference: Bouchet & Sansonnet (2009-2013), RnB Framework papers
"""

from .base import CompositeInfluenceOperator, InfluenceOperator
from .context import InfluenceContext
from .gloss_engine import (
    ActiveGloss,
    ContextStyle,
    GlossInfluenceEngine,
)
from .gloss_operator import (
    GlossBasedInfluence,
    TraitGlossInfluence,
    create_agreeableness_gloss_operator,
    create_conscientiousness_gloss_operator,
    create_extraversion_gloss_operator,
    create_neuroticism_gloss_operator,
    create_openness_gloss_operator,
)
from .registry import OperatorRegistry

__all__ = [
    # Base
    "InfluenceContext",
    "InfluenceOperator",
    "CompositeInfluenceOperator",
    "OperatorRegistry",
    # Engine
    "GlossInfluenceEngine",
    "ContextStyle",
    "ActiveGloss",
    # Operators
    "GlossBasedInfluence",
    "TraitGlossInfluence",
    # Factory functions
    "create_openness_gloss_operator",
    "create_conscientiousness_gloss_operator",
    "create_extraversion_gloss_operator",
    "create_agreeableness_gloss_operator",
    "create_neuroticism_gloss_operator",
]
