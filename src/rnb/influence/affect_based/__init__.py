"""
Affect-based Influence Operators - RnB Interpersonal Dynamics

This module implements influence operators based on interpersonal affects
from the RnB framework's Model M.A.

From RnB papers (Bouchet & Sansonnet):
"Affects represent relationship-specific emotional states that develop through
interactions. Unlike traits (stable) and moods (short-term fluctuations),
affects are tied to specific relationships and evolve based on interaction history."

RnB Affect Dimensions:
- Dominance: Power dynamics, assertiveness in relationship
- Cooperation: Willingness to help, engagement level
- Trust: Openness, vulnerability in communication
- Familiarity: Formality vs casualness based on relationship history

Affect characteristics:
- Relationship-specific (different for each user)
- Evolve through interactions (update rules)
- Influence communication style
- Bidirectional (agent → user, user → agent)

Reference: Bouchet & Sansonnet (2009-2013), RnB Model M.A
"""

from .cooperation import (
    CooperationEngagementInfluence,
    CooperationHelpfulnessInfluence,
    CooperationVerbosityInfluence,
)
from .dominance import DominanceAssertivenessInfluence, DominanceDirectivenessInfluence
from .trust import TrustOpennessInfluence, TrustVulnerabilityInfluence

__all__ = [
    # Cooperation operators
    "CooperationVerbosityInfluence",
    "CooperationHelpfulnessInfluence",
    "CooperationEngagementInfluence",
    # Trust operators
    "TrustOpennessInfluence",
    "TrustVulnerabilityInfluence",
    # Dominance operators
    "DominanceAssertivenessInfluence",
    "DominanceDirectivenessInfluence",
]
