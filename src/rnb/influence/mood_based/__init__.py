"""
Mood-based Influence Operators - RnB Emotional Dynamics

This module implements influence operators based on dynamic mood states
from the RnB framework's Model M.A.

From RnB papers (Bouchet & Sansonnet):
"Moods are short-term emotional states that fluctuate during interactions.
Unlike traits (stable) and affects (relationship-specific), moods represent
the agent's current emotional condition and evolve rapidly based on events."

RnB Mood Dimensions:
- Happiness: General positive/negative emotional state
- Energy: Activation level, vigor
- Satisfaction: Contentment with current state
- Calmness: Composure vs agitation

Mood characteristics:
- Dynamic (change during conversation)
- Event-driven (triggered by interaction events)
- Short-term (decay over time)
- Independent of specific relationship

Mood update rules (examples from RnB):
- Success event → happiness +0.1, satisfaction +0.1
- Frustration → happiness -0.2, calmness -0.1, energy -0.1
- Long interaction → energy -0.1 (fatigue)
- Rest/time → gradual return to baseline

Reference: Bouchet & Sansonnet (2009-2013), RnB Model M.A
"""

from .energy import (
    EnergyLengthInfluence,
    EnergyInitiativeInfluence
)
from .happiness import (
    HappinessToneInfluence,
    HappinessPositivityInfluence
)
from .satisfaction import (
    SatisfactionPatienceInfluence
)

__all__ = [
    # Energy operators
    "EnergyLengthInfluence",
    "EnergyInitiativeInfluence",
    # Happiness operators
    "HappinessToneInfluence",
    "HappinessPositivityInfluence",
    # Satisfaction operators
    "SatisfactionPatienceInfluence",
]