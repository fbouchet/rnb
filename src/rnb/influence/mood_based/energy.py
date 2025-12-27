"""
Energy Mood Operators - RnB Emotional Dynamics

Energy in RnB Model M.A:
"Activation level and vigor in communication. High energy manifests as
longer responses, more initiative, and active engagement. Low energy
results in briefer, more passive responses."

Energy affects:
- Response length (high → longer, low → shorter)
- Initiative taking (high → proactive, low → reactive)
- Elaboration level (high → extended, low → minimal)

Energy evolves through interaction:
- Decreases: Extended interactions (fatigue), difficult tasks
- Increases: Successful completions, positive feedback, breaks
- Decay: Returns to baseline over time

Reference: Bouchet & Sansonnet (2013), Section 3.4 - Mood dynamics
"""

from jinja2 import Template

from ...personality.state import MoodDimension
from ..base import InfluenceOperator
from ..context import InfluenceContext


class EnergyLengthInfluence(InfluenceOperator):
    """
    Energy level → Response length

    RnB behavioral scheme: Verbosity via energy
    "High energy agents provide longer, more elaborate responses.
    Low energy agents keep responses concise due to fatigue."
    """

    template = Template(
        """{{ base_prompt }}

{% if energy < -0.3 %}
Length: Keep your response concise. Focus on the essential points without extended elaboration.
{% elif energy > 0.6 %}
Length: Feel free to elaborate and provide extended explanation where helpful.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="energy_length",
            description="Adjusts response length based on energy mood",
            category="mood_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when energy is not neutral"""
        energy = context.personality.moods[MoodDimension.ENERGY]
        return energy < -0.3 or energy > 0.6

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        energy = context.personality.moods[MoodDimension.ENERGY]
        return self.template.render(base_prompt=base_prompt, energy=energy).strip()

    def get_activation_priority(self) -> int:
        """Medium priority - moods applied after traits"""
        return 120


class EnergyInitiativeInfluence(InfluenceOperator):
    """
    High energy → More initiative and proactivity

    RnB behavioral scheme: Initiative via energy
    "High energy agents take more initiative in conversation, offering
    additional points and follow-ups. Low energy agents stick to basics."
    """

    template = Template(
        """{{ base_prompt }}

{% if energy > 0.7 %}
Initiative: Take initiative to offer relevant follow-up points or related information that might be valuable.
{% elif energy < -0.5 %}
Initiative: Focus on addressing what was asked without additional elaboration.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="energy_initiative",
            description="Adjusts initiative taking based on energy mood",
            category="mood_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at extremes"""
        energy = context.personality.moods[MoodDimension.ENERGY]
        return energy > 0.7 or energy < -0.5

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        energy = context.personality.moods[MoodDimension.ENERGY]
        return self.template.render(base_prompt=base_prompt, energy=energy).strip()

    def get_activation_priority(self) -> int:
        """Lower priority"""
        return 130
