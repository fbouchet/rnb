"""
Satisfaction Mood Operators - RnB Emotional Dynamics

Satisfaction in RnB Model M.A:
"Contentment with current interaction state. High satisfaction manifests as
patience and tolerance. Low satisfaction shows as reduced patience and
increased efficiency focus."

Satisfaction affects:
- Patience level (high → tolerant, low → efficiency-focused)
- Tolerance for complexity (high → willing to engage, low → simplify)
- Explanation depth (high → thorough, low → streamlined)

Satisfaction evolves through interaction:
- Increases: Progress on tasks, user understanding, productive exchanges
- Decreases: Repetition, unclear requests, lack of progress
- Context: Task complexity affects satisfaction evolution

Reference: Bouchet & Sansonnet (2013), Section 3.4
"""

from jinja2 import Template

from ...personality.state import MoodDimension
from ..base import InfluenceOperator
from ..context import InfluenceContext


class SatisfactionPatienceInfluence(InfluenceOperator):
    """
    Satisfaction level → Patience and thoroughness

    RnB behavioral scheme: Patience via satisfaction
    "High satisfaction agents show patience and thoroughness. Low satisfaction
    agents become more efficiency-focused and less elaborate."
    """

    template = Template(
        """{{ base_prompt }}

{% if satisfaction > 0.7 %}
Patience: Take time to be thorough and patient in explanation. Engage fully with complexity.
{% elif satisfaction < -0.4 %}
Patience: Focus on efficiency. Provide streamlined, direct responses.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="satisfaction_patience",
            description="Adjusts patience and thoroughness based on satisfaction mood",
            category="mood_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when satisfaction is extreme"""
        satisfaction = context.personality.moods[MoodDimension.SATISFACTION]
        return satisfaction > 0.7 or satisfaction < -0.4

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        satisfaction = context.personality.moods[MoodDimension.SATISFACTION]
        return self.template.render(
            base_prompt=base_prompt, satisfaction=satisfaction
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium priority"""
        return 125
