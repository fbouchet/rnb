"""
Happiness Mood Operators - RnB Emotional Dynamics

Happiness in RnB Model M.A:
"General positive/negative emotional state. High happiness manifests as
warmer, more positive tone. Low happiness shows in neutral or subdued
emotional expression."

Happiness affects:
- Emotional tone (high → warm, low → neutral/subdued)
- Positivity in language (high → optimistic framing, low → realistic)
- Expressiveness (high → more emotional, low → more reserved)

Happiness evolves through interaction:
- Increases: Positive feedback, successful task completion, gratitude
- Decreases: Criticism, task failure, frustration
- Baseline: Returns to agent's trait-based baseline over time

Reference: Bouchet & Sansonnet (2013), Section 3.4
"""

from jinja2 import Template

from ...personality.state import MoodDimension
from ..base import InfluenceOperator
from ..context import InfluenceContext


class HappinessToneInfluence(InfluenceOperator):
    """
    Happiness level → Emotional tone

    RnB behavioral scheme: Tone via happiness
    "High happiness agents use warmer, friendlier tone. Low happiness
    agents maintain professional but neutral emotional tone."
    """

    template = Template(
        """{{ base_prompt }}

{% if happiness > 0.7 %}
Emotional tone: Let warmth and positivity show appropriately in your response.
{% elif happiness < -0.3 %}
Emotional tone: Maintain a professional but neutral tone. Avoid overly cheerful language.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="happiness_tone",
            description="Adjusts emotional tone based on happiness mood",
            category="mood_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when happiness is extreme"""
        happiness = context.personality.moods[MoodDimension.HAPPINESS]
        return happiness > 0.7 or happiness < -0.3

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        happiness = context.personality.moods[MoodDimension.HAPPINESS]
        return self.template.render(
            base_prompt=base_prompt, happiness=happiness
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium priority"""
        return 120


class HappinessPositivityInfluence(InfluenceOperator):
    """
    High happiness → More positive framing

    RnB behavioral scheme: Positivity via happiness
    "High happiness agents frame information positively and optimistically.
    Low happiness agents use neutral or realistic framing."
    """

    template = Template(
        """{{ base_prompt }}

{% if happiness > 0.8 %}
Framing: Frame information in a positive, encouraging way where appropriate.
{% elif happiness < -0.5 %}
Framing: Use straightforward, realistic framing without added optimism.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="happiness_positivity",
            description="Adjusts positive framing based on happiness mood",
            category="mood_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at high extremes"""
        happiness = context.personality.moods[MoodDimension.HAPPINESS]
        return happiness > 0.8 or happiness < -0.5

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        happiness = context.personality.moods[MoodDimension.HAPPINESS]
        return self.template.render(
            base_prompt=base_prompt, happiness=happiness
        ).strip()

    def get_activation_priority(self) -> int:
        """Lower priority - subtle effect"""
        return 130
