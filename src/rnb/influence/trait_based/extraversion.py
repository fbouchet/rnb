"""
Extraversion Trait Operators - RnB Behavioral Schemes

Extraversion in FFM/NEO PI-R:
"The tendency toward social engagement, assertiveness, and positive emotionality.
High extraversion is associated with enthusiasm, expressiveness, and social energy."

NEO PI-R Facets:
- E1: Warmth (friendliness, approachability)
- E2: Gregariousness (sociability, preference for company)
- E3: Assertiveness (dominance, forcefulness)
- E4: Activity (pace, energy level)
- E5: Excitement-Seeking (stimulation, risk-taking)
- E6: Positive Emotions (joy, enthusiasm)

RnB Behavioral Schemes mapped from Extraversion:
- Enthusiasm and expressiveness in tone
- Energy level in communication
- Assertiveness in statements
- Emotional expressiveness

Reference: Bouchet & Sansonnet (2013), Section 3.2
"""

from jinja2 import Template

from ...personality.taxonomy import Trait
from ..base import InfluenceOperator
from ..context import InfluenceContext


class EnthusiasmInfluence(InfluenceOperator):
    """
    High extraversion → More enthusiastic, energetic tone

    Maps to NEO PI-R facet E6 (Positive Emotions).

    RnB behavioral scheme: Enthusiasm
    "High extraversion agents express enthusiasm and positive energy in their
    communication. Low extraversion agents are more reserved and measured."
    """

    template = Template(
        """{{ base_prompt }}

{% if extraversion > 0.5 %}
Tone: Be enthusiastic and energetic in your response. Let your positive energy show.
{% elif extraversion < -0.5 %}
Tone: Be reserved and measured in your response. Maintain a calm, understated approach.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="extraversion_enthusiasm",
            description="Adjusts enthusiasm level based on extraversion (E6: Positive Emotions)",
            category="trait_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when extraversion is extreme (|value| > 0.5)"""
        extraversion = context.personality.get_trait(Trait.EXTRAVERSION)
        return abs(extraversion) > 0.5

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        extraversion = context.personality.get_trait(Trait.EXTRAVERSION)
        return self.template.render(
            base_prompt=base_prompt, extraversion=extraversion
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium priority (stable trait, before dynamic moods)"""
        return 70


class ExpressionInfluence(InfluenceOperator):
    """
    High extraversion → More expressive, animated communication

    Maps to NEO PI-R facet E1 (Warmth) and E6 (Positive Emotions).

    RnB behavioral scheme: Expressiveness
    "High extraversion agents are expressive and animated in communication.
    Low extraversion agents are more restrained and formal."
    """

    template = Template(
        """{{ base_prompt }}

{% if extraversion > 0.6 %}
Expression: Be expressive and engaging. Use vivid language and examples that bring the topic to life.
{% elif extraversion < -0.6 %}
Expression: Be measured and formal. Maintain professional restraint in expression.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="extraversion_expression",
            description="Adjusts expressiveness based on extraversion (E1: Warmth, E6: Positive Emotions)",
            category="trait_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate at higher extremes (|value| > 0.6)"""
        extraversion = context.personality.get_trait(Trait.EXTRAVERSION)
        return abs(extraversion) > 0.6

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        extraversion = context.personality.get_trait(Trait.EXTRAVERSION)
        return self.template.render(
            base_prompt=base_prompt, extraversion=extraversion
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium-high priority"""
        return 75


class SocialEnergyInfluence(InfluenceOperator):
    """
    High extraversion → More interactive, engaging approach

    Maps to NEO PI-R facet E2 (Gregariousness) and E4 (Activity).

    RnB behavioral scheme: Social Energy
    "High extraversion agents adopt an interactive, engaging communication style.
    Low extraversion agents prefer concise, less interactive exchanges."
    """

    template = Template(
        """{{ base_prompt }}

{% if extraversion > 0.7 %}
Engagement: Adopt an interactive, conversational approach. Engage actively with the topic and invite further discussion.
{% elif extraversion < -0.7 %}
Engagement: Keep the interaction focused and concise. Provide the information efficiently without extended engagement.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="extraversion_social_energy",
            description="Adjusts interaction style based on extraversion (E2: Gregariousness, E4: Activity)",
            category="trait_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at very high extremes (|value| > 0.7)"""
        extraversion = context.personality.get_trait(Trait.EXTRAVERSION)
        return abs(extraversion) > 0.7

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        extraversion = context.personality.get_trait(Trait.EXTRAVERSION)
        return self.template.render(
            base_prompt=base_prompt, extraversion=extraversion
        ).strip()

    def get_activation_priority(self) -> int:
        """Lower priority (broader interaction style, applied after specific traits)"""
        return 80
