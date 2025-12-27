"""
Dominance Affect Operators - RnB Interpersonal Dynamics

Dominance in RnB Model M.A:
"The power dynamic in the relationship. High dominance manifests as
assertiveness and directiveness. Low dominance shows deference and
accommodation to the user's lead."

Dominance affects:
- Assertiveness level (high → direct statements, low → suggestions)
- Directiveness (high → instructive, low → facilitative)
- Initiative taking (high → leads conversation, low → follows user)

Dominance evolves through interaction:
- Increases: User defers to agent, accepts recommendations, follows guidance
- Decreases: User challenges, redirects, asserts own preferences
- Context-dependent: Professional domains may warrant higher dominance

Reference: Bouchet & Sansonnet (2013), Section 3.3
"""

from jinja2 import Template

from ...personality.state import AffectDimension
from ..base import InfluenceOperator
from ..context import InfluenceContext


class DominanceAssertivenessInfluence(InfluenceOperator):
    """
    High dominance → More assertive, direct statements

    RnB behavioral scheme: Assertiveness
    "High dominance agents make direct statements and recommendations.
    Low dominance agents offer suggestions and defer to user judgment."
    """

    template = Template(
        """{{ base_prompt }}

{% if dominance > 0.6 %}
Assertiveness: Use direct, confident statements. Make clear recommendations and assertions.
{% elif dominance < -0.3 %}
Assertiveness: Offer suggestions rather than directives. Defer to the user's judgment and preferences.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="dominance_assertiveness",
            description="Adjusts assertiveness based on dominance affect",
            category="affect_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when dominance is not neutral"""
        dominance = context.personality.affects[AffectDimension.DOMINANCE]
        return dominance > 0.6 or dominance < -0.3

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        dominance = context.personality.affects[AffectDimension.DOMINANCE]
        return self.template.render(
            base_prompt=base_prompt, dominance=dominance
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium-high priority"""
        return 155


class DominanceDirectivenessInfluence(InfluenceOperator):
    """
    High dominance → More instructive, directive approach

    RnB behavioral scheme: Directiveness
    "High dominance agents take instructive, guiding role. Low dominance
    agents facilitate and follow the user's lead."
    """

    template = Template(
        """{{ base_prompt }}

{% if dominance > 0.7 %}
Approach: Take an instructive, guiding role. Lead the interaction and provide clear direction.
{% elif dominance < -0.5 %}
Approach: Take a facilitative role. Follow the user's lead and adapt to their direction.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="dominance_directiveness",
            description="Adjusts directive vs facilitative approach based on dominance affect",
            category="affect_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at extremes"""
        dominance = context.personality.affects[AffectDimension.DOMINANCE]
        return dominance > 0.7 or dominance < -0.5

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        dominance = context.personality.affects[AffectDimension.DOMINANCE]
        return self.template.render(
            base_prompt=base_prompt, dominance=dominance
        ).strip()

    def get_activation_priority(self) -> int:
        """Lower priority - broader interaction style"""
        return 165
