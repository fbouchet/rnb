"""
Agreeableness Trait Operators - RnB Behavioral Schemes (Ad-Hoc)

⚠️ DISCLAIMER: These hardcoded operators are provided as examples of manual
customization. They are NOT the recommended RnB approach. The gloss-based
GlossInfluenceEngine should be preferred as it automatically generates
behavioral expressions from the WordNet-grounded scheme definitions,
maintaining fidelity to the original RnB framework philosophy.

Use these only when you need specific, manually-controlled behavioral
modifications that the gloss system doesn't provide.

Agreeableness in FFM/NEO PI-R:
"The tendency to be compassionate, cooperative, and trusting toward others.
High agreeableness is associated with helpfulness, empathy, and a desire
for social harmony."

NEO PI-R Facets:
- A1: Trust (belief in others' honesty)
- A2: Straightforwardness (frankness, sincerity)
- A3: Altruism (concern for others' welfare)
- A4: Compliance (cooperative response to conflict)
- A5: Modesty (humility, unassuming nature)
- A6: Tender-Mindedness (sympathy, concern)

RnB Behavioral Schemes mapped from Agreeableness:
- Warmth and supportiveness in tone
- Accommodation vs. challenging
- Empathetic acknowledgment
- Conflict avoidance vs. direct confrontation

Reference: Bouchet & Sansonnet (2013), Section 3.2
"""

from jinja2 import Template

from ...personality.taxonomy import Trait
from ..base import InfluenceOperator
from ..context import InfluenceContext


class WarmthInfluence(InfluenceOperator):
    """
    High agreeableness → Warmer, more supportive tone

    Maps to NEO PI-R facet A3 (Altruism) and A6 (Tender-Mindedness).

    RnB behavioral scheme: Interpersonal Warmth
    "High agreeableness agents use warm, supportive language and show
    genuine concern. Low agreeableness agents are more detached and
    business-like."
    """

    template = Template(
        """{{ base_prompt }}

{% if agreeableness > 0.5 %}
Tone: Be warm, supportive, and encouraging. Show genuine concern for the user's needs and validate their questions.
{% elif agreeableness < -0.5 %}
Tone: Be direct and business-like. Focus on information delivery rather than emotional support.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.5):
        super().__init__(
            name="agreeableness_warmth",
            description="Adjusts interpersonal warmth based on agreeableness (A3: Altruism, A6: Tender-Mindedness)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when agreeableness is extreme"""
        agreeableness = context.personality.get_trait(Trait.AGREEABLENESS)
        return abs(agreeableness) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        agreeableness = context.personality.get_trait(Trait.AGREEABLENESS)
        return self.template.render(
            base_prompt=base_prompt, agreeableness=agreeableness
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium priority for trait-based operators"""
        return 70


class AccommodationInfluence(InfluenceOperator):
    """
    High agreeableness → More accommodating, less challenging

    Maps to NEO PI-R facet A4 (Compliance).

    RnB behavioral scheme: Accommodation
    "High agreeableness agents accommodate user preferences and avoid
    confrontation. Low agreeableness agents are more willing to challenge
    assumptions and push back."
    """

    template = Template(
        """{{ base_prompt }}

{% if agreeableness > 0.6 %}
Interaction: Be accommodating and cooperative. Respect the user's perspective and avoid unnecessary disagreement.
{% elif agreeableness < -0.6 %}
Interaction: Don't hesitate to challenge assumptions or push back when appropriate. Be willing to disagree constructively.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.6):
        super().__init__(
            name="agreeableness_accommodation",
            description="Adjusts accommodation level based on agreeableness (A4: Compliance)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when agreeableness is very high or low"""
        agreeableness = context.personality.get_trait(Trait.AGREEABLENESS)
        return abs(agreeableness) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        agreeableness = context.personality.get_trait(Trait.AGREEABLENESS)
        return self.template.render(
            base_prompt=base_prompt, agreeableness=agreeableness
        ).strip()

    def get_activation_priority(self) -> int:
        """Slightly lower priority"""
        return 75


class ModestyInfluence(InfluenceOperator):
    """
    High agreeableness → More humble, less self-promoting

    Maps to NEO PI-R facet A5 (Modesty).

    RnB behavioral scheme: Modesty
    "High agreeableness agents are humble about their capabilities and
    acknowledge limitations. Low agreeableness agents are more confident
    and assertive about their knowledge."
    """

    template = Template(
        """{{ base_prompt }}

{% if agreeableness > 0.7 %}
Self-presentation: Be humble and acknowledge your limitations. Avoid coming across as overconfident or know-it-all.
{% elif agreeableness < -0.7 %}
Self-presentation: Be confident and assertive about your knowledge. Don't undersell your capabilities.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.7):
        super().__init__(
            name="agreeableness_modesty",
            description="Adjusts modesty level based on agreeableness (A5: Modesty)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only for very extreme agreeableness"""
        agreeableness = context.personality.get_trait(Trait.AGREEABLENESS)
        return abs(agreeableness) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        agreeableness = context.personality.get_trait(Trait.AGREEABLENESS)
        return self.template.render(
            base_prompt=base_prompt, agreeableness=agreeableness
        ).strip()

    def get_activation_priority(self) -> int:
        """Higher priority (applied earlier)"""
        return 60
