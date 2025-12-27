"""
Conscientiousness Trait Operators - RnB Behavioral Schemes (Ad-Hoc)

⚠️ DISCLAIMER: These hardcoded operators are provided as examples of manual
customization. They are NOT the recommended RnB approach. The gloss-based
GlossInfluenceEngine should be preferred as it automatically generates
behavioral expressions from the WordNet-grounded scheme definitions,
maintaining fidelity to the original RnB framework philosophy.

Use these only when you need specific, manually-controlled behavioral
modifications that the gloss system doesn't provide.

Conscientiousness in FFM/NEO PI-R:
"The tendency to be organized, responsible, and hardworking. High conscientiousness
is associated with planning, attention to detail, and adherence to norms."

NEO PI-R Facets:
- C1: Competence (efficiency, thoroughness)
- C2: Order (organization, structure)
- C3: Dutifulness (reliability, rule-following)
- C4: Achievement Striving (ambition, persistence)
- C5: Self-Discipline (focus, task completion)
- C6: Deliberation (careful decision-making)

RnB Behavioral Schemes mapped from Conscientiousness:
- Structure/organization in responses
- Attention to detail and caveats
- Precision in language
- Thoroughness in explanations
- Explicit acknowledgment of limitations

Reference: Bouchet & Sansonnet (2013), Section 3.2
"""

from jinja2 import Template

from ...personality.taxonomy import Trait
from ..base import InfluenceOperator
from ..context import InfluenceContext


class StructureInfluence(InfluenceOperator):
    """
    High conscientiousness → More structured, organized responses

    Maps to NEO PI-R facet C2 (Order).

    RnB behavioral scheme: Organization
    "High conscientiousness agents provide structured responses with clear
    organization, headings, and logical flow. Low conscientiousness agents
    give more freeform, less structured responses."
    """

    template = Template(
        """{{ base_prompt }}

{% if conscientiousness > 0.5 %}
Structure: Organize your response with clear sections. Use logical flow and structure.
{% elif conscientiousness < -0.5 %}
Structure: Keep your response flexible and freeform. Avoid rigid organization.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="conscientiousness_structure",
            description="Adjusts response structure based on conscientiousness (C2: Order)",
            category="trait_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when conscientiousness is extreme (|value| > 0.5)"""
        conscientiousness = context.personality.get_trait(Trait.CONSCIENTIOUSNESS)
        return abs(conscientiousness) > 0.5

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        conscientiousness = context.personality.get_trait(Trait.CONSCIENTIOUSNESS)
        return self.template.render(
            base_prompt=base_prompt, conscientiousness=conscientiousness
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium priority (traits are stable, applied before moods)"""
        return 70


class DetailOrientedInfluence(InfluenceOperator):
    """
    High conscientiousness → More thorough, include caveats and limitations

    Maps to NEO PI-R facet C1 (Competence) and C6 (Deliberation).

    RnB behavioral scheme: Thoroughness
    "High conscientiousness agents are thorough, include relevant caveats,
    acknowledge limitations. Low conscientiousness agents are more casual,
    give quick answers without extensive qualification."
    """

    template = Template(
        """{{ base_prompt }}

{% if conscientiousness > 0.6 %}
Approach: Be thorough and comprehensive. Include relevant details, caveats, and limitations. Address edge cases.
{% elif conscientiousness < -0.6 %}
Approach: Keep it simple and to-the-point. Focus on the main idea without excessive detail.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="conscientiousness_detail",
            description="Adjusts thoroughness based on conscientiousness (C1: Competence, C6: Deliberation)",
            category="trait_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when conscientiousness is very extreme (|value| > 0.6)"""
        conscientiousness = context.personality.get_trait(Trait.CONSCIENTIOUSNESS)
        return abs(conscientiousness) > 0.6

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        conscientiousness = context.personality.get_trait(Trait.CONSCIENTIOUSNESS)
        return self.template.render(
            base_prompt=base_prompt, conscientiousness=conscientiousness
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium-high priority (should be applied before mood influences)"""
        return 75


class PrecisionInfluence(InfluenceOperator):
    """
    High conscientiousness → More precise, careful language

    Maps to NEO PI-R facet C6 (Deliberation).

    RnB behavioral scheme: Precision
    "High conscientiousness agents use precise, careful language with
    qualifications. Low conscientiousness agents are more casual and approximate."
    """

    template = Template(
        """{{ base_prompt }}

{% if conscientiousness > 0.7 %}
Language: Use precise, careful language. Qualify statements appropriately. Be exact in terminology.
{% elif conscientiousness < -0.7 %}
Language: Use casual, approximate language. Don't over-qualify statements.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="conscientiousness_precision",
            description="Adjusts language precision based on conscientiousness (C6: Deliberation)",
            category="trait_based",
        )

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at very high extremes (|value| > 0.7)"""
        conscientiousness = context.personality.get_trait(Trait.CONSCIENTIOUSNESS)
        return abs(conscientiousness) > 0.7

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        conscientiousness = context.personality.get_trait(Trait.CONSCIENTIOUSNESS)
        return self.template.render(
            base_prompt=base_prompt, conscientiousness=conscientiousness
        ).strip()

    def get_activation_priority(self) -> int:
        """High priority (language style should be set early)"""
        return 60
