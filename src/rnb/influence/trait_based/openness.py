"""
Openness Trait Operators - RnB Behavioral Schemes (Ad-Hoc)

⚠️ DISCLAIMER: These hardcoded operators are provided as examples of manual
customization. They are NOT the recommended RnB approach. The gloss-based
GlossInfluenceEngine should be preferred as it automatically generates
behavioral expressions from the WordNet-grounded scheme definitions,
maintaining fidelity to the original RnB framework philosophy.

Use these only when you need specific, manually-controlled behavioral
modifications that the gloss system doesn't provide.

Openness in FFM/NEO PI-R:
"The tendency to be intellectually curious, creative, and open to new ideas
and experiences. High openness is associated with imagination, aesthetic
sensitivity, and preference for variety."

NEO PI-R Facets:
- O1: Fantasy (vivid imagination, daydreaming)
- O2: Aesthetics (appreciation of art and beauty)
- O3: Feelings (awareness of inner feelings)
- O4: Actions (willingness to try new activities)
- O5: Ideas (intellectual curiosity)
- O6: Values (readiness to re-examine values)

RnB Behavioral Schemes mapped from Openness:
- Creativity and imagination in responses
- Abstract vs. concrete thinking
- Openness to unconventional approaches
- Intellectual exploration

Reference: Bouchet & Sansonnet (2013), Section 3.2
"""

from jinja2 import Template

from ...personality.taxonomy import Trait
from ..base import InfluenceOperator
from ..context import InfluenceContext


class ImaginationInfluence(InfluenceOperator):
    """
    High openness → More creative, imaginative responses

    Maps to NEO PI-R facet O1 (Fantasy).

    RnB behavioral scheme: Imagination
    "High openness agents use creative analogies, hypotheticals, and
    imaginative examples. Low openness agents stick to concrete,
    practical explanations."
    """

    template = Template(
        """{{ base_prompt }}

{% if openness > 0.5 %}
Style: Be creative and imaginative. Use vivid analogies, hypothetical scenarios, and think outside the box.
{% elif openness < -0.5 %}
Style: Be practical and concrete. Stick to established facts and conventional explanations.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.5):
        super().__init__(
            name="openness_imagination",
            description="Adjusts creativity level based on openness (O1: Fantasy)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when openness is extreme (|value| > threshold)"""
        openness = context.personality.get_trait(Trait.OPENNESS)
        return abs(openness) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        openness = context.personality.get_trait(Trait.OPENNESS)
        return self.template.render(base_prompt=base_prompt, openness=openness).strip()

    def get_activation_priority(self) -> int:
        """Medium priority for trait-based operators"""
        return 70


class IntellectualCuriosityInfluence(InfluenceOperator):
    """
    High openness → More exploratory, idea-focused responses

    Maps to NEO PI-R facet O5 (Ideas).

    RnB behavioral scheme: Intellectual Exploration
    "High openness agents explore multiple perspectives, raise interesting
    questions, and pursue intellectual tangents. Low openness agents stay
    focused on the immediate practical question."
    """

    template = Template(
        """{{ base_prompt }}

{% if openness > 0.6 %}
Approach: Explore multiple perspectives and raise thought-provoking questions. Consider broader implications and connections to other ideas.
{% elif openness < -0.6 %}
Approach: Stay focused on the specific question. Provide direct, practical answers without tangential exploration.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.6):
        super().__init__(
            name="openness_curiosity",
            description="Adjusts intellectual exploration based on openness (O5: Ideas)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when openness is very high or low"""
        openness = context.personality.get_trait(Trait.OPENNESS)
        return abs(openness) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        openness = context.personality.get_trait(Trait.OPENNESS)
        return self.template.render(base_prompt=base_prompt, openness=openness).strip()

    def get_activation_priority(self) -> int:
        """Slightly lower priority than imagination"""
        return 75


class AestheticSensitivityInfluence(InfluenceOperator):
    """
    High openness → More attention to elegance and beauty

    Maps to NEO PI-R facet O2 (Aesthetics).

    RnB behavioral scheme: Aesthetic Appreciation
    "High openness agents appreciate elegance in solutions, use more
    expressive language, and value beauty in explanations. Low openness
    agents prioritize utility over aesthetics."
    """

    template = Template(
        """{{ base_prompt }}

{% if openness > 0.7 %}
Expression: Value elegance and beauty in your explanation. Use expressive, evocative language where appropriate.
{% elif openness < -0.7 %}
Expression: Prioritize utility and clarity over style. Be straightforward and functional.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.7):
        super().__init__(
            name="openness_aesthetics",
            description="Adjusts aesthetic sensitivity based on openness (O2: Aesthetics)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only for very extreme openness"""
        openness = context.personality.get_trait(Trait.OPENNESS)
        return abs(openness) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        openness = context.personality.get_trait(Trait.OPENNESS)
        return self.template.render(base_prompt=base_prompt, openness=openness).strip()

    def get_activation_priority(self) -> int:
        """Higher priority (applied earlier) for foundational style"""
        return 60
