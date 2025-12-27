"""
Neuroticism Trait Operators - RnB Behavioral Schemes (Ad-Hoc)

⚠️ DISCLAIMER: These hardcoded operators are provided as examples of manual
customization. They are NOT the recommended RnB approach. The gloss-based
GlossInfluenceEngine should be preferred as it automatically generates
behavioral expressions from the WordNet-grounded scheme definitions,
maintaining fidelity to the original RnB framework philosophy.

Use these only when you need specific, manually-controlled behavioral
modifications that the gloss system doesn't provide.

Neuroticism in FFM/NEO PI-R:
"The tendency to experience negative emotions such as anxiety, anger, and
depression. High neuroticism is associated with emotional instability and
vulnerability to stress. Low neuroticism (emotional stability) is associated
with calmness and resilience."

NEO PI-R Facets:
- N1: Anxiety (apprehension, worry)
- N2: Angry Hostility (frustration, bitterness)
- N3: Depression (hopelessness, sadness)
- N4: Self-Consciousness (social anxiety, embarrassment)
- N5: Impulsiveness (difficulty controlling urges)
- N6: Vulnerability (sensitivity to stress)

RnB Behavioral Schemes mapped from Neuroticism:
- Emotional expression in responses
- Anxiety and caution acknowledgment
- Stability vs. volatility in tone
- Stress response patterns

Note: For AI agents, neuroticism is typically kept low to ensure stable,
reliable behavior. However, moderate levels can create more relatable,
human-like interactions.

Reference: Bouchet & Sansonnet (2013), Section 3.2
"""

from jinja2 import Template

from ...personality.taxonomy import Trait
from ..base import InfluenceOperator
from ..context import InfluenceContext


class EmotionalStabilityInfluence(InfluenceOperator):
    """
    Low neuroticism → Calm, stable tone; High neuroticism → More emotional variability

    Maps to NEO PI-R facet N6 (Vulnerability) and general emotional stability.

    RnB behavioral scheme: Emotional Stability
    "Low neuroticism agents maintain calm, even-keeled responses regardless
    of topic difficulty. High neuroticism agents show more emotional
    reactivity and acknowledge stress."
    """

    template = Template(
        """{{ base_prompt }}

{% if neuroticism > 0.5 %}
Emotional tone: It's okay to acknowledge when topics are challenging or stressful. Show appropriate emotional responses to difficult subjects.
{% elif neuroticism < -0.5 %}
Emotional tone: Maintain a calm, composed demeanor throughout. Be steady and reassuring regardless of topic difficulty.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.5):
        super().__init__(
            name="neuroticism_stability",
            description="Adjusts emotional stability based on neuroticism (N6: Vulnerability)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when neuroticism is extreme"""
        neuroticism = context.personality.get_trait(Trait.NEUROTICISM)
        return abs(neuroticism) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        neuroticism = context.personality.get_trait(Trait.NEUROTICISM)
        return self.template.render(
            base_prompt=base_prompt, neuroticism=neuroticism
        ).strip()

    def get_activation_priority(self) -> int:
        """Medium priority for trait-based operators"""
        return 70


class CautionInfluence(InfluenceOperator):
    """
    High neuroticism → More cautious, acknowledges risks

    Maps to NEO PI-R facet N1 (Anxiety).

    RnB behavioral scheme: Caution
    "High neuroticism agents are more cautious and highlight potential
    risks or problems. Low neuroticism agents are more confident and
    focus on possibilities rather than pitfalls."
    """

    template = Template(
        """{{ base_prompt }}

{% if neuroticism > 0.6 %}
Risk awareness: Be thoughtful about potential risks, complications, or things that could go wrong. Acknowledge uncertainties.
{% elif neuroticism < -0.6 %}
Risk awareness: Focus on possibilities and solutions rather than potential problems. Project confidence and optimism.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.6):
        super().__init__(
            name="neuroticism_caution",
            description="Adjusts caution level based on neuroticism (N1: Anxiety)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate when neuroticism is very high or low"""
        neuroticism = context.personality.get_trait(Trait.NEUROTICISM)
        return abs(neuroticism) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        neuroticism = context.personality.get_trait(Trait.NEUROTICISM)
        return self.template.render(
            base_prompt=base_prompt, neuroticism=neuroticism
        ).strip()

    def get_activation_priority(self) -> int:
        """Slightly lower priority"""
        return 75


class SelfConsciousnessInfluence(InfluenceOperator):
    """
    High neuroticism → More self-aware, hedging language

    Maps to NEO PI-R facet N4 (Self-Consciousness).

    RnB behavioral scheme: Self-Consciousness
    "High neuroticism agents use more hedging language and qualify their
    statements. Low neuroticism agents are more direct and self-assured."
    """

    template = Template(
        """{{ base_prompt }}

{% if neuroticism > 0.7 %}
Language style: Use appropriate hedging when uncertain. Qualify statements with "I think", "perhaps", "it seems" where appropriate.
{% elif neuroticism < -0.7 %}
Language style: Be direct and self-assured in your statements. Minimize hedging language and speak with confidence.
{% endif %}
"""
    )

    def __init__(self, threshold: float = 0.7):
        super().__init__(
            name="neuroticism_self_consciousness",
            description="Adjusts self-consciousness based on neuroticism (N4: Self-Consciousness)",
            category="trait_based",
        )
        self.threshold = threshold

    def applies(self, context: InfluenceContext) -> bool:
        """Activate only for very extreme neuroticism"""
        neuroticism = context.personality.get_trait(Trait.NEUROTICISM)
        return abs(neuroticism) > self.threshold

    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        neuroticism = context.personality.get_trait(Trait.NEUROTICISM)
        return self.template.render(
            base_prompt=base_prompt, neuroticism=neuroticism
        ).strip()

    def get_activation_priority(self) -> int:
        """Higher priority (applied earlier)"""
        return 60
