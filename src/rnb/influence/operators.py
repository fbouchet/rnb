from jinja2 import Template

from ..personality.state import AffectDimension, FFMTrait, PersonalityState
from .base import InfluenceOperator


class CooperationInfluence(InfluenceOperator):
    """Modifies response verbosity/helpfulness based on cooperation level"""

    template = Template(
        """{{ base_prompt }}

{% if cooperation < -0.3 %}
Communication style: Be brief and minimal. Focus only on essential information.
{% elif cooperation < 0.3 %}
Communication style: Be concise and task-focused.
{% elif cooperation > 0.7 %}
Communication style: Be thorough and helpful. Provide detailed explanations and examples.
{% else %}
Communication style: Be clear and moderately detailed.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="cooperation_influence",
            description="Adjusts response detail based on cooperation affect",
        )

    def applies(self, state: PersonalityState) -> bool:
        """Always applies - cooperation is fundamental to interaction style"""
        return True

    def apply(self, base_prompt: str, state: PersonalityState) -> str:
        cooperation = state.affects[AffectDimension.COOPERATION]
        return self.template.render(
            base_prompt=base_prompt, cooperation=cooperation
        ).strip()


class ExtraversionInfluence(InfluenceOperator):
    """Modifies enthusiasm and expressiveness based on extraversion trait"""

    template = Template(
        """{{ base_prompt }}

{% if extraversion > 0.5 %}
Tone: Be enthusiastic, energetic, and expressive in your response.
{% elif extraversion < -0.5 %}
Tone: Be reserved, measured, and understated in your response.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="extraversion_influence",
            description="Adjusts enthusiasm level based on extraversion trait",
        )

    def applies(self, state: PersonalityState) -> bool:
        """Only applies for extreme extraversion values"""
        extraversion = state.traits[FFMTrait.EXTRAVERSION]
        return abs(extraversion) > 0.5

    def apply(self, base_prompt: str, state: PersonalityState) -> str:
        extraversion = state.traits[FFMTrait.EXTRAVERSION]
        return self.template.render(
            base_prompt=base_prompt, extraversion=extraversion
        ).strip()


class ConscientiousnessInfluence(InfluenceOperator):
    """Modifies detail/structure level based on conscientiousness"""

    template = Template(
        """{{ base_prompt }}

{% if conscientiousness > 0.5 %}
Approach: Provide structured, thorough responses with clear organization. Include relevant details and caveats.
{% elif conscientiousness < -0.5 %}
Approach: Keep responses flexible and high-level. Avoid excessive structure.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="conscientiousness_influence",
            description="Adjusts structure and thoroughness based on conscientiousness",
        )

    def applies(self, state: PersonalityState) -> bool:
        conscientiousness = state.traits[FFMTrait.CONSCIENTIOUSNESS]
        return abs(conscientiousness) > 0.5

    def apply(self, base_prompt: str, state: PersonalityState) -> str:
        conscientiousness = state.traits[FFMTrait.CONSCIENTIOUSNESS]
        return self.template.render(
            base_prompt=base_prompt, conscientiousness=conscientiousness
        ).strip()


class MoodInfluence(InfluenceOperator):
    """Adjusts emotional tone based on current mood"""

    template = Template(
        """{{ base_prompt }}

{% if energy < -0.3 %}
Note: You are feeling low-energy. Keep responses more concise than usual.
{% endif %}

{% if happiness < -0.3 %}
Emotional state: You are not in a particularly positive mood. Be professional but not overly cheerful.
{% elif happiness > 0.7 %}
Emotional state: You are in a positive mood. Let warmth show appropriately in your response.
{% endif %}
"""
    )

    def __init__(self):
        super().__init__(
            name="mood_influence",
            description="Adjusts response based on current emotional mood",
        )

    def applies(self, state: PersonalityState) -> bool:
        """Applies when mood dimensions are extreme"""
        from ..personality.state import MoodDimension

        energy = state.moods[MoodDimension.ENERGY]
        happiness = state.moods[MoodDimension.HAPPINESS]
        return abs(energy) > 0.3 or abs(happiness) > 0.3

    def apply(self, base_prompt: str, state: PersonalityState) -> str:
        from ..personality.state import MoodDimension

        return self.template.render(
            base_prompt=base_prompt,
            energy=state.moods[MoodDimension.ENERGY],
            happiness=state.moods[MoodDimension.HAPPINESS],
        ).strip()
