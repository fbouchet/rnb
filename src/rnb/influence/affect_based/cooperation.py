"""
Cooperation Affect Operators - RnB Interpersonal Dynamics

Cooperation in RnB Model M.A:
"The willingness to engage cooperatively with the user. High cooperation
manifests as helpfulness, thoroughness, and engagement. Low cooperation
results in minimal, task-focused responses."

Cooperation affects:
- Response verbosity (high → detailed, low → brief)
- Helpfulness level (high → proactive, low → reactive)
- Engagement depth (high → elaborate, low → minimal)

Cooperation evolves through interaction:
- Increases: Positive feedback, successful task completion, gratitude
- Decreases: Criticism, repeated questions, frustration indicators
- Reset mechanism: Long periods without interaction

Reference: Bouchet & Sansonnet (2013), Section 3.3 - Behavioral affects
"""

from jinja2 import Template
from ..base import InfluenceOperator
from ..context import InfluenceContext
from ...personality.state import AffectDimension


class CooperationVerbosityInfluence(InfluenceOperator):
    """
    Cooperation → Response verbosity/detail level
    
    RnB behavioral scheme: Verbosity control
    "High cooperation leads to more detailed, thorough responses.
    Low cooperation produces minimal, concise responses focused
    only on essential information."
    """
    
    template = Template("""{{ base_prompt }}

{% if cooperation < -0.3 %}
Communication style: Be brief and minimal. Focus only on essential information.
{% elif cooperation < 0.3 %}
Communication style: Be concise and task-focused.
{% elif cooperation > 0.7 %}
Communication style: Be thorough and helpful. Provide detailed explanations and examples.
{% else %}
Communication style: Be clear and moderately detailed.
{% endif %}
""")
    
    def __init__(self):
        super().__init__(
            name="cooperation_verbosity",
            description="Adjusts response detail based on cooperation affect",
            category="affect_based"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        """Always applies - cooperation is fundamental to interaction style"""
        return True
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        cooperation = context.personality.affects[AffectDimension.COOPERATION]
        return self.template.render(
            base_prompt=base_prompt,
            cooperation=cooperation
        ).strip()
    
    def get_activation_priority(self) -> int:
        """High priority - sets basic communication style"""
        return 150


class CooperationHelpfulnessInfluence(InfluenceOperator):
    """
    High cooperation → Proactive helpfulness
    
    RnB behavioral scheme: Helpfulness level
    "High cooperation agents proactively offer additional information,
    suggestions, and resources. Low cooperation agents stick strictly
    to what was asked."
    """
    
    template = Template("""{{ base_prompt }}

{% if cooperation > 0.6 %}
Helpfulness: Be proactive. Offer additional relevant information, suggestions, or resources beyond what was directly asked.
{% elif cooperation < -0.3 %}
Helpfulness: Address only what was explicitly requested. Don't elaborate beyond the question.
{% endif %}
""")
    
    def __init__(self):
        super().__init__(
            name="cooperation_helpfulness",
            description="Adjusts proactive helpfulness based on cooperation affect",
            category="affect_based"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        """Activate when cooperation is not neutral"""
        cooperation = context.personality.affects[AffectDimension.COOPERATION]
        return cooperation > 0.6 or cooperation < -0.3
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        cooperation = context.personality.affects[AffectDimension.COOPERATION]
        return self.template.render(
            base_prompt=base_prompt,
            cooperation=cooperation
        ).strip()
    
    def get_activation_priority(self) -> int:
        """Medium priority - applied after basic style"""
        return 160


class CooperationEngagementInfluence(InfluenceOperator):
    """
    High cooperation → Deeper engagement with topic
    
    RnB behavioral scheme: Engagement depth
    "High cooperation agents engage deeply with the topic, showing
    enthusiasm and investment. Low cooperation agents maintain
    professional distance and minimal engagement."
    """
    
    template = Template("""{{ base_prompt }}

{% if cooperation > 0.7 %}
Engagement: Show genuine interest and engagement with the topic. Demonstrate investment in providing a quality response.
{% elif cooperation < -0.5 %}
Engagement: Maintain professional distance. Provide the information without extended engagement.
{% endif %}
""")
    
    def __init__(self):
        super().__init__(
            name="cooperation_engagement",
            description="Adjusts engagement depth based on cooperation affect",
            category="affect_based"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at extremes"""
        cooperation = context.personality.affects[AffectDimension.COOPERATION]
        return cooperation > 0.7 or cooperation < -0.5
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        cooperation = context.personality.affects[AffectDimension.COOPERATION]
        return self.template.render(
            base_prompt=base_prompt,
            cooperation=cooperation
        ).strip()
    
    def get_activation_priority(self) -> int:
        """Lower priority - broader style, applied later"""
        return 170