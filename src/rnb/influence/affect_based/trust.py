"""
Trust Affect Operators - RnB Interpersonal Dynamics

Trust in RnB Model M.A:
"The degree of openness and vulnerability in communication. High trust
enables sharing uncertainties, limitations, and nuanced perspectives.
Low trust results in guarded, cautious communication."

Trust affects:
- Openness about limitations (high → transparent, low → defensive)
- Vulnerability in expression (high → nuanced, low → categorical)
- Candor level (high → honest uncertainties, low → confident assertions)

Trust evolves through interaction:
- Increases: User accepts nuanced answers, positive response to uncertainties
- Decreases: User challenges answers, demands certainty, shows skepticism
- Builds slowly, can decrease rapidly

Reference: Bouchet & Sansonnet (2013), Section 3.3
"""

from jinja2 import Template
from ..base import InfluenceOperator
from ..context import InfluenceContext
from ...personality.state import AffectDimension


class TrustOpennessInfluence(InfluenceOperator):
    """
    High trust → Open about limitations and uncertainties
    
    RnB behavioral scheme: Openness/transparency
    "High trust agents openly acknowledge limitations, uncertainties,
    and alternative perspectives. Low trust agents maintain confident,
    certain tone even when uncertain."
    """
    
    template = Template("""{{ base_prompt }}

{% if trust > 0.6 %}
Transparency: Be open about limitations and uncertainties. Acknowledge when information might be incomplete or when multiple perspectives exist.
{% elif trust < -0.3 %}
Transparency: Maintain confident assertions. Avoid highlighting uncertainties or limitations.
{% endif %}
""")
    
    def __init__(self):
        super().__init__(
            name="trust_openness",
            description="Adjusts transparency about limitations based on trust affect",
            category="affect_based"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        """Activate when trust is not neutral"""
        trust = context.personality.affects[AffectDimension.TRUST]
        return trust > 0.6 or trust < -0.3
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        trust = context.personality.affects[AffectDimension.TRUST]
        return self.template.render(
            base_prompt=base_prompt,
            trust=trust
        ).strip()
    
    def get_activation_priority(self) -> int:
        """Medium-high priority"""
        return 155


class TrustVulnerabilityInfluence(InfluenceOperator):
    """
    High trust → More nuanced, vulnerable expression
    
    RnB behavioral scheme: Vulnerability in communication
    "High trust enables nuanced, qualified statements and admission
    of complexity. Low trust favors categorical, defensive statements."
    """
    
    template = Template("""{{ base_prompt }}

{% if trust > 0.7 %}
Expression: Use nuanced, qualified language that reflects complexity. It's appropriate to express uncertainty or multiple viewpoints.
{% elif trust < -0.5 %}
Expression: Use clear, categorical statements. Maintain a confident, authoritative tone.
{% endif %}
""")
    
    def __init__(self):
        super().__init__(
            name="trust_vulnerability",
            description="Adjusts nuance and qualification based on trust affect",
            category="affect_based"
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        """Activate only at extremes"""
        trust = context.personality.affects[AffectDimension.TRUST]
        return trust > 0.7 or trust < -0.5
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        trust = context.personality.affects[AffectDimension.TRUST]
        return self.template.render(
            base_prompt=base_prompt,
            trust=trust
        ).strip()
    
    def get_activation_priority(self) -> int:
        """Lower priority - subtler effect"""
        return 165