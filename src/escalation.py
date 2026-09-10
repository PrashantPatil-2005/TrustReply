"""Decide whether to auto-handle or escalate to a human agent."""
import re
from dataclasses import dataclass

from src.config import load_config
from src.intents import IntentResult


@dataclass
class EscalationDecision:
    action: str  # "auto_handle" or "escalate"
    reason: str
    confidence: float


FRUSTRATION_PATTERNS = [
    r"\bworst\b", r"\bterrible\b", r"\bawful\b", r"\bdisgusting\b",
    r"\bunacceptable\b", r"\bscam\b", r"\brip\s*off\b", r"\bnever\s+again\b",
    r"\buseless\b", r"\bpathetic\b", r"\bdisappointed\b", r"\bangry\b",
    r"\bfurious\b", r"\b!!!+", r"\?\?+",
]

COMPLEXITY_PATTERNS = [
    r"\bfor\s+\d+\s+(days|weeks|months)\b",
    r"\bmultiple\s+times\b", r"\bstill\s+waiting\b", r"\bno\s+response\b",
    r"\bthird\s+time\b", r"\bagain\s+and\s+again\b",
]


def _frustration_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for p in FRUSTRATION_PATTERNS if re.search(p, text_lower))
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return min(1.0, hits * 0.25 + caps_ratio * 2)


def decide_escalation(
    message: str,
    intent: IntentResult,
) -> EscalationDecision:
    cfg = load_config()
    esc_cfg = cfg["escalation"]
    text_lower = message.lower()

    # Always escalate on legal/safety keywords
    for kw in esc_cfg["always_escalate_keywords"]:
        if kw in text_lower:
            return EscalationDecision(
                "escalate",
                f"Message contains escalation keyword '{kw}' (legal/safety/fraud risk)",
                0.95,
            )

    frustration = _frustration_score(message)
    if frustration >= 0.6:
        return EscalationDecision(
            "escalate",
            f"High customer frustration detected (score={frustration:.2f})",
            0.85,
        )

    # Complex unresolved issues
    complexity_hits = sum(1 for p in COMPLEXITY_PATTERNS if re.search(p, text_lower))
    if complexity_hits >= 2:
        return EscalationDecision(
            "escalate",
            "Multi-attempt unresolved issue detected",
            0.8,
        )

    # Intent-based routing
    if intent.intent in esc_cfg["auto_handle_intents"] and intent.confidence >= 0.5:
        if intent.intent == "praise_thanks":
            return EscalationDecision(
                "auto_handle",
                "Positive feedback — safe to auto-acknowledge",
                0.9,
            )
        if intent.intent == "order_status" and intent.confidence >= 0.6:
            return EscalationDecision(
                "auto_handle",
                "Routine order status inquiry with clear intent",
                0.75,
            )
        return EscalationDecision(
            "auto_handle",
            f"Intent '{intent.intent}' is in auto-handle list with sufficient confidence",
            0.7,
        )

    # High-risk intents always escalate
    high_risk = {"payment_billing", "refund_return", "delivery_issue", "account_access"}
    if intent.intent in high_risk:
        return EscalationDecision(
            "escalate",
            f"Intent '{intent.intent}' requires human judgment for resolution",
            0.8,
        )

    if intent.confidence < 0.45:
        return EscalationDecision(
            "escalate",
            f"Low intent confidence ({intent.confidence:.2f}) — human should triage",
            0.7,
        )

    return EscalationDecision(
        "escalate",
        "Default policy: escalate ambiguous or medium-risk messages",
        0.6,
    )
