"""Feature extraction for risk-aware escalation model."""
import re

from src.evidence import ResolutionPattern, evidence_consistency
from src.intents import IntentResult

HIGH_RISK_INTENTS = {"payment_billing", "refund_return", "delivery_issue", "account_access"}

FRUSTRATION_PATTERNS = [
    r"\bworst\b", r"\bterrible\b", r"\bscam\b", r"\bunacceptable\b",
    r"\b!!!+", r"\bstill\s+waiting\b", r"\bthird\s+time\b",
    r"\bno\s+response\b", r"\bdisconnected\b",
]

LEGAL_KEYWORDS = ["lawyer", "sue", "legal", "fraud", "stolen", "police", "chargeback"]


def frustration_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for p in FRUSTRATION_PATTERNS if re.search(p, text_lower))
    caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return min(1.0, hits * 0.2 + caps * 1.5)


def multi_attempt_score(text: str) -> float:
    text_lower = text.lower()
    signals = ["again", "third time", "multiple", "still waiting", "disconnected", "escalat"]
    return min(1.0, sum(0.25 for s in signals if s in text_lower))


def legal_risk_score(text: str) -> float:
    text_lower = text.lower()
    return 1.0 if any(k in text_lower for k in LEGAL_KEYWORDS) else 0.0


def extract_escalation_features(
    message: str,
    intent: IntentResult,
    evidence: list[ResolutionPattern],
    verification_passed: bool,
    groundedness: float,
) -> dict[str, float]:
    top_score = evidence[0].retrieval_score if evidence else 0.0
    margin = 0.0
    if len(evidence) >= 2:
        margin = evidence[0].retrieval_score - evidence[1].retrieval_score

    return {
        "intent_confidence": intent.confidence,
        "retrieval_top1": top_score,
        "retrieval_margin": margin,
        "evidence_consistency": evidence_consistency(evidence),
        "groundedness": groundedness,
        "frustration": frustration_score(message),
        "multi_attempt": multi_attempt_score(message),
        "legal_risk": legal_risk_score(message),
        "high_risk_intent": 1.0 if intent.intent in HIGH_RISK_INTENTS else 0.0,
        "low_intent_confidence": 1.0 if intent.confidence < 0.45 else 0.0,
        "weak_evidence": 1.0 if top_score < 0.35 else 0.0,
        "verification_passed": 1.0 if verification_passed else 0.0,
        "praise_intent": 1.0 if intent.intent == "praise_thanks" else 0.0,
    }


FEATURE_NAMES = [
    "intent_confidence", "retrieval_top1", "retrieval_margin",
    "evidence_consistency", "groundedness", "frustration", "multi_attempt",
    "legal_risk", "high_risk_intent", "low_intent_confidence",
    "weak_evidence", "verification_passed", "praise_intent",
]
