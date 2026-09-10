"""Resolution Pattern Objects (RPOs) — structured historical evidence."""
import re
from dataclasses import dataclass, asdict, field


ALLOWED_ACTIONS = {
    "request_dm",
    "verify_tracking",
    "investigate",
    "apologize",
    "provide_link",
    "ask_details",
    "acknowledge_positive",
    "escalate_internal",
}

CONSTRAINTS = {
    "no_refund_promise",
    "no_delivery_date_promise",
    "no_password_request",
}


@dataclass
class ResolutionPattern:
    evidence_id: str
    customer_problem: str
    brand_reply: str
    actions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_actions_from_reply(reply: str) -> list[str]:
    """Rule-based extraction of resolution actions from a brand reply."""
    text = reply.lower()
    actions: list[str] = []

    if re.search(r"\b(dm|direct message|message us|privat)\b", text):
        actions.append("request_dm")
    if re.search(r"\b(track|tracking|shipment|shipping status)\b", text):
        actions.append("verify_tracking")
    if re.search(r"\b(look into|investigate|review your)\b", text):
        actions.append("investigate")
    if re.search(r"\b(sorry|apolog|regret|understand your)\b", text):
        actions.append("apologize")
    if re.search(r"https?://|t\.co/", text):
        actions.append("provide_link")
    if "?" in reply:
        actions.append("ask_details")
    if re.search(r"\b(thank|glad|appreciate|great)\b", text):
        actions.append("acknowledge_positive")

    if not actions:
        actions.append("ask_details")

    return [a for a in actions if a in ALLOWED_ACTIONS]


def extract_constraints_from_reply(reply: str) -> list[str]:
    text = reply.lower()
    constraints = []
    if not re.search(r"\bwill refund\b|\bguarantee.*refund\b", text):
        constraints.append("no_refund_promise")
    if not re.search(r"\barrive by\b|\bwill deliver on\b", text):
        constraints.append("no_delivery_date_promise")
    if not re.search(r"\bpassword\b|\bssn\b|\bfull card\b", text):
        constraints.append("no_password_request")
    return constraints


def build_rpo(evidence_id: str, customer_msg: str, brand_reply: str, score: float = 0.0) -> ResolutionPattern:
    return ResolutionPattern(
        evidence_id=evidence_id,
        customer_problem=customer_msg,
        brand_reply=brand_reply,
        actions=extract_actions_from_reply(brand_reply),
        constraints=extract_constraints_from_reply(brand_reply),
        retrieval_score=score,
    )


def evidence_consistency(rpos: list[ResolutionPattern]) -> float:
    """Measure agreement among top retrieved resolution actions."""
    if len(rpos) < 2:
        return 1.0 if rpos else 0.0

    action_sets = [set(r.actions) for r in rpos[:3]]
    if not action_sets:
        return 0.0

    union = set.union(*action_sets)
    if not union:
        return 0.0

    overlaps = []
    for i in range(len(action_sets)):
        for j in range(i + 1, len(action_sets)):
            inter = len(action_sets[i] & action_sets[j])
            overlaps.append(inter / max(len(union), 1))

    return sum(overlaps) / len(overlaps) if overlaps else 1.0


def select_evidence(rpos: list[ResolutionPattern], min_score: float = 0.25, max_items: int = 3) -> list[ResolutionPattern]:
    """Select evidence passing minimum retrieval threshold."""
    selected = [r for r in rpos if r.retrieval_score >= min_score]
    return selected[:max_items]
