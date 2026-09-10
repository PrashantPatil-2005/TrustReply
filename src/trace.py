"""Auditable decision trace for ThreadVault agent."""
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentTrace:
    customer_message: str
    thread_context: list[str] = field(default_factory=list)

    # Intent
    intent: str = ""
    intent_confidence: float = 0.0
    intent_method: str = "tfidf_lr"

    # Retrieval
    retrieval_hits: list[dict] = field(default_factory=list)
    retrieval_top_score: float = 0.0
    retrieval_margin: float = 0.0

    # Evidence
    selected_evidence: list[dict] = field(default_factory=list)
    evidence_consistency: float = 0.0
    evidence_gate_passed: bool = False

    # Generation
    draft_reply: str = ""
    cited_evidence_ids: list[str] = field(default_factory=list)
    actions_used: list[str] = field(default_factory=list)

    # Verification
    verification: dict = field(default_factory=dict)
    groundedness_score: float = 0.0

    # Escalation
    escalation_action: str = ""
    escalation_reason: str = ""
    risk_score: float = 0.0
    escalation_features: dict = field(default_factory=dict)

    # Trust
    strict_trust_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
