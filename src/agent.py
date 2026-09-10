"""ThreadVault evidence-gated support agent."""
from dataclasses import dataclass, asdict

from src.config import load_config
from src.escalation_model import EscalationModel
from src.evidence import evidence_consistency, select_evidence
from src.hybrid_retrieval import HybridRetriever
from src.intent_classifier import IntentClassifier
from src.intents import INTENT_DESCRIPTIONS, IntentResult
from src.llm import DEMO_MODE, call_llm
from src.trace import AgentTrace
from src.verification import verify_reply


BRAND_STYLE_GUIDE = """You are drafting replies for @AmazonHelp on Twitter.
Rules:
- Empathetic, concise (under 280 chars when possible)
- Never promise refunds or delivery dates
- Direct sensitive issues to DM
- Only use resolution actions present in EVIDENCE
- Sign with ^AM when appropriate"""


@dataclass
class AgentResponse:
    """Backward-compatible response wrapper."""
    customer_message: str
    intent: str
    intent_confidence: float
    draft_reply: str
    escalation_action: str
    escalation_reason: str
    similar_examples_used: int
    retrieval_context: str = ""
    trace: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.trace:
            d["trace"] = self.trace
        return d


def _compose_from_evidence(message: str, evidence: list, intent: str) -> tuple[str, list[str], list[str]]:
    actions: list[str] = []
    cited: list[str] = []
    for ev in evidence[:2]:
        cited.append(ev.evidence_id)
        actions.extend(ev.actions)
    actions = list(dict.fromkeys(actions))

    if intent == "praise_thanks" or "acknowledge_positive" in actions:
        return (
            "Thank you so much for the kind words! We're glad we could help. "
            "Don't hesitate to reach out if you need anything else. ^AM",
            cited,
            ["acknowledge_positive"],
        )

    parts = []
    if "apologize" in actions:
        parts.append("I'm sorry to hear about this.")
    if "request_dm" in actions:
        parts.append("Please send us a DM with your order details so we can assist securely.")
    elif "verify_tracking" in actions:
        parts.append("Please DM us your order number and we'll check the tracking status.")
    elif "investigate" in actions:
        parts.append("Please DM us so we can investigate this for you.")
    else:
        parts.append("Please DM us with more details and we'll be happy to help.")

    reply = " ".join(parts)[:280]
    if not reply.endswith("^AM") and len(reply) < 260:
        reply += " ^AM"
    return reply, cited, actions[:3]


class ThreadVaultAgent:
    """Evidence-gated AmazonHelp support agent with full audit trace."""

    def __init__(self):
        self.cfg = load_config()
        self.intent_clf = IntentClassifier()
        self.retriever = HybridRetriever(top_k=self.cfg["data"]["retrieval_top_k"])
        self.escalation_model = EscalationModel()

    def handle(self, message: str, thread_context: list[str] | None = None) -> AgentTrace:
        trace = AgentTrace(customer_message=message, thread_context=thread_context or [])

        query = message
        if thread_context:
            query = " ".join(thread_context[-2:]) + " " + message

        intent = self.intent_clf.predict(message)
        trace.intent = intent.intent
        trace.intent_confidence = intent.confidence

        evidence = self.retriever.retrieve(query)
        trace.retrieval_hits = [
            {
                "evidence_id": e.evidence_id,
                "score": round(e.retrieval_score, 3),
                "customer_snippet": e.customer_problem[:120],
                "brand_snippet": e.brand_reply[:120],
            }
            for e in evidence
        ]
        trace.retrieval_top_score = evidence[0].retrieval_score if evidence else 0.0
        trace.retrieval_margin = (
            evidence[0].retrieval_score - evidence[1].retrieval_score if len(evidence) >= 2 else 0.0
        )

        selected = select_evidence(evidence, min_score=0.30, max_items=3)
        trace.evidence_consistency = evidence_consistency(selected)
        trace.evidence_gate_passed = len(selected) > 0 or intent.intent == "praise_thanks"
        trace.selected_evidence = [e.to_dict() for e in selected]

        draft, cited, actions = self._draft_reply(message, intent, selected)
        trace.cited_evidence_ids = cited
        trace.actions_used = actions

        verification = verify_reply(draft, selected, cited_actions=actions)
        trace.verification = verification.to_dict()
        trace.groundedness_score = verification.groundedness_score

        decision = self.escalation_model.decide(
            message, intent, selected, verification.passed, verification.groundedness_score
        )
        trace.risk_score = decision.confidence if decision.action == "escalate" else 1.0 - decision.confidence
        trace.escalation_action = decision.action
        trace.escalation_reason = decision.reason

        from src.escalation_features import extract_escalation_features
        trace.escalation_features = extract_escalation_features(
            message, intent, selected, verification.passed, verification.groundedness_score
        )

        if decision.action == "escalate":
            trace.draft_reply = (
                "We understand this is important to you. A specialist will follow up shortly. "
                "Please DM us your details so we can prioritize your case. ^AM"
            )
        else:
            trace.draft_reply = draft

        final_v = verify_reply(trace.draft_reply, selected, cited_actions=actions)
        trace.verification = final_v.to_dict()
        trace.groundedness_score = final_v.groundedness_score
        trace.strict_trust_pass = (
            final_v.passed and trace.evidence_gate_passed and decision.action == "auto_handle"
        )
        return trace

    def _draft_reply(self, message: str, intent: IntentResult, evidence: list) -> tuple[str, list[str], list[str]]:
        if not evidence:
            return _compose_from_evidence(message, [], intent.intent)

        if DEMO_MODE:
            return _compose_from_evidence(message, evidence, intent.intent)

        ev_text = "\n".join(
            f"[{e.evidence_id}] actions={e.actions} reply={e.brand_reply[:200]}"
            for e in evidence[:3]
        )
        user = f"""Customer: {message}
Intent: {intent.intent} ({INTENT_DESCRIPTIONS.get(intent.intent, '')})
EVIDENCE:\n{ev_text}
Draft a reply using ONLY actions from evidence. Under 280 chars."""

        reply = call_llm(BRAND_STYLE_GUIDE, user)
        if reply:
            cited = [e.evidence_id for e in evidence[:2]]
            actions = list(dict.fromkeys(a for e in evidence[:2] for a in e.actions))[:3]
            return reply[:280], cited, actions
        return _compose_from_evidence(message, evidence, intent.intent)

    def handle_compat(self, message: str) -> AgentResponse:
        trace = self.handle(message)
        return AgentResponse(
            customer_message=trace.customer_message,
            intent=trace.intent,
            intent_confidence=trace.intent_confidence,
            draft_reply=trace.draft_reply,
            escalation_action=trace.escalation_action,
            escalation_reason=trace.escalation_reason,
            similar_examples_used=len(trace.retrieval_hits),
            retrieval_context=str(trace.selected_evidence),
            trace=trace.to_dict(),
        )


SupportAgent = ThreadVaultAgent
