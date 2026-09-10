"""AmazonHelp AI support agent: classify, draft reply, decide escalation."""
from dataclasses import dataclass, asdict

from src.config import load_config
from src.escalation import EscalationDecision, decide_escalation
from src.intents import IntentResult, classify_intent_rules, INTENT_DESCRIPTIONS
from src.llm import call_llm, DEMO_MODE
from src.retrieval import HistoricalRetriever


@dataclass
class AgentResponse:
    customer_message: str
    intent: str
    intent_confidence: float
    draft_reply: str
    escalation_action: str
    escalation_reason: str
    similar_examples_used: int
    retrieval_context: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


BRAND_STYLE_GUIDE = """You are drafting replies for @AmazonHelp on Twitter.
Style rules from historical AmazonHelp responses:
- Start with empathy: "I'm sorry to hear..." or "We understand your frustration..."
- Keep replies concise (under 280 characters when possible)
- Never ask for passwords, full card numbers, or SSN
- Direct complex issues to DM: "Please send us a DM so we can look into this securely"
- Sign with agent initials like ^AM when appropriate
- Do not promise specific refund amounts or delivery dates
- Be professional, warm, and solution-oriented"""


def _template_reply(message: str, intent: IntentResult) -> str:
    """Rule-based reply templates when LLM unavailable."""
    templates = {
        "order_status": (
            "Thanks for reaching out! I'd be happy to help track your order. "
            "Please DM us your order number and we'll look into the status right away. ^AM"
        ),
        "delivery_issue": (
            "I'm sorry your delivery didn't go as expected. "
            "Please send us a DM with your order details so we can investigate. ^AM"
        ),
        "refund_return": (
            "We understand you'd like help with a return or refund. "
            "Please DM us your order number and we'll review your options. ^AM"
        ),
        "account_access": (
            "Sorry you're having trouble accessing your account. "
            "Please DM us so we can verify your identity and assist securely. ^AM"
        ),
        "payment_billing": (
            "We take billing concerns seriously. "
            "Please DM us with details about the charge and we'll review it. ^AM"
        ),
        "product_quality": (
            "I'm sorry the product didn't meet your expectations. "
            "Please DM us your order number so we can help with a resolution. ^AM"
        ),
        "subscription_prime": (
            "Happy to help with your Prime membership question. "
            "Please DM us with details and we'll assist you. ^AM"
        ),
        "technical_app": (
            "Sorry you're experiencing technical difficulties. "
            "Please DM us your device model and we'll troubleshoot together. ^AM"
        ),
        "praise_thanks": (
            "Thank you so much for the kind words! We're glad we could help. "
            "Don't hesitate to reach out if you need anything else. ^AM"
        ),
        "general_inquiry": (
            "Thanks for contacting @AmazonHelp! We'd love to assist. "
            "Could you share more details via DM so we can help? ^AM"
        ),
    }
    return templates.get(intent.intent, templates["general_inquiry"])


class SupportAgent:
    def __init__(self):
        self.cfg = load_config()
        self.retriever = HistoricalRetriever(top_k=self.cfg["data"]["retrieval_top_k"])

    def classify(self, message: str) -> IntentResult:
        return classify_intent_rules(message)

    def draft_reply(
        self,
        message: str,
        intent: IntentResult,
        escalation: EscalationDecision,
    ) -> tuple[str, str, int]:
        examples = self.retriever.retrieve(message)
        context = self.retriever.format_context(examples)

        if escalation.action == "escalate":
            # For escalated cases, draft a holding reply only
            holding = (
                "We understand this is important to you. "
                "A specialist from our team will follow up with you shortly. "
                "Please DM us your details so we can prioritize your case. ^AM"
            )
            return holding, context, len(examples)

        if DEMO_MODE or not examples:
            return _template_reply(message, intent), context, len(examples)

        system = BRAND_STYLE_GUIDE
        user = f"""Customer message: {message}
Detected intent: {intent.intent} ({INTENT_DESCRIPTIONS[intent.intent]})

Similar past conversations where AmazonHelp successfully resolved the issue:
{context}

Draft a reply in AmazonHelp's voice. Ground your response in how similar issues were handled above.
Reply only with the tweet text, no explanation."""

        reply = call_llm(system, user)
        if not reply:
            reply = _template_reply(message, intent)
        return reply, context, len(examples)

    def handle(self, message: str) -> AgentResponse:
        intent = self.classify(message)
        escalation = decide_escalation(message, intent)
        reply, context, n_examples = self.draft_reply(message, intent, escalation)

        return AgentResponse(
            customer_message=message,
            intent=intent.intent,
            intent_confidence=intent.confidence,
            draft_reply=reply,
            escalation_action=escalation.action,
            escalation_reason=escalation.reason,
            similar_examples_used=n_examples,
            retrieval_context=context,
        )
