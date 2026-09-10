"""Baseline models for comparison."""
import random
import re
from collections import Counter

from src.intents import classify_intent_rules, get_all_intents


class TrivialBaseline:
    """Always predict general_inquiry, generic reply, never escalate."""

    name = "trivial_baseline"

    def predict(self, message: str) -> dict:
        return {
            "intent": "general_inquiry",
            "intent_confidence": 0.1,
            "draft_reply": "Thanks for contacting us. How can we help you today?",
            "escalation_action": "auto_handle",
            "escalation_reason": "Trivial baseline always auto-handles",
        }


class SimpleBaseline:
    """Keyword intent + most-common historical reply + simple escalation rules."""

    name = "simple_baseline"

    def __init__(self, train_messages: list[str], train_replies: list[str]):
        self.intent_clf = classify_intent_rules
        # Build per-intent reply bank from training data
        self.reply_bank: dict[str, list[str]] = {}
        for msg, reply in zip(train_messages, train_replies):
            intent = self.intent_clf(msg).intent
            self.reply_bank.setdefault(intent, []).append(reply)
        # Fallback: most common reply overall
        self.default_reply = Counter(train_replies).most_common(1)[0][0] if train_replies else "Please DM us for assistance."

    def predict(self, message: str) -> dict:
        intent_result = self.intent_clf(message)
        intent = intent_result.intent

        replies = self.reply_bank.get(intent, [])
        draft = random.choice(replies) if replies else self.default_reply
        # Truncate to tweet length
        draft = draft[:280]

        # Simple escalation: high-risk intents or frustration
        escalate_intents = {"payment_billing", "refund_return", "delivery_issue"}
        frustration = bool(re.search(r"\b(worst|terrible|scam|fraud|lawyer)\b", message.lower()))
        if intent in escalate_intents or frustration:
            action, reason = "escalate", f"Simple baseline escalates {intent} or frustration"
        else:
            action, reason = "auto_handle", f"Simple baseline auto-handles {intent}"

        return {
            "intent": intent,
            "intent_confidence": intent_result.confidence,
            "draft_reply": draft,
            "escalation_action": action,
            "escalation_reason": reason,
        }
