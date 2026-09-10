"""Intent taxonomy and classification for AmazonHelp support."""
import re
from dataclasses import dataclass

INTENT_KEYWORDS: dict[str, list[str]] = {
    "order_status": [
        r"\bwhere\s+is\b", r"\btrack(ing)?\b", r"\border\s+status\b",
        r"\bshipped\b", r"\bnot\s+received\b", r"\bwhen\s+will\b.*\barriv",
        r"\bpackage\b", r"\bdelivery\s+date\b", r"\bstill\s+waiting\b",
    ],
    "delivery_issue": [
        r"\blate\b", r"\blost\b", r"\bdamaged\b", r"\bnever\s+arrived\b",
        r"\bwrong\s+address\b", r"\bmissing\b.*\bpackage\b", r"\bdelivered\b.*\bnot\b",
        r"\bstolen\b", r"\bporch\b", r"\bdriver\b",
    ],
    "refund_return": [
        r"\brefund\b", r"\breturn\b", r"\bmoney\s+back\b", r"\breplacement\b",
        r"\bsend\s+back\b", r"\bexchange\b", r"\bcancel\b.*\border\b",
    ],
    "account_access": [
        r"\blogin\b", r"\bpassword\b", r"\baccount\b.*\b(lock|access|sign)",
        r"\bsign\s*in\b", r"\bcan'?t\s+access\b", r"\breset\b.*\bpassword\b",
        r"\bverify\b.*\baccount\b", r"\b2fa\b", r"\botp\b",
    ],
    "payment_billing": [
        r"\bcharg(e|ed|ing)\b", r"\bbilling\b", r"\bpayment\b", r"\bcard\b",
        r"\bunauthorized\b", r"\bdouble\s+charg\b", r"\binvoice\b",
        r"\bsubscription\s+fee\b", r"\bprice\b",
    ],
    "product_quality": [
        r"\bbroken\b", r"\bdefect\b", r"\bnot\s+working\b", r"\bfaulty\b",
        r"\bquality\b", r"\bdoesn'?t\s+work\b", r"\bstopped\s+working\b",
        r"\bdead\s+on\s+arrival\b", r"\bdamaged\b.*\bproduct\b",
    ],
    "subscription_prime": [
        r"\bprime\b", r"\bmembership\b", r"\bsubscri(be|ption)\b",
        r"\bfree\s+trial\b", r"\bprime\s+video\b", r"\bprime\s+music\b",
        r"\brenew\b",
    ],
    "technical_app": [
        r"\bapp\b", r"\bwebsite\b", r"\berror\b", r"\bkindle\b",
        r"\bfire\s*tv\b", r"\becho\b", r"\balexa\b", r"\bdownload\b",
        r"\bstreaming\b", r"\bglitch\b", r"\bcrash\b", r"\bwon'?t\s+load\b",
    ],
    "praise_thanks": [
        r"\bthank(s| you)\b", r"\bappreciate\b", r"\bgreat\s+job\b",
        r"\bawesome\b", r"\bexcellent\b", r"\blove\s+amazon\b",
        r"\bresolved\b", r"\bhelpful\b", r"\b5\s*stars\b",
    ],
    "general_inquiry": [
        r"\bhelp\b", r"\bhow\s+do\s+i\b", r"\bquestion\b", r"\bcan\s+you\b",
        r"\binfo(rmation)?\b", r"\bassist\b", r"\bsupport\b",
    ],
}

INTENT_DESCRIPTIONS = {
    "order_status": "Customer asking about order tracking or delivery timing",
    "delivery_issue": "Problem with delivery: late, lost, damaged, or wrong address",
    "refund_return": "Requesting refund, return, exchange, or cancellation",
    "account_access": "Login, password, or account access problems",
    "payment_billing": "Billing, charges, payment method issues",
    "product_quality": "Product defective, broken, or not working",
    "subscription_prime": "Prime membership or subscription questions",
    "technical_app": "App, device, or website technical issues",
    "praise_thanks": "Positive feedback or thanks",
    "general_inquiry": "General help request without specific category",
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    scores: dict[str, float]


def classify_intent_rules(text: str) -> IntentResult:
    """Rule-based intent classifier using keyword patterns."""
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for intent, patterns in INTENT_KEYWORDS.items():
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1.0
        scores[intent] = score

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score == 0:
        return IntentResult("general_inquiry", 0.3, scores)

    total = sum(scores.values()) or 1.0
    confidence = min(0.95, 0.4 + (best_score / total) * 0.55)
    return IntentResult(best_intent, confidence, scores)


def get_all_intents() -> list[str]:
    return list(INTENT_KEYWORDS.keys())
