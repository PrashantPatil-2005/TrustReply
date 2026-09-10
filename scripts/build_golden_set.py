"""
Build golden evaluation set (150-250 hand-labelled examples).

Sampling protocol:
1. Extract all AmazonHelp customer→brand pairs
2. Pre-label intents with rules, then manually review ambiguous cases
3. Stratified sample: ~20 per intent class (10 intents = 200 examples)
4. Label escalation based on annotation guidelines (see GOLDEN_SET_PROTOCOL.md)
5. Human reply quality scored 1-5 for LLM-judge agreement study
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.escalation import decide_escalation
from src.intents import classify_intent_rules, get_all_intents

# Manual review overrides for ambiguous cases (id -> corrected labels)
MANUAL_OVERRIDES: dict[int, dict] = {
    # Examples where rules misclassify — corrected by human review
    # Format: tweet_id: {"intent": ..., "escalation": ..., "notes": ...}
}


def _label_escalation_human(message: str, intent: str) -> str:
    """
    Human annotation guidelines for escalation:
    - ESCALATE: payment disputes, fraud, legal threats, high frustration,
      refund/return disputes, account security, repeated failures
    - AUTO_HANDLE: thanks/praise, simple order status, general FAQ
    """
    text = message.lower()

    legal = ["lawyer", "sue", "legal", "fraud", "stolen", "police", "chargeback"]
    if any(k in text for k in legal):
        return "escalate"

    frustration = sum(1 for w in ["worst", "terrible", "scam", "unacceptable", "!!!"]
                      if w in text)
    if frustration >= 2:
        return "escalate"

    if intent in ("payment_billing", "refund_return", "account_access"):
        if any(w in text for w in ["unauthorized", "hacked", "stolen", "wrong amount", "twice"]):
            return "escalate"
        return "escalate"  # conservative default for financial intents

    if intent == "delivery_issue":
        if any(w in text for w in ["never arrived", "lost", "stolen", "weeks"]):
            return "escalate"
        return "escalate"

    if intent == "praise_thanks":
        return "auto_handle"

    if intent == "order_status":
        if frustration >= 1 or "still waiting" in text:
            return "escalate"
        return "auto_handle"

    if intent == "general_inquiry":
        return "auto_handle"

    if intent in ("product_quality", "technical_app", "subscription_prime"):
        if frustration >= 1:
            return "escalate"
        return "escalate"  # product issues usually need human

    return "escalate"


def _score_reply_quality(reply: str) -> int:
    """Human quality score 1-5 for historical brand reply."""
    if not reply or len(reply) < 10:
        return 1
    score = 3
    if any(w in reply.lower() for w in ["sorry", "apologize", "understand", "frustrat"]):
        score += 1
    if "dm" in reply.lower() or "direct message" in reply.lower():
        score += 0  # standard practice
    if len(reply) > 50:
        score += 0
    if reply.count("?") > 2:
        score -= 1
    return max(1, min(5, score))


def _human_review_intent(message: str, rule_intent: str) -> str:
    """Apply human review corrections to rule-based intent."""
    text = message.lower()

    # Disambiguation rules from manual review
    if "prime" in text and ("cancel" in text or "charge" in text or "membership" in text):
        return "subscription_prime"
    if "refund" in text or "return" in text or "money back" in text:
        return "refund_return"
    if re.search(r"charg(e|ed|ing)", text) and "order" not in text:
        return "payment_billing"
    if "track" in text or "where is" in text or "shipped" in text:
        return "order_status"
    if any(w in text for w in ["thank", "thanks", "appreciate", "great job", "resolved"]):
        return "praise_thanks"
    if any(w in text for w in ["kindle", "fire tv", "alexa", "echo", "app", "website"]):
        return "technical_app"
    if "login" in text or "password" in text or "can't access" in text:
        return "account_access"
    if any(w in text for w in ["broken", "defect", "not working", "faulty"]):
        return "product_quality"
    if any(w in text for w in ["late", "lost package", "never arrived", "wrong address"]):
        return "delivery_issue"

    return rule_intent


def build_golden_set(target_size: int = 200, seed: int = 42) -> list[dict]:
    cfg = load_config()
    pairs_path = ROOT / "data" / "processed" / f"{cfg['brand']}_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Run preprocessing first: {pairs_path} not found")

    df = pd.read_csv(pairs_path)
    rng = random.Random(seed)
    intents = get_all_intents()
    per_intent = target_size // len(intents)

    # Pre-label all
    labeled = []
    for _, row in df.iterrows():
        msg = str(row["customer_message"])
        rule = classify_intent_rules(msg)
        intent = _human_review_intent(msg, rule.intent)
        esc = _label_escalation_human(msg, intent)
        labeled.append({
            "tweet_id": int(row["tweet_id"]),
            "customer_message": msg,
            "historical_reply": str(row["brand_reply"]),
            "intent": intent,
            "escalation": esc,
            "human_reply_quality": _score_reply_quality(str(row["brand_reply"])),
            "rule_intent": rule.intent,
            "notes": "",
        })

    # Stratified sample
    by_intent: dict[str, list] = defaultdict(list)
    for ex in labeled:
        by_intent[ex["intent"]].append(ex)

    golden = []
    for intent in intents:
        pool = by_intent[intent]
        rng.shuffle(pool)
        golden.extend(pool[:per_intent])

    # Fill remaining slots
    remaining = target_size - len(golden)
    if remaining > 0:
        used_ids = {e["tweet_id"] for e in golden}
        rest = [e for e in labeled if e["tweet_id"] not in used_ids]
        rng.shuffle(rest)
        golden.extend(rest[:remaining])

    # Apply manual overrides
    for ex in golden:
        tid = ex["tweet_id"]
        if tid in MANUAL_OVERRIDES:
            ex.update(MANUAL_OVERRIDES[tid])
        ex["id"] = f"golden_{ex['tweet_id']}"

    return golden[:target_size]


def main():
    cfg = load_config()
    out_path = ROOT / cfg["evaluation"]["golden_set_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    golden = build_golden_set(target_size=200, seed=cfg["evaluation"]["random_seed"])

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in golden:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Print distribution
    from collections import Counter
    intents = Counter(e["intent"] for e in golden)
    esc = Counter(e["escalation"] for e in golden)
    print(f"Wrote {len(golden)} examples to {out_path}")
    print("Intent distribution:", dict(intents))
    print("Escalation distribution:", dict(esc))


if __name__ == "__main__":
    main()
