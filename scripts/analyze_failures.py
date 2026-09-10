"""Analyze top failure modes from evaluation predictions."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
preds_path = ROOT / "data" / "eval_results" / "agent_predictions.jsonl"

preds = [json.loads(l) for l in open(preds_path, encoding="utf-8")]
intent_fails = [p for p in preds if p["gold_intent"] != p["pred_intent"]]
esc_fails = [p for p in preds if p["gold_escalation"] != p["pred_escalation"]]

print(f"Intent failures: {len(intent_fails)}/200")
for (g, pr), c in Counter((p["gold_intent"], p["pred_intent"]) for p in intent_fails).most_common(8):
    print(f"  {g} -> {pr}: {c}")

print(f"\nEscalation failures: {len(esc_fails)}/200")
for k, v in Counter((p["gold_escalation"], p["pred_escalation"]) for p in esc_fails).most_common():
    print(f"  {k}: {v}")

print("\n--- Top 5 intent failure examples ---")
for p in intent_fails[:5]:
    print(f"\nCustomer: {p['customer_message'][:150]}")
    print(f"Gold: {p['gold_intent']} | Pred: {p['pred_intent']}")

print("\n--- Top 5 escalation failure examples ---")
for p in esc_fails[:5]:
    print(f"\nCustomer: {p['customer_message'][:150]}")
    print(f"Gold: {p['gold_escalation']} | Pred: {p['pred_escalation']}")
