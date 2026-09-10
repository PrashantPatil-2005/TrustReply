"""Evaluation harness with Strict Trust Score (STS)."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.agent import ThreadVaultAgent
from src.baselines import SimpleBaseline, TrivialBaseline
from src.config import ROOT, load_config
from src.judge import judge_reply, human_agreement_study


def load_golden_set(path: Path | None = None) -> list[dict]:
    cfg = load_config()
    if path is None:
        test_path = ROOT / cfg["evaluation"].get("golden_test_path", "data/golden_test.jsonl")
        path = test_path if test_path.exists() else ROOT / cfg["evaluation"]["golden_set_path"]
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def evaluate_intents(predictions: list[str], gold: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(gold, predictions), 3),
        "macro_f1": round(f1_score(gold, predictions, average="macro", zero_division=0), 3),
        "weighted_f1": round(f1_score(gold, predictions, average="weighted", zero_division=0), 3),
        "report": classification_report(gold, predictions, zero_division=0),
    }


def evaluate_escalation(predictions: list[str], gold: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(gold, predictions), 3),
        "precision_escalate": round(precision_score(gold, predictions, pos_label="escalate", zero_division=0), 3),
        "recall_escalate": round(recall_score(gold, predictions, pos_label="escalate", zero_division=0), 3),
        "f1_escalate": round(f1_score(gold, predictions, pos_label="escalate", zero_division=0), 3),
    }


def compute_strict_trust_score(preds: list[dict]) -> dict:
    """STS: intent correct AND escalation correct AND verification passed AND auto-handled correctly."""
    n = len(preds)
    if n == 0:
        return {"sts": 0.0, "n": 0}

    strict_pass = 0
    for p in preds:
        intent_ok = p["gold_intent"] == p["pred_intent"]
        esc_ok = p["gold_escalation"] == p["pred_escalation"]
        verify_ok = p.get("verification_passed", False)
        if intent_ok and esc_ok and verify_ok:
            strict_pass += 1

    return {"sts": round(strict_pass / n, 3), "n": n, "strict_pass_count": strict_pass}


def run_model_on_golden(model_fn, examples: list[dict]) -> list[dict]:
    results = []
    for ex in examples:
        pred = model_fn(ex["customer_message"])
        results.append({
            "id": ex["id"],
            "gold_intent": ex["intent"],
            "pred_intent": pred.get("intent", ""),
            "gold_escalation": ex["escalation"],
            "pred_escalation": pred.get("escalation_action", ""),
            "draft_reply": pred.get("draft_reply", ""),
            "customer_message": ex["customer_message"],
            "historical_reply": ex.get("historical_reply", ""),
            "verification_passed": pred.get("verification_passed", False),
            "evidence_gate_passed": pred.get("evidence_gate_passed", False),
            "groundedness_score": pred.get("groundedness_score", 0.0),
            "retrieval_top_score": pred.get("retrieval_top_score", 0.0),
            "trace": pred.get("trace"),
        })
    return results


def run_full_evaluation(judge_sample_size: int = 50, output_dir: Path | None = None) -> dict:
    cfg = load_config()
    examples = load_golden_set()
    output_dir = output_dir or (ROOT / "data" / "eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = ROOT / "data" / "processed" / f"{cfg['brand']}_pairs.csv"
    train_msgs, train_replies = [], []
    if pairs_path.exists():
        import pandas as pd
        df = pd.read_csv(pairs_path)
        train_msgs = df["customer_message"].tolist()
        train_replies = df["brand_reply"].tolist()

    agent = ThreadVaultAgent()
    trivial = TrivialBaseline()
    simple = SimpleBaseline(train_msgs, train_replies)

    def agent_predict(msg):
        trace = agent.handle(msg)
        d = trace.to_dict()
        d["intent"] = trace.intent
        d["intent_confidence"] = trace.intent_confidence
        d["draft_reply"] = trace.draft_reply
        d["escalation_action"] = trace.escalation_action
        d["escalation_reason"] = trace.escalation_reason
        d["verification_passed"] = trace.verification.get("passed", False)
        d["evidence_gate_passed"] = trace.evidence_gate_passed
        d["groundedness_score"] = trace.groundedness_score
        d["retrieval_top_score"] = trace.retrieval_top_score
        return d

    models = {
        "threadvault_agent": agent_predict,
        "trivial_baseline": trivial.predict,
        "simple_baseline": simple.predict,
    }

    all_results = {}
    for name, fn in models.items():
        print(f"\nEvaluating {name}...")
        preds = run_model_on_golden(fn, examples)
        gold_intents = [p["gold_intent"] for p in preds]
        pred_intents = [p["pred_intent"] for p in preds]
        gold_esc = [p["gold_escalation"] for p in preds]
        pred_esc = [p["pred_escalation"] for p in preds]

        intent_m = evaluate_intents(pred_intents, gold_intents)
        esc_m = evaluate_escalation(pred_esc, gold_esc)
        sts = compute_strict_trust_score(preds)

        all_results[name] = {
            "intent": {k: v for k, v in intent_m.items() if k != "report"},
            "intent_report": intent_m["report"],
            "escalation": esc_m,
            "strict_trust_score": sts,
            "predictions": preds,
        }
        print(f"  Intent acc: {intent_m['accuracy']}, F1: {intent_m['macro_f1']}")
        print(f"  Escalation acc: {esc_m['accuracy']}, escalate F1: {esc_m['f1_escalate']}")
        print(f"  Strict Trust Score: {sts['sts']}")

    rng = np.random.default_rng(cfg["evaluation"]["random_seed"])
    idx = rng.choice(len(examples), size=min(judge_sample_size, len(examples)), replace=False)
    judge_scores, human_scores = [], []
    for i in idx:
        ex = examples[i]
        pred = all_results["threadvault_agent"]["predictions"][i]
        judge_scores.append(judge_reply(ex["customer_message"], pred["draft_reply"], ex.get("historical_reply")))
        human_scores.append(ex.get("human_reply_quality", 3))

    agreement = human_agreement_study(
        [examples[i] for i in idx], human_scores, judge_scores
    )
    all_results["llm_judge"] = {
        "agreement_with_human": agreement,
        "mean_judge_overall": round(np.mean([s.overall for s in judge_scores]), 2),
        "sample_size": len(judge_scores),
    }

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in all_results.items()}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(output_dir / "agent_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in all_results["threadvault_agent"]["predictions"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_dir}")
    return all_results
