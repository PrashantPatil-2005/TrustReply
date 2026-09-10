"""Evaluation harness for the support agent."""
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

from src.agent import SupportAgent
from src.baselines import SimpleBaseline, TrivialBaseline
from src.config import ROOT, load_config
from src.judge import judge_reply, human_agreement_study


def load_golden_set(path: Path | None = None) -> list[dict]:
    if path is None:
        path = ROOT / load_config()["evaluation"]["golden_set_path"]
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


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
        "precision_escalate": round(
            precision_score(gold, predictions, pos_label="escalate", zero_division=0), 3
        ),
        "recall_escalate": round(
            recall_score(gold, predictions, pos_label="escalate", zero_division=0), 3
        ),
        "f1_escalate": round(
            f1_score(gold, predictions, pos_label="escalate", zero_division=0), 3
        ),
    }


def run_model_on_golden(model_fn, examples: list[dict]) -> list[dict]:
    results = []
    for ex in examples:
        pred = model_fn(ex["customer_message"])
        results.append({
            "id": ex["id"],
            "gold_intent": ex["intent"],
            "pred_intent": pred["intent"],
            "gold_escalation": ex["escalation"],
            "pred_escalation": pred["escalation_action"],
            "draft_reply": pred["draft_reply"],
            "customer_message": ex["customer_message"],
            "historical_reply": ex.get("historical_reply", ""),
        })
    return results


def run_full_evaluation(
    judge_sample_size: int = 50,
    output_dir: Path | None = None,
) -> dict:
    cfg = load_config()
    examples = load_golden_set()
    output_dir = output_dir or (ROOT / "data" / "eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data for simple baseline
    pairs_path = ROOT / "data" / "processed" / f"{cfg['brand']}_pairs.csv"
    train_msgs, train_replies = [], []
    if pairs_path.exists():
        import pandas as pd
        train_df = pd.read_csv(pairs_path)
        train_msgs = train_df["customer_message"].tolist()
        train_replies = train_df["brand_reply"].tolist()

    agent = SupportAgent()
    trivial = TrivialBaseline()
    simple = SimpleBaseline(train_msgs, train_replies)

    models = {
        "agent": lambda msg: agent.handle(msg).to_dict() | {
            "escalation_action": agent.handle(msg).escalation_action,
        },
        "trivial_baseline": trivial.predict,
        "simple_baseline": simple.predict,
    }

    # Fix agent lambda to avoid double call
    def agent_predict(msg):
        r = agent.handle(msg)
        d = r.to_dict()
        d["escalation_action"] = r.escalation_action
        return d

    models["agent"] = agent_predict

    all_results = {}
    for name, fn in models.items():
        print(f"\nEvaluating {name}...")
        preds = run_model_on_golden(fn, examples)
        gold_intents = [p["gold_intent"] for p in preds]
        pred_intents = [p["pred_intent"] for p in preds]
        gold_esc = [p["gold_escalation"] for p in preds]
        pred_esc = [p["pred_escalation"] for p in preds]

        intent_metrics = evaluate_intents(pred_intents, gold_intents)
        esc_metrics = evaluate_escalation(pred_esc, gold_esc)

        all_results[name] = {
            "intent": {k: v for k, v in intent_metrics.items() if k != "report"},
            "intent_report": intent_metrics["report"],
            "escalation": esc_metrics,
            "predictions": preds,
        }
        print(f"  Intent accuracy: {intent_metrics['accuracy']}, F1: {intent_metrics['macro_f1']}")
        print(f"  Escalation accuracy: {esc_metrics['accuracy']}, escalate F1: {esc_metrics['f1_escalate']}")

    # LLM judge on agent replies (sample)
    rng = np.random.default_rng(cfg["evaluation"]["random_seed"])
    judge_indices = rng.choice(len(examples), size=min(judge_sample_size, len(examples)), replace=False)
    judge_scores = []
    human_scores = []
    for idx in judge_indices:
        ex = examples[idx]
        pred = all_results["agent"]["predictions"][idx]
        score = judge_reply(
            ex["customer_message"],
            pred["draft_reply"],
            ex.get("historical_reply"),
        )
        judge_scores.append(score)
        human_scores.append(ex.get("human_reply_quality", 3))

    agreement = human_agreement_study(examples[:len(judge_scores)], human_scores, judge_scores)
    all_results["llm_judge"] = {
        "agreement_with_human": agreement,
        "mean_judge_overall": round(np.mean([s.overall for s in judge_scores]), 2),
        "mean_judge_average": round(np.mean([s.average for s in judge_scores]), 2),
        "sample_size": len(judge_scores),
    }

    # Save results
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
        for k, v in all_results.items()
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(output_dir / "agent_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in all_results["agent"]["predictions"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_dir}")
    return all_results
