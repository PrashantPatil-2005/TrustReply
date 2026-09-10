"""Train intent classifier, build retrieval index, train escalation model."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config
from src.escalation_features import FEATURE_NAMES, extract_escalation_features
from src.escalation_model import EscalationModel
from src.evidence import select_evidence
from src.hybrid_retrieval import HybridRetriever
from src.intent_classifier import IntentClassifier
from src.intents import classify_intent_rules
from src.verification import verify_reply


def train_intent_classifier(cfg: dict) -> None:
    pairs_path = ROOT / "data" / "processed" / f"{cfg['brand']}_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Run preprocess first: {pairs_path}")

    df = pd.read_csv(pairs_path).head(15000)
    texts = df["customer_message"].astype(str).tolist()
    labels = [classify_intent_rules(t).intent for t in texts]

    out = ROOT / "data" / "processed" / "models" / f"{cfg['brand']}_intent.pkl"
    IntentClassifier.train(texts, labels, out)
    print(f"Intent classifier saved to {out}")


def build_retrieval_index(cfg: dict) -> None:
    pairs_path = ROOT / "data" / "processed" / f"{cfg['brand']}_pairs.csv"
    out_dir = ROOT / "data" / "processed" / "indices" / cfg["brand"]
    max_idx = min(cfg["data"]["max_train_samples"], 15000)
    HybridRetriever.build_index(pairs_path, out_dir, max_examples=max_idx)
    print(f"Hybrid retrieval index saved to {out_dir}")


def train_escalation_model(cfg: dict) -> None:
    golden_path = ROOT / cfg["evaluation"].get("golden_train_path", "data/golden_train.jsonl")
    if not golden_path.exists():
        golden_path = ROOT / cfg["evaluation"]["golden_set_path"]

    examples = [json.loads(l) for l in open(golden_path, encoding="utf-8")]
    retriever = HybridRetriever(top_k=5)
    intent_clf = IntentClassifier()

    X_rows, y = [], []
    for ex in examples:
        msg = ex["customer_message"]
        intent = intent_clf.predict(msg)
        evidence = retriever.retrieve(msg)
        selected = select_evidence(evidence, min_score=0.30)
        draft, _, actions = ("", [], [])
        from src.agent import _compose_from_evidence
        draft, _, actions = _compose_from_evidence(msg, selected, intent.intent)
        v = verify_reply(draft, selected, cited_actions=actions)
        feats = extract_escalation_features(msg, intent, selected, v.passed, v.groundedness_score)
        X_rows.append([feats[n] for n in FEATURE_NAMES])
        y.append(1 if ex["escalation"] == "escalate" else 0)

    X = np.array(X_rows)
    y_arr = np.array(y)
    out = ROOT / "data" / "processed" / "models" / f"{cfg['brand']}_escalation.pkl"
    EscalationModel.train(X, y_arr, out)
    print(f"Escalation model saved to {out} (n={len(y)})")


def main():
    cfg = load_config()
    print("=== Building hybrid retrieval index ===")
    build_retrieval_index(cfg)
    print("=== Training intent classifier ===")
    train_intent_classifier(cfg)
    print("=== Training escalation model ===")
    train_escalation_model(cfg)
    print("Done.")


if __name__ == "__main__":
    main()
