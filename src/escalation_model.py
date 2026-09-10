"""Calibrated escalation model trained on golden set."""
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.config import ROOT, load_config
from src.escalation import EscalationDecision, decide_escalation
from src.escalation_features import FEATURE_NAMES, extract_escalation_features
from src.evidence import ResolutionPattern, select_evidence
from src.intents import IntentResult


class EscalationModel:
    def __init__(self, model_path: Path | None = None, threshold: float = 0.5):
        cfg = load_config()
        if model_path is None:
            model_path = ROOT / "data" / "processed" / "models" / f"{cfg['brand']}_escalation.pkl"
        self.model_path = model_path
        self.threshold = threshold
        self.model: CalibratedClassifierCV | None = None
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.threshold = data.get("threshold", 0.5)

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        output_path: Path,
        threshold: float | None = None,
    ) -> "EscalationModel":
        base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        model.fit(X, y)

        if threshold is None:
            # Optimize threshold for escalate recall >= 0.90 on training data
            proba = model.predict_proba(X)[:, 1]
            best_t, best_f1 = 0.5, 0.0
            for t in np.arange(0.2, 0.8, 0.05):
                preds = (proba >= t).astype(int)
                tp = ((preds == 1) & (y == 1)).sum()
                fp = ((preds == 1) & (y == 0)).sum()
                fn = ((preds == 0) & (y == 1)).sum()
                prec = tp / (tp + fp) if (tp + fp) else 0
                rec = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
                if rec >= 0.85 and f1 > best_f1:
                    best_f1, best_t = f1, t
            threshold = best_t

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({"model": model, "threshold": threshold}, f)

        em = cls(model_path=output_path, threshold=threshold)
        em.model = model
        return em

    def predict_proba_escalate(self, features: dict[str, float]) -> float:
        if self.model is None:
            return 0.5
        x = np.array([[features[n] for n in FEATURE_NAMES]])
        return float(self.model.predict_proba(x)[0, 1])

    def decide(
        self,
        message: str,
        intent: IntentResult,
        evidence: list[ResolutionPattern],
        verification_passed: bool,
        groundedness: float,
    ) -> EscalationDecision:
        features = extract_escalation_features(
            message, intent, evidence, verification_passed, groundedness
        )

        if self.model is None:
            return decide_escalation(message, intent)

        p_escalate = self.predict_proba_escalate(features)
        reasons = []

        if features["legal_risk"] > 0:
            reasons.append("legal/safety keyword detected")
        if features["weak_evidence"] > 0:
            reasons.append(f"weak retrieval evidence ({features['retrieval_top1']:.2f})")
        if features["frustration"] >= 0.5:
            reasons.append(f"high frustration ({features['frustration']:.2f})")
        if features["multi_attempt"] >= 0.5:
            reasons.append("multi-attempt language detected")
        if features["high_risk_intent"] > 0:
            reasons.append(f"high-risk intent: {intent.intent}")
        if not verification_passed:
            reasons.append("verification checks failed")

        # Evidence gate: force escalate if evidence too weak for auto-handle
        selected = select_evidence(evidence, min_score=0.30)
        if not selected and intent.intent != "praise_thanks":
            return EscalationDecision(
                "escalate",
                "Evidence gate: insufficient historical evidence to auto-handle",
                p_escalate,
            )

        if p_escalate >= self.threshold:
            reason = "; ".join(reasons) if reasons else f"risk score {p_escalate:.2f} >= {self.threshold:.2f}"
            return EscalationDecision("escalate", reason, p_escalate)

        reason = "; ".join(reasons) if reasons else f"risk score {p_escalate:.2f} below threshold"
        if features["praise_intent"] > 0:
            reason = "positive feedback — safe to auto-acknowledge"
        return EscalationDecision("auto_handle", reason, 1.0 - p_escalate)
