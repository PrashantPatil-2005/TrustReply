"""TF-IDF + LogisticRegression intent classifier (trainable, beats pure rules)."""
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import ROOT, load_config
from src.intents import IntentResult, classify_intent_rules, get_all_intents


class IntentClassifier:
    def __init__(self, model_path: Path | None = None):
        cfg = load_config()
        if model_path is None:
            model_path = ROOT / "data" / "processed" / "models" / f"{cfg['brand']}_intent.pkl"
        self.model_path = model_path
        self.pipeline: Pipeline | None = None
        self.labels = get_all_intents()
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.pipeline = pickle.load(f)

    @classmethod
    def train(cls, texts: list[str], labels: list[str], output_path: Path) -> "IntentClassifier":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ])
        pipeline.fit(texts, labels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(pipeline, f)
        clf = cls(model_path=output_path)
        clf.pipeline = pipeline
        return clf

    def predict(self, text: str) -> IntentResult:
        if self.pipeline is None:
            return classify_intent_rules(text)

        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = int(np.argmax(proba))
        intent = classes[best_idx]
        confidence = float(proba[best_idx])
        scores = {c: float(p) for c, p in zip(classes, proba)}
        return IntentResult(intent=intent, confidence=confidence, scores=scores)
