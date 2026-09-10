"""Retrieve similar historical resolutions for reply grounding."""
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

from src.config import ROOT, load_config


class HistoricalRetriever:
    """Simple keyword + fuzzy retrieval over past brand replies."""

    def __init__(self, index_path: Path | None = None, top_k: int = 5):
        cfg = load_config()
        if index_path is None:
            index_path = ROOT / "data" / "processed" / f"{cfg['brand']}_retrieval.jsonl"
        self.top_k = top_k
        self.examples: list[dict] = []
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    self.examples.append(json.loads(line))

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) > 2}

    def retrieve(self, query: str) -> list[dict]:
        if not self.examples:
            return []

        query_tokens = self._tokenize(query)
        scored = []
        for ex in self.examples:
            cust = ex["customer_message"]
            cust_tokens = self._tokenize(cust)
            overlap = len(query_tokens & cust_tokens)
            fuzzy = fuzz.token_set_ratio(query, cust) / 100.0
            score = overlap * 0.4 + fuzzy * 0.6
            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[: self.top_k]]

    def format_context(self, examples: list[dict]) -> str:
        if not examples:
            return "No similar historical conversations found."
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Customer: {ex['customer_message'][:300]}")
            lines.append(f"  AmazonHelp replied: {ex['brand_reply'][:300]}")
        return "\n".join(lines)
