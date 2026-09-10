"""Hybrid BM25 + TF-IDF retrieval over historical support cases."""
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ROOT, load_config
from src.evidence import ResolutionPattern, build_rpo


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """BM25 + TF-IDF cosine hybrid retriever with persisted index."""

    def __init__(
        self,
        index_dir: Path | None = None,
        top_k: int = 5,
        bm25_weight: float = 0.45,
        tfidf_weight: float = 0.55,
    ):
        cfg = load_config()
        brand = cfg["brand"]
        if index_dir is None:
            index_dir = ROOT / "data" / "processed" / "indices" / brand
        self.index_dir = index_dir
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight

        self.examples: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.tfidf: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        if (index_dir / "meta.json").exists():
            self._load()

    def _load(self) -> None:
        with open(self.index_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.examples = meta["examples"]
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.index_dir / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        from scipy import sparse
        npz_path = self.index_dir / "tfidf_matrix.npz"
        npy_path = self.index_dir / "tfidf_matrix.npy"
        if npz_path.exists():
            self.tfidf_matrix = sparse.load_npz(npz_path)
        else:
            self.tfidf_matrix = np.load(npy_path)

    @classmethod
    def build_index(
        cls,
        pairs_path: Path,
        output_dir: Path,
        max_examples: int | None = None,
    ) -> "HybridRetriever":
        """Build and persist hybrid index from pairs CSV."""
        import pandas as pd

        df = pd.read_csv(pairs_path)
        if max_examples:
            df = df.head(max_examples)

        examples = []
        corpus_tokens = []
        corpus_texts = []

        for i, row in df.iterrows():
            cust = str(row["customer_message"])
            brand = str(row["brand_reply"])
            examples.append({
                "evidence_id": f"ev_{row.get('tweet_id', i)}",
                "customer_message": cust,
                "brand_reply": brand,
            })
            corpus_tokens.append(_tokenize(cust))
            corpus_texts.append(cust)

        output_dir.mkdir(parents=True, exist_ok=True)

        bm25 = BM25Okapi(corpus_tokens)
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
        tfidf_matrix = tfidf.fit_transform(corpus_texts)

        with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"examples": examples, "n": len(examples)}, f)
        with open(output_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
        with open(output_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        from scipy import sparse
        sparse.save_npz(output_dir / "tfidf_matrix.npz", tfidf_matrix)

        retriever = cls(index_dir=output_dir)
        retriever.examples = examples
        retriever.bm25 = bm25
        retriever.tfidf = tfidf
        retriever.tfidf_matrix = tfidf_matrix
        return retriever

    def retrieve(self, query: str) -> list[ResolutionPattern]:
        if not self.examples or self.bm25 is None or self.tfidf is None:
            return []

        tokens = _tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokens))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        q_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        hybrid = self.bm25_weight * bm25_scores + self.tfidf_weight * tfidf_scores
        top_idx = np.argsort(hybrid)[::-1][: self.top_k]

        results = []
        for rank, idx in enumerate(top_idx):
            ex = self.examples[idx]
            score = float(hybrid[idx])
            results.append(build_rpo(
                evidence_id=ex["evidence_id"],
                customer_msg=ex["customer_message"],
                brand_reply=ex["brand_reply"],
                score=score,
            ))
        return results

    def top_score(self, query: str) -> float:
        hits = self.retrieve(query)
        return hits[0].retrieval_score if hits else 0.0

    def score_margin(self, query: str) -> float:
        hits = self.retrieve(query)
        if len(hits) < 2:
            return 0.0
        return hits[0].retrieval_score - hits[1].retrieval_score
