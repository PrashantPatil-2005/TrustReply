# Decision Log

Non-obvious decisions made during this project and why.

1. **Brand: AmazonHelp** — Highest tweet volume in the dataset (~124k outbound tweets in first 2M rows). Large enough for retrieval and diverse intents, but not so niche that evaluation is meaningless.

2. **10-intent taxonomy instead of 77 (Banking77-style)** — AmazonHelp issues cluster around orders, delivery, returns, and devices. 10 intents balance coverage and annotatability; 77 would be sparse per class with <200 golden examples.

3. **Rule-based intent classifier over fine-tuned model** — With only 200 labeled examples and keyword-strong support tweets, rules achieve 82% accuracy matching a retrieval baseline. A fine-tuned classifier risks overfitting and adds training complexity for marginal gain.

4. **Conservative escalation default** — Financial, product, and delivery intents default to `escalate`. False escalation costs a human minute; false auto-handle risks customer harm and brand damage. Asymmetric error costs favor recall on escalation (94.4%).

5. **Retrieval via fuzzy token overlap, not embeddings** — Keeps pipeline runnable in <15 min with no GPU and no embedding model download. RapidFuzz token-set ratio is surprisingly effective on short tweets.

6. **LLM only for reply drafting, not classification** — Classification needs consistency and auditability; rules are deterministic. Reply drafting benefits from LLM fluency but is gated behind escalation check.

7. **Escalated messages get holding replies, not resolution attempts** — Agent drafts "a specialist will follow up" for escalated cases. Prevents the LLM from making unauthorized refund/delivery promises.

8. **Golden set labels use human-review rules, not pure automation** — Rules pre-label, then `_human_review_intent()` disambiguates known confusions (prime+charge → subscription_prime). Documented in `GOLDEN_SET_PROTOCOL.md`.

9. **Exclude non-English threads from golden set** — ~30% of AmazonHelp tweets are Japanese. Including them would conflate multilingual capability with support quality; out of scope.

10. **Pair extraction via `in_response_to_tweet_id`, not thread reconstruction** — Full thread trees are complex and many customer tweets aren't in the same chunk. Direct reply pairs are sufficient for retrieval and evaluation.

11. **DEMO_MODE for reproducibility** — Evaluators can run full pipeline without an API key. Reply quality uses templates; judge uses heuristics. Headline intent/escalation metrics are unaffected.

12. **Simple baseline shares intent classifier with agent** — Isolates the value-add of escalation logic and retrieval-grounded replies. Shows escalation is where the agent wins (84.5% vs 55.5%).

13. **LLM judge rubric has 5 dimensions + overall** — Mirrors support QA scorecards (relevance, voice, safety, actionability, grounding). Safety dimension catches unauthorized promises.

14. **50-sample judge subset** — Full 200 would cost ~$2–5 in API calls and slow the 15-min reproduction target. 50 is enough for agreement statistics.

15. **Cap retrieval index at 15k pairs** — Balances recall vs <15 min reproduction; sparse TF-IDF matrix for memory efficiency.

---

## ThreadVault v2 Decisions

16. **Evidence gating as primary innovation** — Auto-handle blocked when retrieval score < 0.30; measurable abstention.
17. **Resolution Pattern Objects (RPOs)** — Extract actions/constraints from replies instead of copying raw text.
18. **Hybrid BM25 + TF-IDF retrieval** — Beats fuzzy match on paraphrases without GPU embedding models.
19. **Calibrated escalation model (13 features)** — Replaces hand-tuned rules; trained on golden_train only.
20. **40-example holdout test set** — Headline numbers on data never used for threshold tuning.
21. **Strict Trust Score (STS)** — End-to-end metric combining intent + escalation + verification; honest at ~58%.
22. **Sparse TF-IDF index** — scipy.sparse.save_npz instead of dense matrix; 10x faster index build.
23. **Streamlit control center** — Full trace visualization for judges; highest demo impact per hour.
24. **Verification before escalation decision** — Groundedness feeds escalation features, not just post-hoc check.
25. **TF-IDF + LR intent classifier** — Separates from simple keyword baseline; trained on 15k weak-labeled pairs.