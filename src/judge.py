"""LLM-as-judge for reply quality evaluation."""
import json
import re
from dataclasses import dataclass

from src.llm import call_llm, DEMO_MODE

JUDGE_RUBRIC = """You are evaluating a customer support reply draft for @AmazonHelp on Twitter.

Score each dimension 1-5 (1=poor, 5=excellent):

1. RELEVANCE: Does the reply address the customer's actual issue?
2. BRAND_VOICE: Does it match AmazonHelp's empathetic, professional Twitter tone?
3. SAFETY: Does it avoid promising refunds, sharing sensitive info requests improperly, or making unauthorized commitments?
4. ACTIONABILITY: Does it give the customer a clear next step?
5. GROUNDING: Is the reply consistent with how AmazonHelp historically handles similar issues?

Respond ONLY with valid JSON:
{
  "relevance": <1-5>,
  "brand_voice": <1-5>,
  "safety": <1-5>,
  "actionability": <1-5>,
  "grounding": <1-5>,
  "overall": <1-5>,
  "brief_rationale": "<one sentence>"
}"""


@dataclass
class JudgeScore:
    relevance: int
    brand_voice: int
    safety: int
    actionability: int
    grounding: int
    overall: int
    brief_rationale: str

    @property
    def average(self) -> float:
        return (self.relevance + self.brand_voice + self.safety +
                self.actionability + self.grounding) / 5.0


def _demo_score(message: str, reply: str) -> JudgeScore:
    """Heuristic scoring when LLM unavailable."""
    score = 3
    if "DM" in reply or "dm" in reply.lower():
        score += 1
    if len(reply) > 20:
        score += 0
    else:
        score -= 1
    if any(w in message.lower() for w in ["thank", "great", "awesome"]):
        if any(w in reply.lower() for w in ["thank", "glad", "appreciate"]):
            score += 1
    score = max(1, min(5, score))
    return JudgeScore(score, score, score, score, score, score, "Demo heuristic score")


def judge_reply(
    customer_message: str,
    draft_reply: str,
    historical_reply: str | None = None,
) -> JudgeScore:
    if DEMO_MODE:
        return _demo_score(customer_message, draft_reply)

    context = ""
    if historical_reply:
        context = f"\nHistorical AmazonHelp reply for reference: {historical_reply}"

    user = f"""Customer message: {customer_message}
Draft reply to evaluate: {draft_reply}{context}

Evaluate the draft reply."""

    raw = call_llm(JUDGE_RUBRIC, user, temperature=0.1)
    if not raw:
        return _demo_score(customer_message, draft_reply)

    try:
        # Extract JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else json.loads(raw)
        return JudgeScore(
            relevance=int(data["relevance"]),
            brand_voice=int(data["brand_voice"]),
            safety=int(data["safety"]),
            actionability=int(data["actionability"]),
            grounding=int(data["grounding"]),
            overall=int(data["overall"]),
            brief_rationale=str(data.get("brief_rationale", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return _demo_score(customer_message, draft_reply)


def human_agreement_study(
    examples: list[dict],
    human_scores: list[int],
    agent_scores: list[JudgeScore],
) -> dict:
    """Compute agreement between LLM judge and human ratings."""
    if len(human_scores) != len(agent_scores):
        raise ValueError("Score lists must be same length")

    n = len(human_scores)
    llm_overall = [s.overall for s in agent_scores]

    # Cohen's kappa approximation via agreement within 1 point
    agree_exact = sum(1 for h, l in zip(human_scores, llm_overall) if h == l) / n
    agree_within_1 = sum(1 for h, l in zip(human_scores, llm_overall) if abs(h - l) <= 1) / n
    mae = sum(abs(h - l) for h, l in zip(human_scores, llm_overall)) / n

    return {
        "n_samples": n,
        "exact_agreement": round(agree_exact, 3),
        "agreement_within_1": round(agree_within_1, 3),
        "mean_absolute_error": round(mae, 3),
        "human_mean": round(sum(human_scores) / n, 2),
        "llm_mean": round(sum(llm_overall) / n, 2),
    }
