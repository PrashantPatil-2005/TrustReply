"""Grounding and policy verification for drafted replies."""
import re
from dataclasses import dataclass, field

from src.evidence import ResolutionPattern


@dataclass
class VerificationResult:
    passed: bool
    groundedness_score: float
    checks: dict[str, bool] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "groundedness_score": round(self.groundedness_score, 3),
            "checks": self.checks,
            "violations": self.violations,
        }


UNSAFE_PATTERNS = [
    (r"\bwill\s+refund\b", "promised_refund"),
    (r"\bguarantee\b.*\brefund\b", "guaranteed_refund"),
    (r"\barrive\s+by\s+\w+day\b", "promised_delivery_date"),
    (r"\bwill\s+deliver\s+on\b", "promised_delivery_date"),
    (r"\bpassword\b", "requested_password"),
    (r"\bssn\b|\bsocial security\b", "requested_ssn"),
    (r"\bfull card number\b", "requested_card"),
]


def verify_reply(
    reply: str,
    evidence: list[ResolutionPattern],
    cited_actions: list[str] | None = None,
) -> VerificationResult:
    """Run deterministic policy and grounding checks."""
    checks: dict[str, bool] = {}
    violations: list[str] = []

    # Policy safety checks
    for pattern, name in UNSAFE_PATTERNS:
        if re.search(pattern, reply, re.IGNORECASE):
            checks[name] = False
            violations.append(name)
        else:
            checks[name] = True

    # Evidence support: actions in reply should come from evidence
    allowed_actions = set()
    for ev in evidence:
        allowed_actions.update(ev.actions)

    if cited_actions:
        unsupported = [a for a in cited_actions if a not in allowed_actions and allowed_actions]
        checks["actions_supported"] = len(unsupported) == 0
        if unsupported:
            violations.append(f"unsupported_actions:{','.join(unsupported)}")
    else:
        checks["actions_supported"] = True

    # Citation validity
    checks["has_evidence"] = len(evidence) > 0

    # DM redirect for sensitive issues (soft check)
    checks["actionable"] = bool(
        re.search(r"\b(dm|direct message|message us)\b", reply, re.IGNORECASE)
        or "thank" in reply.lower()
    )

    policy_pass = all(v for k, v in checks.items() if k not in ("has_evidence", "actionable"))
    has_evidence = checks.get("has_evidence", False)
    actionable = checks.get("actionable", False)

    groundedness = (
        0.35 * (1.0 if policy_pass else 0.0)
        + 0.25 * (1.0 if checks.get("actions_supported", False) else 0.0)
        + 0.20 * (1.0 if has_evidence else 0.0)
        + 0.20 * (1.0 if actionable else 0.0)
    )

    passed = policy_pass and groundedness >= 0.6 and len(violations) == 0

    return VerificationResult(
        passed=passed,
        groundedness_score=groundedness,
        checks=checks,
        violations=violations,
    )
