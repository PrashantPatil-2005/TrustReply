"""ThreadVault Support Control Center — Streamlit demo."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import ThreadVaultAgent

st.set_page_config(page_title="ThreadVault Control Center", layout="wide")
st.title("ThreadVault — Evidence-Gated Support Agent")
st.caption("@AmazonHelp customer support control center")

@st.cache_resource
def load_agent():
    return ThreadVaultAgent()

agent = load_agent()

col_in, col_out = st.columns([1, 1])

with col_in:
    message = st.text_area(
        "Customer message",
        value="@AmazonHelp my package still hasn't arrived after 2 weeks, I've contacted support 3 times!",
        height=120,
    )
    context = st.text_area("Thread context (optional, one message per line)", height=80)
    run = st.button("Analyze", type="primary")

with col_out:
    if run and message.strip():
        ctx = [l.strip() for l in context.splitlines() if l.strip()]
        trace = agent.handle(message.strip(), thread_context=ctx)

        st.subheader("Intent")
        st.write(f"**{trace.intent}** (confidence: {trace.intent_confidence:.2f})")

        st.subheader("Retrieval")
        for hit in trace.retrieval_hits[:5]:
            st.markdown(f"- `{hit['evidence_id']}` score={hit['score']:.3f} — _{hit['customer_snippet']}_")

        st.subheader("Selected Evidence")
        st.write(f"Consistency: **{trace.evidence_consistency:.2f}** | Gate passed: **{trace.evidence_gate_passed}**")
        for ev in trace.selected_evidence:
            st.json(ev)

        st.subheader("Draft Reply")
        st.info(trace.draft_reply)

        st.subheader("Verification")
        st.json(trace.verification)
        st.metric("Groundedness", f"{trace.groundedness_score:.2f}")

        decision_color = "🟢" if trace.escalation_action == "auto_handle" else "🔴"
        st.subheader(f"{decision_color} {trace.escalation_action.upper()}")
        st.write(trace.escalation_reason)
        st.metric("Risk score", f"{trace.risk_score:.2f}")

        st.subheader("Strict Trust Pass")
        st.write("✅" if trace.strict_trust_pass else "❌")

        with st.expander("Full JSON trace"):
            st.code(json.dumps(trace.to_dict(), indent=2), language="json")
