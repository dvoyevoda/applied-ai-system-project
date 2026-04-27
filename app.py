from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.orchestrator import InboxTriageOrchestrator


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


@st.cache_data
def load_samples() -> pd.DataFrame:
    sample_path = DATA_DIR / "sample_messages.csv"
    if sample_path.exists():
        samples = pd.read_csv(sample_path)
        if "section" not in samples.columns:
            samples.insert(0, "section", "Real Requests")
        return samples
    return pd.DataFrame(columns=["section", "label", "message"])


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AI Inbox Triage Assistant", layout="wide")

    st.title("AI Inbox Triage Assistant")
    st.caption("Classify incoming messages, retrieve policy context, draft a response, and flag cases for human review.")

    with st.sidebar:
        st.header("AI Settings")
        api_key = st.text_input(
            "OpenAI API key",
            value="",
            type="password",
            help="Paste your key here for testing. It is only used for this session and is not written to logs.",
        )
        env_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key and env_key:
            api_key = env_key
            st.success("Using OPENAI_API_KEY from environment.")
        model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
        st.divider()
        st.write("Without an API key, the app uses a deterministic local fallback so the workflow still runs.")

    st.subheader("Example Library")
    samples = load_samples()
    sample_message = ""
    if samples.empty:
        st.info("No sample messages were found. Write your own message below.")
    else:
        example_sections = ["Write my own message"] + samples["section"].drop_duplicates().tolist()
        selected_section = st.selectbox("Choose an example section", example_sections)
        if selected_section != "Write my own message":
            section_samples = samples[samples["section"] == selected_section]
            selected_label = st.selectbox("Choose an example request", section_samples["label"].tolist())
            sample_row = section_samples.loc[section_samples["label"] == selected_label].iloc[0]
            sample_message = str(sample_row["message"])
            st.caption(f"Loaded from: {selected_section} / {selected_label}")

    message = st.text_area(
        "Incoming message",
        value=sample_message,
        height=180,
        placeholder="Paste a support-style message here...",
    )

    submitted = st.button("Analyze Message", type="primary")
    if submitted:
        if not message.strip():
            st.error("Please enter a message to analyze.")
            return

        orchestrator = InboxTriageOrchestrator(
            api_key=api_key,
            model=model,
            knowledge_path=DATA_DIR / "faq.json",
            log_path=ROOT / "logs" / "app_logs.csv",
        )
        with st.spinner("Running classifier, retriever, generator, and checker..."):
            result = orchestrator.run(message)
        render_result(result.to_dict())


def render_result(result: dict) -> None:
    classification = result["classification"]
    draft = result["draft"]
    check = result["check"]
    docs = result["retrieved_documents"]

    st.success("Analysis complete. Result was logged for review.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Category", classification["category"])
    metric_cols[1].metric("Urgency", classification["urgency"])
    metric_cols[2].metric("Confidence", f"{classification['confidence']:.0%}")
    metric_cols[3].metric("Human Review", "Yes" if check["needs_human_review"] else "No")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Summary")
        st.write(draft["summary"])
        st.subheader("Suggested Internal Action")
        st.info(draft["suggested_action"])
        st.subheader("Quality Check")
        st.write(f"**Evidence coverage:** {check['evidence_coverage']}")
        st.write(f"**Professionalism:** {check['professionalism']}")
        st.write(f"**Review reason:** {check['review_reason']}")
        if check["issues"]:
            st.write("**Issues:**")
            for issue in check["issues"]:
                st.warning(issue)

    with right:
        st.subheader("Draft Reply")
        st.text_area("Editable draft", value=draft["draft_reply"], height=240)
        st.subheader("Retrieved Knowledge")
        if docs:
            for doc in docs:
                with st.expander(f"{doc['title']} ({doc['score']:.2f})"):
                    st.write(doc["text"])
                    st.caption(f"ID: {doc['id']} | Category: {doc['category']}")
        else:
            st.warning("No knowledge base records were retrieved.")

    with st.expander("Raw JSON result"):
        st.json(result)
        st.download_button(
            "Download result JSON",
            data=json.dumps(result, indent=2),
            file_name="triage_result.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
