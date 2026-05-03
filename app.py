from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.orchestrator import MusicRecommendationAgent
from src.recommender import load_songs, unique_labels


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


@st.cache_data
def load_samples() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_queries.csv")


@st.cache_data
def load_catalog() -> list[dict]:
    return load_songs(str(DATA_DIR / "songs.csv"))


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="MoodMap Music Recommender", layout="wide")
    st.title("MoodMap Music Recommender")
    st.caption(
        "A Module 3 music recommender upgraded with retrieval, an external LLM option, guardrails, confidence scoring, and evaluation."
    )

    songs = load_catalog()
    genres = ["Auto"] + unique_labels(songs, "genre")
    moods = ["Auto"] + unique_labels(songs, "mood")
    env_key = os.getenv("OPENAI_API_KEY", "")

    with st.sidebar:
        st.header("Controls")
        count = st.slider("Recommendations", min_value=3, max_value=8, value=5)
        diversity = st.slider("Diversity", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        selected_genre = st.selectbox("Genre override", genres)
        selected_mood = st.selectbox("Mood override", moods)
        target_energy = st.slider("Energy override", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
        use_energy = st.checkbox("Use energy override", value=False)
        likes_acoustic = st.checkbox("Prefer acoustic texture", value=False)
        use_acoustic = st.checkbox("Use acoustic override", value=False)

    samples = load_samples()
    left, right = st.columns([0.45, 0.55])
    with left:
        st.subheader("Request")
        st.markdown("**External LLM Settings**")
        api_key = st.text_input(
            "OpenAI API key",
            value="",
            type="password",
            help="Optional. If provided, the app uses OpenAI for profile refinement and grounded explanations.",
        )
        if not api_key and env_key:
            api_key = env_key
            st.success("Using OPENAI_API_KEY from environment.")
        model = st.text_input(
            "OpenAI model",
            value=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        )
        use_llm = st.checkbox("Use external LLM", value=bool(api_key))
        if use_llm and not api_key:
            st.warning("Add an API key to use the LLM path. The app will fall back locally.")

        sections = ["Write my own"] + samples["section"].drop_duplicates().tolist()
        selected_section = st.selectbox("Example group", sections, key="example_group")
        sample_query = ""
        if selected_section != "Write my own":
            section_samples = samples[samples["section"] == selected_section]
            labels = section_samples["label"].tolist()
            if labels:
                label = st.selectbox("Example", labels, key=f"example_label_{selected_section}")
                matching_samples = section_samples[section_samples["label"] == label]
                if matching_samples.empty:
                    sample_query = str(section_samples.iloc[0]["query"])
                else:
                    sample_query = str(matching_samples.iloc[0]["query"])
            else:
                st.info("No examples are available in this group.")

        query = st.text_area(
            "Music request",
            value=sample_query,
            height=170,
            key=f"query_{selected_section}_{sample_query[:32]}",
            placeholder="Example: I need calm focus music for coding tonight.",
        )
        submitted = st.button("Recommend", type="primary")

    if not submitted:
        with right:
            st.subheader("Catalog")
            st.dataframe(pd.DataFrame(songs), hide_index=True, use_container_width=True)
        return

    explicit_preferences = {"diversity": diversity}
    if selected_genre != "Auto":
        explicit_preferences["favorite_genre"] = selected_genre
    if selected_mood != "Auto":
        explicit_preferences["favorite_mood"] = selected_mood
    if use_energy:
        explicit_preferences["target_energy"] = target_energy
    if use_acoustic:
        explicit_preferences["likes_acoustic"] = likes_acoustic

    agent = MusicRecommendationAgent(
        song_path=DATA_DIR / "songs.csv",
        knowledge_path=DATA_DIR / "music_knowledge.json",
        log_path=ROOT / "logs" / "recommendation_runs.jsonl",
        use_llm=use_llm,
        api_key=api_key,
        model=model,
    )
    with st.spinner("Planning, optionally calling the LLM, retrieving context, scoring songs, and checking reliability..."):
        result = agent.run(query, k=count, explicit_preferences=explicit_preferences)

    render_result(result.to_dict())


def render_result(result: dict) -> None:
    profile = result["profile"]
    self_check = result["self_check"]
    recommendations = result["recommendations"]
    docs = result["retrieved_documents"]

    st.divider()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Activity", profile["activity"])
    metric_cols[1].metric("Confidence", f"{result['overall_confidence']:.0%}")
    metric_cols[2].metric("LLM Used", "Yes" if result["llm_used"] else "No")
    metric_cols[3].metric("Retrieved Docs", len(docs))

    if result["llm_enabled"] and result["llm_error"]:
        st.warning("LLM fallback: " + result["llm_error"])
    if result["guardrail_flags"]:
        st.warning("Guardrail flags: " + ", ".join(result["guardrail_flags"]))
    if self_check["issues"]:
        for issue in self_check["issues"]:
            st.info(issue)

    left, right = st.columns([0.58, 0.42])
    with left:
        st.subheader("Recommendations")
        for index, rec in enumerate(recommendations, start=1):
            song = rec["song"]
            with st.expander(
                f"{index}. {song['title']} - {song['artist']} "
                f"({song['genre']}, {song['mood']})",
                expanded=index <= 3,
            ):
                score_cols = st.columns(3)
                score_cols[0].metric("Score", f"{rec['score']:.2f}")
                score_cols[1].metric("Confidence", f"{rec['confidence']:.0%}")
                score_cols[2].metric("Energy", f"{song['energy']:.2f}")
                st.write(rec["explanation"])
                if rec["evidence"]:
                    st.caption("Evidence: " + " | ".join(rec["evidence"]))

    with right:
        st.subheader("Agent Trace")
        st.write(
            f"**LLM mode**: {'enabled' if result['llm_enabled'] else 'disabled'}"
            + (f" using `{result['llm_model']}`" if result["llm_model"] else "")
        )
        st.write(f"**Human Review**: {'Yes' if self_check['needs_human_review'] else 'No'}")
        for step in result["plan_steps"]:
            st.write(f"**{step['name']}**: {step['details']}")

        st.subheader("Parsed Profile")
        st.json(profile)

        st.subheader("Retrieved Context")
        for doc in docs:
            with st.expander(f"{doc['title']} ({doc['kind']}, {doc['score']:.2f})"):
                st.write(doc["text"])

        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2),
            file_name="music_recommendation_result.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
