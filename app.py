"""
app.py
------
Streamlit frontend for the Mood Trajectory Journal.

Run with: streamlit run app.py

This is a personal-use demo tool. Journal entries are stored ONLY in memory for the
current session (nothing is saved to disk) unless you explicitly extend it to do so -
worth noting in your report's privacy/ethics discussion.
"""

import json
import os
import pandas as pd
import streamlit as st
from pipeline import MoodTrajectoryPipeline

st.set_page_config(page_title="Mood Trajectory Journal", page_icon="📓", layout="centered")

st.title("📓 Mood Trajectory Journal")
st.caption("Tracks patterns in your own journal language over time - not a diagnostic tool.")

st.info(
    "**This tool does not diagnose anything.** It shows patterns in your own writing "
    "over time (sentiment, self-focus, social references) as a self-reflection aid. "
    "If you're struggling, please reach out to a trusted person or professional - "
    "this tool is not a substitute for that.",
    icon="ℹ️"
)

if "entries" not in st.session_state:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_journal_entries.json")) as f:
        st.session_state["entries"] = json.load(f)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return MoodTrajectoryPipeline()


pipeline = load_pipeline()

with st.expander("Add a new journal entry"):
    entry_date = st.date_input("Date")
    entry_text = st.text_area("What's on your mind today?", height=120)
    if st.button("Add Entry"):
        if entry_text.strip():
            st.session_state["entries"].append({
                "date": str(entry_date),
                "text": entry_text.strip()
            })
            st.success("Entry added.")
        else:
            st.warning("Please write something first.")

with st.expander("View / reset journal entries"):
    for e in st.session_state["entries"]:
        st.write(f"**{e['date']}**: {e['text']}")
    if st.button("Reset to sample data"):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "data", "sample_journal_entries.json")) as f:
            st.session_state["entries"] = json.load(f)
        st.rerun()

st.divider()

if st.button("Analyze Trajectory", type="primary"):
    with st.spinner("Analyzing patterns..."):
        result = pipeline.run(st.session_state["entries"])

    st.subheader("Summary")
    st.write(result["narrative"])

    if result["suggest_support"]:
        st.warning(
            "If your recent entries genuinely reflect how you've been feeling, it might "
            "help to talk to someone you trust, or a mental health professional. You don't "
            "have to navigate this alone. If you're in the US, you can reach the 988 Suicide "
            "& Crisis Lifeline by calling or texting **988**, available 24/7.",
            icon="💛"
        )

    st.divider()
    st.subheader("Trends Over Time")

    df = pd.DataFrame(result["timeseries"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    st.write("**Sentiment (-1 negative to +1 positive)**")
    st.line_chart(df["sentiment_compound"])

    st.write("**Self-focus (first-person singular word ratio)**")
    st.line_chart(df["first_person_singular_ratio"])

    st.write("**Social word ratio**")
    st.line_chart(df["social_word_ratio"])

    st.write("**Entry length (word count)**")
    st.line_chart(df["word_count"])

    with st.expander("Raw trend statistics"):
        st.json(result["analysis"]["trends"])
