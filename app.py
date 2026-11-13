import json
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Learner Analytics Dashboard", page_icon="📊", layout="wide")

HESITATION = ["uh", "umm", "actually", "like", "sorry", "for example"]
DATA = json.loads((Path(__file__).parent / "data.json").read_text(encoding="utf-8"))
OBS, CARDS, TEXT = DATA["obs"], DATA["cards"], DATA["text"]
STUDENTS = sorted(OBS, key=lambda s: int(s.split()[1]))


# ---------- helpers ----------
@st.cache_resource
def get_analyzer():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def split_keywords(value):
    return [w.strip().lower() for w in str(value).replace(",", ";").split(";") if w.strip()]


def analyze(text, high_conf, low_conf):
    t = text.lower()
    high = sum(1 for w in high_conf if w in t)
    low = sum(1 for w in low_conf if w in t)
    total = high + low
    confidence = round((high / total) * 10, 1) if total > 0 else 5.0
    hes = sum(t.count(w) for w in HESITATION)
    load = "High" if hes > 5 else "Moderate" if hes > 2 else "Low"
    s = get_analyzer().polarity_scores(text)
    overall = "Positive" if s["compound"] >= 0.05 else "Negative" if s["compound"] <= -0.05 else "Neutral"
    return {"high": high, "low": low, "confidence": confidence, "hes": hes, "load": load,
            "sentiment": overall, "compound": round(s["compound"], 2), "pos": s["pos"], "neg": s["neg"], "neu": s["neu"]}


def radar(student):
    o = OBS[student]
    cats = ["Vocal Confidence", "Processing Flow", "Linguistic Precision"]
    vals = [o["vocal_total"] + 2, o["flow_total"] + 2, o["ling_total"] + 2]
    ang = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    ax.fill(ang + ang[:1], vals + vals[:1], color="#1f77b4", alpha=0.4)
    ax.plot(ang + ang[:1], vals + vals[:1], color="#1f77b4", linewidth=2.5)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels(cats, fontsize=9, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 8)
    ax.grid(True, color="grey", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def text_charts(r, title):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    axes[0].bar(["High", "Low"], [r["high"], r["low"]], color=["#2ecc71", "#e74c3c"], width=0.4)
    axes[0].set_title(f"Confidence\n{r['confidence']}/10")
    axes[1].bar(["Hesitation words"], [r["hes"]], color=["#e67e22"], width=0.4)
    axes[1].set_title(f"Cognitive load\n{r['load']}")
    axes[2].bar(["Positive", "Negative", "Neutral"], [r["pos"], r["neg"], r["neu"]],
                color=["#2ecc71", "#e74c3c", "#3498db"], width=0.4)
    axes[2].set_title(f"Sentiment\n{r['sentiment']}")
    plt.tight_layout()
    return fig


def overview_df():
    rows = []
    for s in STUDENTS:
        t = TEXT.get(s)
        rows.append({
            "Student": s,
            "Vocal": OBS[s]["vocal_total"], "Flow": OBS[s]["flow_total"], "Linguistic": OBS[s]["ling_total"],
            "Confidence (/10)": t["confidence"] if t else None,
            "Cognitive load": t["cognitive_load"] if t else "no transcript",
            "Sentiment": t["sentiment"] if t else "no transcript",
        })
    return pd.DataFrame(rows)


# ---------- page ----------
st.title("Learner Analytics Dashboard")
st.caption("Live observation (structured data) + text analysis and AI profiling (unstructured data) for 19 learners of a mentoring programme. "
           "Names are anonymized as Student 1 to 19. Shared with the organisation's permission.")

tab_over, tab_one, tab_try, tab_about = st.tabs(["Overview", "Learner profile", "Try it yourself", "About"])

with tab_over:
    df = overview_df()
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Sentiment (18 learners with transcripts)")
    counts = df[df["Sentiment"] != "no transcript"]["Sentiment"].value_counts()
    s1, s2, s3 = st.columns(3)
    s1.metric("Positive", int(counts.get("Positive", 0)))
    s2.metric("Neutral", int(counts.get("Neutral", 0)))
    s3.metric("Negative", int(counts.get("Negative", 0)))

    st.subheader("Confidence score per learner (/10)")
    conf = df.dropna(subset=["Confidence (/10)"])
    fig_c, ax_c = plt.subplots(figsize=(10, 3.2))
    ax_c.bar([int(s.split()[1]) for s in conf["Student"]], conf["Confidence (/10)"], color="#2e86de")
    ax_c.set_xlabel("Student number")
    ax_c.set_ylabel("Confidence (/10)")
    ax_c.set_ylim(0, 10)
    ax_c.set_xticks([int(s.split()[1]) for s in conf["Student"]])
    plt.tight_layout()
    st.pyplot(fig_c)
    plt.close(fig_c)

with tab_one:
    student = st.selectbox("Choose a learner", STUDENTS)
    o = OBS[student]
    m1, m2, m3 = st.columns(3)
    m1.metric("Vocal", o["vocal_total"])
    m2.metric("Flow", o["flow_total"])
    m3.metric("Linguistic", o["ling_total"])

    t_card, t_obs, t_text = st.tabs(["AI profile card", "Live observation", "Text analysis"])

    with t_card:
        card = CARDS.get(student, {})
        if card.get("education"):
            st.markdown(f"**Education:** {card['education']}")
        if card.get("org_feedback"):
            st.info(card["org_feedback"])
        cols = st.columns(2)
        for i, (title, items) in enumerate(card.get("sections", {}).items()):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                for it in items:
                    st.markdown(f"- {it}")

    with t_obs:
        a, b = st.columns([1, 1])
        with a:
            fig_r = radar(student)
            st.pyplot(fig_r)
            plt.close(fig_r)
        with b:
            st.markdown("**Score on each part (-2 to +2)**")
            for group, key, lab in [("Vocal", "vocal_detail", "vocal_label"),
                                    ("Flow", "flow_detail", "flow_label"),
                                    ("Linguistic", "ling_detail", "ling_label")]:
                st.markdown(f"{group}: *{o[lab]}*")
                st.write(", ".join(f"{k} {v}" for k, v in o[key].items()))

    with t_text:
        t = TEXT.get(student)
        if not t:
            st.warning("No transcript analysis for this learner (transcript not available).")
        else:
            x1, x2, x3 = st.columns(3)
            x1.metric("Confidence", f"{t['confidence']}/10")
            x2.metric("Cognitive load", f"{t['cognitive_load']} ({t['hesitation_words']} hesitation words)")
            x3.metric("Sentiment", f"{t['sentiment']} ({t['compound']})")
            r = {"high": t["high"], "low": t["low"], "confidence": t["confidence"],
                 "hes": t["hesitation_words"], "load": t["cognitive_load"], "sentiment": t["sentiment"],
                 "pos": t["pos"], "neg": t["neg"], "neu": t["neu"]}
            fig_t = text_charts(r, f"{student} - interaction analysis")
            st.pyplot(fig_t)
            plt.close(fig_t)
            st.caption("High-confidence keywords: " + ", ".join(t["high_conf"]))
            st.caption("Low-confidence keywords: " + ", ".join(t["low_conf"]))

with tab_try:
    st.write("Paste any English text and see the same analysis. Nothing is stored.")
    text = st.text_area("Text", value="Yes, I can do that. Actually, umm, I am not sure about the second part, but I will definitely try my best.", height=140)
    c1, c2 = st.columns(2)
    high = c1.text_input("High-confidence keywords (separate with ;)", "yes; sure; definitely; i can; good; okay")
    low = c2.text_input("Low-confidence keywords (separate with ;)", "maybe; not sure; sorry; uh; umm; problem")
    if st.button("Analyse", type="primary") and text.strip():
        r = analyze(text, split_keywords(high), split_keywords(low))
        y1, y2, y3 = st.columns(3)
        y1.metric("Confidence", f"{r['confidence']}/10")
        y2.metric("Cognitive load", f"{r['load']} ({r['hes']} hesitation words)")
        y3.metric("Sentiment", f"{r['sentiment']} ({r['compound']})")
        fig_y = text_charts(r, "Interaction analysis")
        st.pyplot(fig_y)
        plt.close(fig_y)

with tab_about:
    st.markdown(
        "**Structured data:** team members scored learners live from -2 to +2 on vocal, linguistic and interaction-flow dimensions. "
        "\n\n"
        "**Unstructured data:** session transcripts analysed with NLTK VADER (sentiment), keyword-based confidence score (/10) and "
        "cognitive load from hesitation words. A Groq LLM wrote the profile cards.\n\n"
        "- Confidence = high-confidence words found / all keywords found x 10 (5.0 if none).\n"
        "- Cognitive load: hesitation words (uh, umm, actually, like, sorry, for example). Low 0-2, Moderate 3-5, High above 5.\n"
        "- Text analysis covers 18 learners (one learner has no transcript). Profile cards and observation cover all 19.\n\n"
        "Code and notebooks: see the GitHub repository."
    )
