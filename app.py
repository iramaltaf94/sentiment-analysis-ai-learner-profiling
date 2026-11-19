import json
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.fill(ang + ang[:1], vals + vals[:1], color="#1f77b4", alpha=0.4)
    ax.plot(ang + ang[:1], vals + vals[:1], color="#1f77b4", linewidth=2.5)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels(cats, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 8)
    ax.set_yticks(range(1, 9))
    ax.set_yticklabels([])
    ax.grid(True, color="grey", linestyle="--", alpha=0.4)
    ax.set_title(f"Neural profile: {student}", fontsize=11, fontweight="bold", pad=22)
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


# ---------- crystal 3D ----------
CRYSTAL_PALETTE = ["rgb(34,211,238)", "rgb(99,102,241)", "rgb(168,85,247)", "rgb(236,72,153)"]
GOLD = "rgb(255,196,61)"


def crystal_3d(df, selected=None, height=760):
    """Big, glassy, faceted crystals (one per learner) on a dark glowing stage."""
    cols = ["Vocal", "Flow", "Linguistic"]
    lo = float(df[cols].min().min())
    hi = float(df[cols].max().max())
    span = max(hi - lo, 1.0)
    rng = [lo - span * 0.2, hi + span * 0.2]   # same range on all axes -> crystals are not stretched
    base = span * 0.06                           # crystal size (bigger = more visible)

    score = df[cols].mean(axis=1)
    norm = (score - score.min()) / (score.max() - score.min()) if score.max() > score.min() else score * 0 + 0.5
    colors = px.colors.sample_colorscale(CRYSTAL_PALETTE, norm.tolist())

    fig = go.Figure()
    ex, ey, ez = [], [], []                      # all facet edges in one trace (fast)
    hx, hy, hz, hcol, hsize, htext = [], [], [], [], [], []

    for (_, row), col in zip(df.iterrows(), colors):
        is_sel = selected is not None and row["Student"] == selected
        s = base * (1.8 if is_sel else 1.0)
        if is_sel:
            col = GOLD
        x, y, z = float(row["Vocal"]), float(row["Flow"]), float(row["Linguistic"])

        # elongated octahedron = diamond crystal
        vx = [x + s, x - s, x, x, x, x]
        vy = [y, y, y + s, y - s, y, y]
        vz = [z, z, z, z, z + 1.6 * s, z - 1.6 * s]
        fi, fj, fk = [], [], []
        for a in (0, 1):
            for b in (2, 3):
                for c in (4, 5):
                    fi.append(a); fj.append(b); fk.append(c)
        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz, i=fi, j=fj, k=fk,
            color=col, opacity=0.85 if is_sel else 0.6, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.9, specular=1.6, roughness=0.05, fresnel=1.4),
            lightposition=dict(x=300, y=200, z=400),
            hoverinfo="skip", showlegend=False,
        ))
        for a, b in [(a, b) for a in (0, 1) for b in (2, 3)] + \
                    [(a, c) for a in (0, 1) for c in (4, 5)] + \
                    [(b, c) for b in (2, 3) for c in (4, 5)]:
            ex += [vx[a], vx[b], None]; ey += [vy[a], vy[b], None]; ez += [vz[a], vz[b], None]

        hx.append(x); hy.append(y); hz.append(z)
        hcol.append(col)
        hsize.append(46 if is_sel else 30)
        htext.append(row["Student"])

    # glowing facet edges
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="rgba(255,255,255,0.6)", width=2),
                               hoverinfo="skip", showlegend=False))
    # soft glow halo (also the easy hover target)
    fig.add_trace(go.Scatter3d(
        x=hx, y=hy, z=hz, mode="markers", text=htext,
        marker=dict(size=hsize, color=hcol, opacity=0.14, line=dict(width=0)),
        hovertemplate="<b>%{text}</b><br>Vocal: %{x}<br>Flow: %{y}<br>Linguistic: %{z}<extra></extra>",
        showlegend=False,
    ))
    # bright core sparkle
    fig.add_trace(go.Scatter3d(x=hx, y=hy, z=hz, mode="markers",
                               marker=dict(size=4, color="white", opacity=0.95),
                               hoverinfo="skip", showlegend=False))

    def axis(title):
        return dict(title=dict(text=title, font=dict(color="#c7d2fe", size=14)), range=rng,
                    showbackground=True, backgroundcolor="rgba(30,35,90,0.35)",
                    gridcolor="rgba(160,170,255,0.22)", zerolinecolor="rgba(160,170,255,0.4)",
                    tickfont=dict(color="#a5b4fc"))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b1026",
        scene=dict(xaxis=axis("Vocal"), yaxis=axis("Flow"), zaxis=axis("Linguistic"),
                   aspectmode="cube", dragmode="orbit", bgcolor="#0b1026",
                   camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))),
        hoverlabel=dict(bgcolor="#1e1b4b", font=dict(color="white", size=14)),
    )
    return fig


def show_crystal(fig):
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["resetCameraLastSave3d"]})


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

    st.subheader("3D view: Vocal, Flow and Linguistic")
    show_crystal(crystal_3d(df))
    st.caption("Each crystal is one learner. Drag to rotate, scroll to zoom, hover for scores. "
               "Colour shows the average score (cyan = lower, pink = higher). Scores only, no grouping.")

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

        st.markdown("**3D view: where this learner sits among all learners**")
        show_crystal(crystal_3d(overview_df(), selected=student, height=720))
        st.caption("The golden, bigger crystal is the selected learner. Drag to rotate, scroll to zoom. "
                   "Scores only; no grouping or labels are applied.")

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
