"""RiskRadar - Narrative Risk Triage Streamlit App.

Screens:
A) Entity selection + match/confidence overview
B) Narrative list ranked by risk score + driver tags
C) Narrative detail view (summary, score, drivers, evidence posts)
D) Feedback capture (entity corrections, risk ratings)
"""

import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_all
from src.entity_resolution import (
    resolve_all_posts,
    get_posts_for_entity,
    get_entity_confidence_stats,
)
from src.narrative_clustering import cluster_narratives
from src.risk_scoring import score_all_narratives, get_evidence_posts
from src.feedback import save_feedback, load_feedback
from src.config import USE_LLM, RISK_TAXONOMY, RISK_WEIGHTS, OUTPUT_DIR

CACHE_PATH = OUTPUT_DIR / "resolved_posts.pkl"

st.set_page_config(
    page_title="RiskRadar - Narrative Risk Triage",
    page_icon="🔍",  # keeping emoji only in page icon config
    layout="wide",
)


@st.cache_data(show_spinner="Loading and processing data...")
def load_and_resolve():
    """Load data from cache if available, otherwise process from scratch."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        return cached["posts"], cached["authors"], cached["entities"], cached["lookup"]

    posts, authors, entities_df, entity_lookup = load_all()
    posts = resolve_all_posts(posts, entity_lookup, use_llm=USE_LLM)
    return posts, authors, entities_df, entity_lookup


@st.cache_data(show_spinner="Clustering narratives...")
def get_narratives(_posts_for_entity, entity_id, entity_name):
    """Cluster and score narratives for an entity (cached)."""
    narratives = cluster_narratives(
        _posts_for_entity, entity_id, entity_name, use_llm=USE_LLM
    )
    return score_all_narratives(narratives)


def render_risk_badge(score: float, level: str) -> str:
    """Return colored risk badge HTML."""
    colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#28a745",
    }
    color = colors.get(level, "#6c757d")
    return f'<span style="background-color:{color};color:white;padding:2px 10px;border-radius:12px;font-weight:bold;">{score:.0f} - {level.upper()}</span>'


def main():
    st.title("RiskRadar - Narrative Risk Triage")
    st.caption(
        "Entity resolution, narrative clustering, and explainable risk scoring from social signals"
    )

    # Show LLM mode
    mode = "Bedrock (Claude)" if USE_LLM else "Local-only (no LLM)"
    st.sidebar.markdown(f"**Mode:** {mode}")
    st.sidebar.markdown("---")

    # Load data
    posts, authors, entities_df, entity_lookup = load_and_resolve()

    # ── SCREEN A: Entity Selection ──
    st.sidebar.header("Entity Selection")
    entity_options = {
        row["entity_id"]: f"{row['canonical_name']} ({row['entity_type']})"
        for _, row in entities_df.iterrows()
    }
    selected_entity = st.sidebar.selectbox(
        "Choose an entity",
        options=list(entity_options.keys()),
        format_func=lambda x: entity_options[x],
    )

    entity_name = entity_lookup[selected_entity]["canonical_name"]

    # Entity overview
    entity_posts = get_posts_for_entity(posts, selected_entity)
    stats = get_entity_confidence_stats(posts, selected_entity)

    st.header(f"Entity: {entity_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched Posts", stats["total"])
    col2.metric("High Confidence", stats["high"])
    col3.metric("Medium Confidence", stats["medium"])
    col4.metric("Low Confidence", stats["low"])

    if stats["total"] > 0:
        # Confidence distribution chart
        col_chart, col_methods = st.columns(2)

        with col_chart:
            conf_data = pd.DataFrame(
                {
                    "Confidence Level": ["High (>=0.9)", "Medium (0.7-0.9)", "Low (<0.7)"],
                    "Count": [stats["high"], stats["medium"], stats["low"]],
                }
            )
            fig = px.bar(
                conf_data,
                x="Confidence Level",
                y="Count",
                color="Confidence Level",
                color_discrete_map={
                    "High (>=0.9)": "#28a745",
                    "Medium (0.7-0.9)": "#ffc107",
                    "Low (<0.7)": "#dc3545",
                },
                title="Match Confidence Distribution",
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_methods:
            if stats["methods"]:
                method_data = pd.DataFrame(
                    list(stats["methods"].items()),
                    columns=["Method", "Count"],
                )
                fig = px.pie(
                    method_data,
                    values="Count",
                    names="Method",
                    title="Resolution Methods",
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No posts matched for this entity.")
        return

    st.markdown("---")

    # ── SCREEN B: Narrative List ──
    st.header("Narratives Ranked by Risk")

    narratives = get_narratives(entity_posts, selected_entity, entity_name)

    if not narratives:
        st.info("No narratives found. Not enough posts to form clusters.")
        return

    # Scoring methodology expander
    with st.sidebar.expander("Scoring Methodology"):
        st.markdown("**Risk Score = Weighted Sum of Drivers**")
        for driver, weight in RISK_WEIGHTS.items():
            st.markdown(f"- {driver.replace('_', ' ').title()}: **{weight*100:.0f}%**")
        st.markdown(
            "\nEach driver scored 0-100, then weighted. "
            "Language risk has highest weight because content determines reputational harm."
        )

    for i, narrative in enumerate(narratives):
        risk_badge = render_risk_badge(narrative["risk_score"], narrative["risk_level"])
        driver_tags = " | ".join(
            f"**{d['driver']}**: {d['score']:.0f}"
            for d in narrative["risk_drivers"][:3]
        )

        with st.expander(
            f"#{i+1} — {narrative['title']} ({narrative['post_count']} posts)",
            expanded=(i == 0),
        ):
            st.markdown(risk_badge, unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {narrative['risk_confidence']}")
            st.markdown(f"**Top drivers:** {driver_tags}")

            # Taxonomy labels
            if narrative["taxonomy_labels"]:
                labels = [
                    RISK_TAXONOMY.get(l, l) for l in narrative["taxonomy_labels"]
                ]
                st.markdown(
                    f"**Risk categories:** {', '.join(labels)}"
                )

            # ── SCREEN C: Narrative Detail (inside expander) ──
            st.markdown("---")
            st.subheader("Score Breakdown")

            # Driver breakdown table
            driver_df = pd.DataFrame(
                [
                    {
                        "Driver": d["driver"],
                        "Raw Score": d["score"],
                        "Weight": f"{d['weight']*100:.0f}%",
                        "Contribution": d["weighted_contribution"],
                        "Detail": d["explanation"],
                    }
                    for d in narrative["risk_drivers"]
                ]
            )
            st.dataframe(driver_df, use_container_width=True, hide_index=True)

            # Driver visualization
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[d["driver"] for d in narrative["risk_drivers"]],
                    y=[d["weighted_contribution"] for d in narrative["risk_drivers"]],
                    marker_color=[
                        "#dc3545" if d["weighted_contribution"] > 15
                        else "#ffc107" if d["weighted_contribution"] > 8
                        else "#28a745"
                        for d in narrative["risk_drivers"]
                    ],
                    text=[f"{d['weighted_contribution']:.1f}" for d in narrative["risk_drivers"]],
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Risk Driver Contributions",
                yaxis_title="Weighted Score Contribution",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Evidence posts
            st.subheader("Evidence Posts")
            evidence = get_evidence_posts(narrative, top_n=5)
            for _, post in evidence.iterrows():
                with st.container():
                    pcol1, pcol2 = st.columns([3, 1])
                    with pcol1:
                        st.markdown(f"**{post.get('platform', 'unknown')}** — {post.get('created_at', '')}")
                        st.text(post["text"][:500])
                    with pcol2:
                        st.markdown(
                            f"Likes: {post.get('likes', 0)} | "
                            f"Shares: {post.get('shares', 0)} | "
                            f"Comments: {post.get('comments', 0)}"
                        )
                        if post.get("followers", 0) > 0:
                            st.markdown(f"Author followers: {post['followers']:,}")
                        if post.get("url"):
                            st.markdown(f"[View post]({post['url']})")

                    # ── SCREEN D: Feedback per post ──
                    feedback_key = f"fb_{narrative['narrative_id']}_{post['post_id']}"
                    with st.popover("Flag issue"):
                        fb_type = st.selectbox(
                            "Issue type",
                            ["Incorrect entity match", "Risk too high", "Risk too low"],
                            key=f"type_{feedback_key}",
                        )
                        correct_entity = None
                        if fb_type == "Incorrect entity match":
                            correct_entity = st.selectbox(
                                "Correct entity (or none)",
                                ["none"] + list(entity_options.keys()),
                                format_func=lambda x: "None" if x == "none" else entity_options.get(x, x),
                                key=f"ent_{feedback_key}",
                            )
                        if st.button("Submit", key=f"btn_{feedback_key}"):
                            save_feedback(
                                feedback_type="entity_correction" if fb_type == "Incorrect entity match" else "risk_rating",
                                entity_id=selected_entity,
                                narrative_id=narrative["narrative_id"],
                                post_id=str(post["post_id"]),
                                details={
                                    "issue": fb_type,
                                    "correct_entity": correct_entity,
                                },
                            )
                            st.success("Feedback saved!")

                    st.markdown("---")

            # Narrative-level feedback
            st.subheader("Narrative Feedback")
            narr_fb_key = f"narr_fb_{narrative['narrative_id']}"
            risk_rating = st.slider(
                "Your risk assessment (0-100)",
                0, 100,
                value=int(narrative["risk_score"]),
                key=f"slider_{narr_fb_key}",
            )
            fb_notes = st.text_input("Notes (optional)", key=f"notes_{narr_fb_key}")
            if st.button("Submit Narrative Feedback", key=f"btn_{narr_fb_key}"):
                save_feedback(
                    feedback_type="narrative_feedback",
                    entity_id=selected_entity,
                    narrative_id=narrative["narrative_id"],
                    details={
                        "system_score": narrative["risk_score"],
                        "user_score": risk_rating,
                        "notes": fb_notes,
                    },
                )
                st.success("Narrative feedback saved!")

    # Sidebar: feedback log
    st.sidebar.markdown("---")
    st.sidebar.header("Feedback Log")
    feedback_entries = load_feedback()
    st.sidebar.metric("Total Feedback Entries", len(feedback_entries))
    if feedback_entries:
        with st.sidebar.expander("View Recent Feedback"):
            for entry in feedback_entries[-5:]:
                st.sidebar.json(entry)

    # Audit trail info
    st.sidebar.markdown("---")
    st.sidebar.header("Audit Trail")
    st.sidebar.markdown(
        f"""
        - **Posts processed:** {len(posts)}
        - **Entity:** {entity_name}
        - **Matched posts:** {stats['total']}
        - **Narratives found:** {len(narratives)}
        - **Avg confidence:** {stats['avg_confidence']}
        - **Feedback file:** `feedback.jsonl`
        """
    )


if __name__ == "__main__":
    main()
