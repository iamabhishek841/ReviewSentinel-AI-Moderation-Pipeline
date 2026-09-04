from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import METRICS_PATH, PROCESSED_PATH, QUEUE_PATH, run_pipeline

st.set_page_config(page_title="ReviewSentinel - Product Abuse Investigation", layout="wide")

REQUIRED_PROCESSED = {"case_id", "risk_score", "risk_tier", "risk_reasons", "ground_truth_abuse"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    needs_refresh = True
    if PROCESSED_PATH.exists() and QUEUE_PATH.exists() and METRICS_PATH.exists():
        probe = pd.read_csv(PROCESSED_PATH, nrows=5)
        needs_refresh = not REQUIRED_PROCESSED.issubset(probe.columns)

    if needs_refresh:
        with st.spinner("Building synthetic trust & safety investigation dataset..."):
            return run_pipeline()

    processed = pd.read_csv(PROCESSED_PATH)
    queue = pd.read_csv(QUEUE_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return processed, queue, metrics


df, queue, metrics = load_data()
df["timestamp"] = pd.to_datetime(df["timestamp"])

st.title("ReviewSentinel")
st.caption(
    "Synthetic Product Abuse & Trust Investigation System - behavioural signals, "
    "case triage, explainable risk scoring, and detection evaluation."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Reviews", f"{len(df):,}")
m2.metric("Accounts", f"{queue['user_id'].nunique():,}")
m3.metric("High-risk cases", int((queue["risk_tier"] == "high").sum()))
m4.metric("Precision", f"{metrics['precision']:.1%}")
m5.metric("Recall", f"{metrics['recall']:.1%}")

st.info(
    "This project uses synthetic data with known ground truth. Risk scores are deterministic and "
    "explainable; they are designed for investigation prioritisation, not automatic enforcement."
)

overview_tab, queue_tab, case_tab, eval_tab = st.tabs(
    ["Overview", "Investigation Queue", "Case Drill-down", "Evaluation"]
)

with overview_tab:
    st.subheader("Platform activity")
    daily = (
        df.assign(day=df["timestamp"].dt.date)
        .groupby("day")
        .agg(reviews=("review_id", "count"), violating_reviews=("violation_flag", "sum"))
    )
    st.line_chart(daily)

    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        risk_counts = queue["risk_tier"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
        st.bar_chart(risk_counts)
    with right:
        st.subheader("Top investigation reasons")
        reason_rows = []
        for reasons in queue.loc[queue["risk_tier"] != "low", "risk_reasons"]:
            reason_rows.extend([r.strip() for r in str(reasons).split(";") if r.strip()])
        reason_counts = pd.Series(reason_rows).value_counts().head(10)
        st.dataframe(reason_counts.rename("cases").to_frame(), use_container_width=True)

with queue_tab:
    st.subheader("Prioritised investigation queue")
    c1, c2 = st.columns(2)
    selected_tiers = c1.multiselect(
        "Risk tier",
        ["high", "medium", "low"],
        default=["high", "medium"],
    )
    abuse_types = sorted(queue["ground_truth_abuse_type"].dropna().unique().tolist())
    selected_types = c2.multiselect("Synthetic scenario (evaluation only)", abuse_types, default=[])

    filtered = queue[queue["risk_tier"].isin(selected_tiers)].copy()
    if selected_types:
        filtered = filtered[filtered["ground_truth_abuse_type"].isin(selected_types)]

    columns = [
        "case_id",
        "user_id",
        "risk_score",
        "risk_tier",
        "risk_reasons",
        "review_count",
        "max_reviews_1h",
        "duplicate_ratio",
        "violation_rate",
        "url_rate",
    ]
    display_queue = filtered[columns].copy()
    for col in ["duplicate_ratio", "violation_rate", "url_rate"]:
        display_queue[col] = display_queue[col].map(lambda value: f"{value:.1%}")
    st.dataframe(
        display_queue,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(min_value=0, max_value=100),
        },
    )

with case_tab:
    st.subheader("Case evidence and analyst triage")
    case_options = queue.sort_values("risk_score", ascending=False)["case_id"].tolist()
    selected_case = st.selectbox("Select case", case_options)
    case = queue.loc[queue["case_id"] == selected_case].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", int(case["risk_score"]))
    c2.metric("Risk tier", str(case["risk_tier"]).upper())
    c3.metric("Reviews", int(case["review_count"]))
    c4.metric("Max reviews/hour", int(case["max_reviews_1h"]))

    st.markdown(f"**Why this case was surfaced:** {case['risk_reasons']}")

    account_reviews = df[df["user_id"] == case["user_id"]].sort_values("timestamp")
    timeline_cols = [
        "timestamp",
        "review_id",
        "product_id",
        "review",
        "violation_type",
        "device_id",
        "ip_cluster",
    ]
    st.dataframe(account_reviews[timeline_cols], use_container_width=True, hide_index=True)

    if "case_dispositions" not in st.session_state:
        st.session_state.case_dispositions = {}

    b1, b2, b3 = st.columns(3)
    if b1.button("Dismiss", use_container_width=True):
        st.session_state.case_dispositions[selected_case] = "dismissed"
    if b2.button("Escalate", use_container_width=True):
        st.session_state.case_dispositions[selected_case] = "escalated"
    if b3.button("Confirm abuse", use_container_width=True):
        st.session_state.case_dispositions[selected_case] = "confirmed_abuse"

    disposition = st.session_state.case_dispositions.get(selected_case, "needs_review")
    st.caption(f"Session-only analyst disposition: {disposition}")

with eval_tab:
    st.subheader("Detection quality on synthetic ground truth")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Precision", f"{metrics['precision']:.1%}")
    e2.metric("Recall", f"{metrics['recall']:.1%}")
    e3.metric("F1", f"{metrics['f1']:.1%}")
    e4.metric(f"Precision@{metrics['top_k']}", f"{metrics['precision_at_k']:.1%}")

    confusion = pd.DataFrame(
        {
            "Predicted abuse": [metrics["true_positives"], metrics["false_positives"]],
            "Predicted benign": [metrics["false_negatives"], metrics["true_negatives"]],
        },
        index=["Actual abuse", "Actual benign"],
    )
    st.dataframe(confusion, use_container_width=True)
    st.caption(
        "Thresholds are intentionally transparent. In a real trust & safety system, the operating "
        "point would be calibrated against investigation capacity and the cost of false positives "
        "versus missed abuse."
    )
