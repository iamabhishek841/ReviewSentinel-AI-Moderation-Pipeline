from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.abuse_features import add_behavioral_features
from pipeline.data_ingestion import generate_fake_reviews
from pipeline.evaluation import evaluate_detection
from pipeline.investigation import build_investigation_queue
from pipeline.sentiment_analyzer import analyze_sentiment
from pipeline.text_cleaning import preprocess_reviews
from pipeline.violation_detector import detect_violations

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "user_reviews.csv"
PROCESSED_PATH = DATA_DIR / "processed_reviews.csv"
QUEUE_PATH = DATA_DIR / "investigation_queue.csv"
METRICS_PATH = DATA_DIR / "evaluation_metrics.json"

RAW_REQUIRED_COLUMNS = {
    "review_id",
    "user_id",
    "review",
    "timestamp",
    "product_id",
    "device_id",
    "ip_cluster",
    "account_age_days",
    "ground_truth_abuse",
    "ground_truth_abuse_type",
}


def _load_or_regenerate_raw() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        existing = pd.read_csv(RAW_PATH)
        if RAW_REQUIRED_COLUMNS.issubset(existing.columns):
            return existing

    generated = generate_fake_reviews(num_reviews=3000, num_users=250, seed=42)
    generated.to_csv(RAW_PATH, index=False)
    return generated


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = _load_or_regenerate_raw()
    df = preprocess_reviews(df)
    df = detect_violations(df)
    df = analyze_sentiment(df)
    df = add_behavioral_features(df)

    queue = build_investigation_queue(df)
    metrics = evaluate_detection(queue, top_k=20)

    account_fields = queue[
        ["user_id", "case_id", "risk_score", "risk_tier", "risk_reasons", "case_status"]
    ]
    df = df.merge(account_fields, on="user_id", how="left")

    df.to_csv(PROCESSED_PATH, index=False)
    queue.to_csv(QUEUE_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return df, queue, metrics


def main() -> None:
    df, queue, metrics = run_pipeline()
    print(f"Processed {len(df):,} reviews across {queue['user_id'].nunique():,} accounts")
    print(
        "Detection metrics: "
        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, precision@{metrics['top_k']}={metrics['precision_at_k']:.3f}"
    )


if __name__ == "__main__":
    main()
