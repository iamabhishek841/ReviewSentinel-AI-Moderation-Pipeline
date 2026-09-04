from __future__ import annotations

import pandas as pd


def _score_account(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if row["max_reviews_1h"] >= 4:
        score += 20
        reasons.append(f"burst activity: {int(row['max_reviews_1h'])} reviews/hour")
    if row["duplicate_ratio"] >= 0.45:
        score += 20
        reasons.append(f"repeated content: {row['duplicate_ratio']:.0%} duplicate ratio")
    if row["max_same_text_accounts"] >= 3:
        score += 20
        reasons.append(f"cross-account reuse: {int(row['max_same_text_accounts'])} accounts share text")
    if row["max_shared_device_accounts"] >= 3:
        score += 10
        reasons.append(f"shared device: {int(row['max_shared_device_accounts'])} accounts")
    if row["max_shared_ip_accounts"] >= 4:
        score += 10
        reasons.append(f"shared network cluster: {int(row['max_shared_ip_accounts'])} accounts")
    if row["violation_rate"] >= 0.20:
        score += 15
        reasons.append(f"policy signals: {row['violation_rate']:.0%} of reviews")
    if row["url_rate"] >= 0.35:
        score += 10
        reasons.append(f"link-heavy activity: {row['url_rate']:.0%} of reviews")
    if row["account_age_days"] <= 14:
        score += 5
        reasons.append(f"new account: {int(row['account_age_days'])} days old")

    return min(score, 100), reasons


def build_investigation_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review-level signals into one explainable case per account."""
    required = {
        "user_id",
        "account_age_days",
        "account_review_count",
        "max_reviews_1h",
        "duplicate_ratio",
        "same_text_accounts",
        "shared_device_accounts",
        "shared_ip_accounts",
        "violation_rate",
        "url_rate",
        "unique_products",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    aggregations = {
        "account_age_days": ("account_age_days", "first"),
        "review_count": ("account_review_count", "first"),
        "max_reviews_1h": ("max_reviews_1h", "first"),
        "duplicate_ratio": ("duplicate_ratio", "first"),
        "max_same_text_accounts": ("same_text_accounts", "max"),
        "max_shared_device_accounts": ("shared_device_accounts", "max"),
        "max_shared_ip_accounts": ("shared_ip_accounts", "max"),
        "violation_rate": ("violation_rate", "first"),
        "url_rate": ("url_rate", "first"),
        "unique_products": ("unique_products", "first"),
    }

    if "ground_truth_abuse" in df.columns:
        aggregations["ground_truth_abuse"] = ("ground_truth_abuse", "max")
    if "ground_truth_abuse_type" in df.columns:
        aggregations["ground_truth_abuse_type"] = ("ground_truth_abuse_type", "first")

    queue = df.groupby("user_id").agg(**aggregations).reset_index()

    scores = queue.apply(_score_account, axis=1)
    queue["risk_score"] = [value[0] for value in scores]
    queue["risk_reasons"] = ["; ".join(value[1]) if value[1] else "no elevated signals" for value in scores]
    queue["risk_tier"] = pd.cut(
        queue["risk_score"],
        bins=[-1, 34, 54, 100],
        labels=["low", "medium", "high"],
    ).astype(str)
    queue["predicted_abuse"] = queue["risk_tier"].eq("high")
    queue["case_id"] = [f"CASE-{idx:04d}" for idx in range(1, len(queue) + 1)]
    queue["case_status"] = "needs_review"

    return queue.sort_values(["risk_score", "review_count"], ascending=[False, False]).reset_index(drop=True)
