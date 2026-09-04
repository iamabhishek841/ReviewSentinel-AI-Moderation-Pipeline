from __future__ import annotations

import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explainable account- and cross-account signals to each review."""
    required = {
        "user_id",
        "review",
        "cleaned_review",
        "timestamp",
        "device_id",
        "ip_cluster",
        "product_id",
        "account_age_days",
        "violation_flag",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["review_hour"] = result["timestamp"].dt.floor("h")
    result["has_url"] = result["review"].str.contains(r"https?://", case=False, regex=True, na=False)

    hourly = (
        result.groupby(["user_id", "review_hour"])
        .size()
        .rename("reviews_in_hour")
        .reset_index()
    )
    result = result.merge(hourly, on=["user_id", "review_hour"], how="left")

    user_stats = result.groupby("user_id").agg(
        account_review_count=("review_id", "count"),
        max_reviews_1h=("reviews_in_hour", "max"),
        unique_review_texts=("cleaned_review", "nunique"),
        unique_products=("product_id", "nunique"),
        violation_rate=("violation_flag", "mean"),
        url_rate=("has_url", "mean"),
    )
    user_stats["duplicate_ratio"] = (
        1 - user_stats["unique_review_texts"] / user_stats["account_review_count"]
    ).clip(lower=0.0, upper=1.0)

    text_accounts = result.groupby("cleaned_review")["user_id"].nunique().rename("same_text_accounts")
    device_accounts = result.groupby("device_id")["user_id"].nunique().rename("shared_device_accounts")
    ip_accounts = result.groupby("ip_cluster")["user_id"].nunique().rename("shared_ip_accounts")

    result = result.join(text_accounts, on="cleaned_review")
    result = result.join(device_accounts, on="device_id")
    result = result.join(ip_accounts, on="ip_cluster")
    result = result.join(user_stats, on="user_id")

    return result
