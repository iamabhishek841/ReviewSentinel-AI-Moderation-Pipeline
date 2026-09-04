from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_detection(queue: pd.DataFrame, top_k: int = 20) -> dict[str, Any]:
    if "ground_truth_abuse" not in queue.columns:
        raise ValueError("ground_truth_abuse is required for synthetic evaluation")

    truth = queue["ground_truth_abuse"].astype(bool)
    pred = queue["predicted_abuse"].astype(bool)

    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fn = int((truth & ~pred).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    ranked = queue.sort_values("risk_score", ascending=False).head(top_k)
    precision_at_k = float(ranked["ground_truth_abuse"].astype(bool).mean()) if len(ranked) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "precision_at_k": precision_at_k,
        "top_k": min(top_k, len(queue)),
    }
