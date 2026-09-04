from pipeline.abuse_features import add_behavioral_features
from pipeline.data_ingestion import generate_fake_reviews
from pipeline.evaluation import evaluate_detection
from pipeline.investigation import build_investigation_queue
from pipeline.text_cleaning import preprocess_reviews
from pipeline.violation_detector import detect_violations


def build_core(seed=42):
    df = generate_fake_reviews(num_reviews=2200, num_users=200, seed=seed)
    df = preprocess_reviews(df)
    df = detect_violations(df)
    df = add_behavioral_features(df)
    queue = build_investigation_queue(df)
    return df, queue


def test_generator_contains_known_abuse_scenarios():
    df, _ = build_core()
    scenarios = set(df["ground_truth_abuse_type"])
    assert {"none", "spam_burst", "coordinated_campaign", "policy_evasion"}.issubset(scenarios)


def test_investigation_queue_is_one_case_per_account():
    _, queue = build_core()
    assert queue["user_id"].is_unique
    assert queue["case_id"].is_unique
    assert set(queue["risk_tier"]).issubset({"low", "medium", "high"})
    assert queue["risk_score"].between(0, 100).all()


def test_detection_has_useful_precision_and_recall_on_synthetic_ground_truth():
    _, queue = build_core()
    metrics = evaluate_detection(queue, top_k=20)
    assert metrics["precision"] >= 0.75
    assert metrics["recall"] >= 0.75
    assert metrics["precision_at_k"] >= 0.80
