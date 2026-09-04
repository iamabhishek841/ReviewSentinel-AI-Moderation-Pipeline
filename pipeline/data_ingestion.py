from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

BENIGN_REVIEWS = [
    "This app is amazing and easy to use.",
    "The latest update fixed my issue.",
    "Customer support was helpful today.",
    "The app works fine but can be slow sometimes.",
    "I like the design and navigation.",
    "The checkout flow was straightforward.",
    "I had a payment issue but support resolved it.",
    "Good experience overall.",
    "The feature is useful but needs more filters.",
    "The app crashed once but has been stable since.",
]

SPAM_TEXT = [
    "Limited offer - click https://promo.example/deal for a bonus now",
    "Get a bonus today at https://promo.example/deal",
    "Special promo: visit https://promo.example/deal before it expires",
]

COORDINATED_TEXT = [
    "Five stars - use code BOOST50 at https://promo.example/boost",
    "Five stars! Use code BOOST50 at https://promo.example/boost",
]

EVASION_TEXT = [
    "This service is a sc@m - message me at promo dot example for the real one",
    "Totally fr@ud - use my private link instead",
    "Ignore the rules and h a t e everyone who disagrees",
]


def generate_fake_reviews(
    num_reviews: int = 3000,
    num_users: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic review activity with account-level abuse patterns.

    The data is intentionally synthetic. A minority of accounts are assigned to one of
    three abuse scenarios so the project can evaluate detection quality against known
    ground truth without using real user data.
    """
    if num_users < 30:
        raise ValueError("num_users must be at least 30")
    if num_reviews < num_users:
        raise ValueError("num_reviews must be at least num_users")

    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    users = [f"user_{idx:04d}" for idx in range(1, num_users + 1)]
    abusive_count = max(18, int(num_users * 0.12))
    abusive_users = rng.sample(users, abusive_count)

    split = abusive_count // 3
    spam_users = set(abusive_users[:split])
    coordinated_users = set(abusive_users[split : split * 2])
    evasion_users = set(abusive_users[split * 2 :])

    coordinated_list = sorted(coordinated_users)
    campaign_by_user: dict[str, str] = {}
    for idx, user_id in enumerate(coordinated_list):
        campaign_by_user[user_id] = f"campaign_{idx // 4 + 1:02d}"

    now = datetime.now().replace(second=0, microsecond=0)
    start = now - timedelta(days=30)

    account_age = {u: rng.randint(2, 1200) for u in users}
    device_by_user = {u: f"device_{rng.randint(1, num_users * 2):04d}" for u in users}
    ip_by_user = {u: f"ip_cluster_{rng.randint(1, max(20, num_users // 3)):03d}" for u in users}

    # Coordinated accounts deliberately share infrastructure to create an interpretable
    # cross-account signal. Benign accounts may still share devices/IP clusters by chance.
    for user_id in coordinated_users:
        campaign_num = campaign_by_user[user_id].split("_")[-1]
        device_by_user[user_id] = f"shared_device_{campaign_num}"
        ip_by_user[user_id] = f"shared_ip_{campaign_num}"
        account_age[user_id] = rng.randint(1, 21)

    burst_anchor = {
        u: start + timedelta(days=rng.randint(5, 27), hours=rng.randint(0, 22))
        for u in spam_users | coordinated_users
    }

    user_weights = [4 if u in abusive_users else 1 for u in users]
    rows: list[dict] = []

    for review_id in range(1, num_reviews + 1):
        user_id = rng.choices(users, weights=user_weights, k=1)[0]
        is_abusive = user_id in abusive_users

        if user_id in spam_users:
            abuse_type = "spam_burst"
        elif user_id in coordinated_users:
            abuse_type = "coordinated_campaign"
        elif user_id in evasion_users:
            abuse_type = "policy_evasion"
        else:
            abuse_type = "none"

        if abuse_type in {"spam_burst", "coordinated_campaign"} and rng.random() < 0.70:
            timestamp = burst_anchor[user_id] + timedelta(minutes=rng.randint(0, 45))
        else:
            timestamp = start + timedelta(
                days=rng.randint(0, 29),
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )

        if abuse_type == "spam_burst":
            text = rng.choice(SPAM_TEXT)
            product_id = f"product_{rng.randint(1, 3):03d}"
            rating = 5
            campaign_id = ""
        elif abuse_type == "coordinated_campaign":
            text = rng.choice(COORDINATED_TEXT)
            product_id = "product_001"
            rating = 5
            campaign_id = campaign_by_user[user_id]
        elif abuse_type == "policy_evasion":
            text = rng.choice(EVASION_TEXT) if rng.random() < 0.75 else rng.choice(BENIGN_REVIEWS)
            product_id = f"product_{rng.randint(1, 20):03d}"
            rating = rng.choice([1, 5])
            campaign_id = ""
        else:
            text = rng.choice(BENIGN_REVIEWS)
            product_id = f"product_{rng.randint(1, 40):03d}"
            rating = rng.randint(1, 5)
            campaign_id = ""

        rows.append(
            {
                "review_id": f"rev_{review_id:06d}",
                "user_id": user_id,
                "review": text,
                "rating": rating,
                "timestamp": timestamp.isoformat(timespec="minutes"),
                "product_id": product_id,
                "device_id": device_by_user[user_id],
                "ip_cluster": ip_by_user[user_id],
                "account_age_days": account_age[user_id],
                "campaign_id": campaign_id,
                "ground_truth_abuse": is_abusive,
                "ground_truth_abuse_type": abuse_type,
            }
        )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def save_reviews_csv(path: str = "data/user_reviews.csv", **kwargs) -> None:
    df = generate_fake_reviews(**kwargs)
    df.to_csv(path, index=False)
    print(f"[ok] Synthetic reviews saved to {path}")


if __name__ == "__main__":
    save_reviews_csv()
