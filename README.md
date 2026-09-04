# ReviewSentinel — Product Abuse & Trust Investigation System

An end-to-end Trust & Safety analytics project that turns synthetic user-review activity into an explainable **detection → triage → investigation** workflow.

The project is intentionally built around investigation quality rather than automatic enforcement. It generates synthetic account behaviour with known ground truth, detects account- and cross-account abuse signals, prioritises cases with transparent reason codes, and evaluates the resulting queue with precision/recall metrics.

## What it detects

The synthetic dataset includes ordinary users plus three controlled abuse scenarios:

- **Spam bursts** — high-velocity repeated promotional content and link-heavy activity.
- **Coordinated campaigns** — multiple accounts reusing the same content and shared synthetic device/network infrastructure.
- **Policy evasion** — obfuscated policy-triggering terms designed to bypass naive keyword rules.

Signals include review velocity, duplicate-content ratio, cross-account text reuse, shared-device/network counts, policy-signal rate, URL rate, account age, and product diversity.

## Investigation workflow

Each account becomes a case with:

- a 0–100 risk score;
- low / medium / high risk tier;
- human-readable reason codes;
- review history and supporting evidence;
- an investigation queue ranked by risk;
- session-level analyst actions: **Dismiss / Escalate / Confirm abuse**.

The score is deterministic and explainable. It is used to prioritise human review, not to automatically penalise users.

## Evaluation

Because the data is synthetic, every account has a known ground-truth label. The pipeline reports:

- Precision
- Recall
- F1
- False-positive rate
- Precision@K for the highest-priority investigation queue
- Confusion-matrix counts

This makes the trade-off between catching abuse and incorrectly flagging legitimate users explicit and testable.

## Architecture

```text
Synthetic review activity
        |
        v
Text cleaning + policy-signal detection
        |
        v
Behavioural feature engineering
  - velocity / bursts
  - repeated content
  - cross-account reuse
  - shared device/network
  - URL + policy rates
        |
        v
Explainable account risk scoring
        |
        +----> Evaluation against synthetic ground truth
        |
        v
Prioritised investigation queue
        |
        v
Streamlit case drill-down + analyst triage
```

## Repository structure

```text
pipeline/
  data_ingestion.py      # deterministic synthetic users + abuse scenarios
  text_cleaning.py       # text normalisation
  violation_detector.py  # policy/evasion signal detection
  sentiment_analyzer.py  # legacy sentiment enrichment
  abuse_features.py      # account and cross-account behavioural features
  investigation.py       # risk scoring + case queue
  evaluation.py          # precision/recall/F1/Precision@K
  dashboard.py           # investigation UI
app.py                   # end-to-end pipeline runner
tests/                   # deterministic abuse-detection tests
.github/workflows/ci.yml # automated test workflow
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
streamlit run pipeline/dashboard.py
```

The first run upgrades the older demo dataset automatically if the required investigation fields are missing.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests verify that all synthetic abuse scenarios are present, the queue contains one case per account, risk scores remain bounded, and the detector clears minimum precision/recall thresholds on deterministic ground truth.

## Important limitations

- All user/account/device/network data is synthetic; this is a portfolio system, not a production abuse classifier.
- Shared device or network infrastructure is a risk signal, not proof of abuse.
- Static thresholds are intentionally transparent for demonstration; a production system would calibrate them by segment, investigation capacity, and false-positive cost.
- The synthetic ground truth is useful for regression testing but is easier than real adversarial platform abuse.
- Analyst dispositions in the public Streamlit app are session-only and are not written to a case-management database.

## Tech stack

Python, Pandas, Streamlit, TextBlob, Faker, pytest, GitHub Actions.
