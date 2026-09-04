from __future__ import annotations

import re

import pandas as pd

POLICY_PATTERNS = {
    "spam_or_promotion": re.compile(r"https?://|\bpromo\b|\blimited offer\b|\buse code\b", re.I),
    "policy_evasion": re.compile(r"sc[@4]m|fr[@4]ud|h\s+a\s+t\s+e", re.I),
    "sensitive_data": re.compile(r"\b(?:ssn|credit card number|private key)\b", re.I),
    "threat_or_abuse": re.compile(r"\b(?:threaten|abuse everyone|attack them)\b", re.I),
}


def _classify_violation(text: str) -> str:
    matches = [name for name, pattern in POLICY_PATTERNS.items() if pattern.search(str(text))]
    return ",".join(matches)


def detect_violations(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    source = result["review"].fillna("")
    result["violation_type"] = source.apply(_classify_violation)
    result["violation_flag"] = result["violation_type"].ne("")
    return result
