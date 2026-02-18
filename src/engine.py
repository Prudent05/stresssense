from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import math

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = joblib.load(ROOT / "models" / "ui_model.joblib")

MODEL = BUNDLE["model"]          # sklearn Pipeline
FEATURES = BUNDLE["features"]    # list of feature names used in training


def tier_from_score(score: int) -> str:
    if score <= 33:
        return "Stable"
    if score <= 66:
        return "Watchlist"
    return "High Stress"


def _top_reasons(inputs: dict) -> list[str]:
    util = inputs.get("utilization_pct")
    missed = inputs.get("missed_payments")
    vol = inputs.get("spending_volatility")
    inc = inputs.get("income_stability")

    reasons = []
    if util is not None and util >= 60:
        reasons.append("High credit utilization is pushing risk up.")
    if missed is not None and missed >= 1:
        reasons.append("Missed/late payments strongly increase risk.")
    if vol is not None and vol >= 60:
        reasons.append("Unpredictable spending can increase risk.")
    if inc is not None and inc <= 40:
        reasons.append("Less stable income can increase risk.")

    return reasons[:3] if reasons else ["Your inputs look generally stable."]


def _actions(tier: str) -> list[str]:
    if tier == "Stable":
        return [
            "Keep utilization under ~30–40% when possible.",
            "Use reminders/autopay to avoid late payments.",
            "Keep spending more predictable month-to-month.",
        ]
    if tier == "Watchlist":
        return [
            "Prioritize on-time minimum payments for the next 60 days.",
            "Reduce utilization (even small paydowns help).",
            "Reduce high-volatility spending for 2–3 weeks.",
        ]
    return [
        "Make minimum payments your top priority right now.",
        "Reduce revolving balance/utilization as soon as possible.",
        "Pause new debt/inquiries until stable again.",
    ]


def _calibrate_prob(prob_raw: float) -> float:
    """
    Consumer-friendly calibration:
    - compress extremes so outputs don't feel unrealistically absolute
    - keep monotonic order (higher raw prob -> higher displayed prob)

    This uses a logit-squash:
      p' = sigmoid( a * logit(p) )
    where a < 1 compresses extremes.
    """
    p = float(prob_raw)
    # avoid inf logit
    eps = 1e-6
    p = max(eps, min(1 - eps, p))

    a = 0.65  # < 1 compresses extremes; tweak 0.55–0.8 to taste
    logit = math.log(p / (1 - p))
    p_cal = 1 / (1 + math.exp(-a * logit))
    return float(p_cal)


def predict_stress(user_inputs: dict) -> dict:
    """
    Consumer inputs expected:
      - utilization_pct (0–100)
      - missed_payments (0–10)
      - spending_volatility (0–100)
      - income_stability (0–100)
    """

    util_pct = float(user_inputs.get("utilization_pct", 35))
    missed = float(user_inputs.get("missed_payments", 0))
    vol = float(user_inputs.get("spending_volatility", 20))
    inc = float(user_inputs.get("income_stability", 80))

    row = {
        "util_proxy": util_pct / 100.0,
        "delinq_proxy": missed,
        "spend_vol_proxy": vol / 100.0,
        "income_stab_proxy": inc / 100.0,
        "util_trend": 0.0,
        "spend_vol_avg": vol / 100.0,
        "missed_rate": 1.0 if missed > 0 else 0.0,
        "history_len": 6.0,
    }

    X = pd.DataFrame([{f: row.get(f, np.nan) for f in FEATURES}])
    X = X.replace([np.inf, -np.inf], np.nan)

    # Raw model probability (keep for debugging/metrics if you want)
    prob_raw = float(MODEL.predict_proba(X)[0, 1])

    # Calibrated probability for consumer display
    prob = _calibrate_prob(prob_raw)

    stress_score = int(round(prob * 100))
    tier = tier_from_score(stress_score)

    top_reasons = _top_reasons(
        {
            "utilization_pct": util_pct,
            "missed_payments": missed,
            "spending_volatility": vol,
            "income_stability": inc,
        }
    )
    actions = _actions(tier)

    return {
        # show calibrated by default
        "prob_default": round(prob, 4),
        "stress_score": stress_score,
        "stress_tier": tier,
        "top_reasons": top_reasons,
        "actions": actions,

        # include raw for debugging (optional)
        "prob_default_raw": round(prob_raw, 4),
    }
