from __future__ import annotations

import pandas as pd


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _recompute(row: pd.Series) -> pd.Series:
    row["monthly_income"] = float(row["AMT_INCOME_TOTAL"]) / 12.0
    row["annuity_to_income_ratio"] = _clip(float(row["AMT_ANNUITY"]) / max(float(row["AMT_INCOME_TOTAL"]), 1.0), 0.0, 1.0)
    row["credit_to_income_ratio"] = _clip(float(row["AMT_CREDIT"]) / max(float(row["AMT_INCOME_TOTAL"]), 1.0), 0.0, 20.0)
    return row


def apply_home_credit_scenario(snapshot: pd.Series, scenario_name: str) -> pd.Series:
    updated = snapshot.copy()

    if scenario_name == "income_drop":
        updated["AMT_INCOME_TOTAL"] *= 0.85
        updated["total_debt_to_income"] = _clip(float(updated["total_debt_to_income"]) / 0.85, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) - 0.08, 0.0, 1.0)
        updated["cc_util_recent"] = _clip(float(updated["cc_util_recent"]) + 0.05, 0.0, 1.0)
        updated["cc_drawings_atm_ratio"] = _clip(float(updated["cc_drawings_atm_ratio"]) + 0.04, 0.0, 1.0)
        updated["financial_engagement_score"] = _clip(float(updated["financial_engagement_score"]) - 0.05, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) + 1.0, 0.0, 10.0)
    elif scenario_name == "missed_payment":
        updated["inst_late_recent"] = _clip(float(updated["inst_late_recent"]) + 0.18, 0.0, 1.0)
        updated["inst_late_deterioration"] = _clip(float(updated["inst_late_deterioration"]) + 0.12, 0.0, 1.0)
        updated["bureau_recent_dpd"] = _clip(float(updated["bureau_recent_dpd"]) + 1.0, 0.0, 10.0)
        updated["cc_min_payment_ratio"] = _clip(float(updated["cc_min_payment_ratio"]) - 0.05, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) - 0.04, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) + 1.0, 0.0, 10.0)
    elif scenario_name == "cash_advance_spike":
        updated["cc_drawings_atm_ratio"] = _clip(float(updated["cc_drawings_atm_ratio"]) + 0.12, 0.0, 1.0)
        updated["cc_util_recent"] = _clip(float(updated["cc_util_recent"]) + 0.08, 0.0, 1.0)
        updated["cc_util_deterioration"] = _clip(float(updated["cc_util_deterioration"]) + 0.07, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) - 0.05, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) + 1.0, 0.0, 10.0)
    elif scenario_name == "paydown_support":
        updated["cc_util_recent"] = _clip(float(updated["cc_util_recent"]) - 0.15, 0.0, 1.0)
        updated["cc_util_deterioration"] = _clip(float(updated["cc_util_deterioration"]) - 0.10, 0.0, 1.0)
        updated["cc_min_payment_ratio"] = _clip(float(updated["cc_min_payment_ratio"]) + 0.08, 0.0, 1.0)
        updated["total_debt_to_income"] = _clip(float(updated["total_debt_to_income"]) - 0.08, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) + 0.10, 0.0, 1.0)
        updated["financial_engagement_score"] = _clip(float(updated["financial_engagement_score"]) + 0.05, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) - 1.0, 0.0, 10.0)
    elif scenario_name == "restructure_offer":
        updated["inst_late_recent"] = _clip(float(updated["inst_late_recent"]) - 0.12, 0.0, 1.0)
        updated["inst_late_deterioration"] = _clip(float(updated["inst_late_deterioration"]) - 0.10, 0.0, 1.0)
        updated["bureau_recent_dpd"] = _clip(float(updated["bureau_recent_dpd"]) - 1.0, 0.0, 10.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) + 0.07, 0.0, 1.0)
        updated["financial_engagement_score"] = _clip(float(updated["financial_engagement_score"]) + 0.07, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    updated = _recompute(updated)
    updated["scenario_name"] = scenario_name
    return updated
