from __future__ import annotations

import pandas as pd


def _recompute(snapshot: pd.Series) -> pd.Series:
    snapshot["revolving_balance"] = float(snapshot["credit_limit"]) * float(snapshot["utilization"])
    snapshot["payment_amount"] = float(snapshot["revolving_balance"]) * float(snapshot["payment_to_balance_ratio"])
    snapshot["expense_to_income_ratio"] = float(snapshot["monthly_expenses"]) / max(float(snapshot["monthly_income"]), 1.0)
    return snapshot


def apply_scenario(snapshot: pd.Series, scenario_name: str) -> pd.Series:
    updated = snapshot.copy()

    if scenario_name == "income_drop":
        updated["monthly_income"] *= 0.85
        updated["utilization"] = min(float(updated["utilization"]) + 0.05, 0.99)
        updated["payment_to_balance_ratio"] = max(float(updated["payment_to_balance_ratio"]) - 0.02, 0.01)
        updated["cash_advance_ratio"] = min(float(updated["cash_advance_ratio"]) + 0.04, 0.50)
        updated["account_activity_index"] = max(float(updated["account_activity_index"]) - 0.05, 0.25)
        updated["days_late"] = min(int(updated["days_late"]) + 3, 30)
    elif scenario_name == "missed_payment":
        updated["payment_to_balance_ratio"] *= 0.55
        updated["utilization"] = min(float(updated["utilization"]) + 0.03, 0.99)
        updated["account_activity_index"] = max(float(updated["account_activity_index"]) - 0.03, 0.25)
        updated["days_late"] = min(int(updated["days_late"]) + 6, 30)
    elif scenario_name == "cash_advance_spike":
        updated["cash_advance_ratio"] = min(float(updated["cash_advance_ratio"]) + 0.10, 0.50)
        updated["utilization"] = min(float(updated["utilization"]) + 0.07, 0.99)
        updated["payment_to_balance_ratio"] = max(float(updated["payment_to_balance_ratio"]) - 0.02, 0.01)
        updated["account_activity_index"] = max(float(updated["account_activity_index"]) - 0.04, 0.25)
    elif scenario_name == "paydown_support":
        updated["utilization"] = max(float(updated["utilization"]) - 0.15, 0.02)
        updated["payment_to_balance_ratio"] = min(float(updated["payment_to_balance_ratio"]) + 0.08, 0.65)
        updated["cash_advance_ratio"] = max(float(updated["cash_advance_ratio"]) - 0.04, 0.0)
        updated["days_late"] = max(int(updated["days_late"]) - 3, 0)
        updated["account_activity_index"] = min(float(updated["account_activity_index"]) + 0.04, 1.20)
    elif scenario_name == "restructure_offer":
        updated["payment_to_balance_ratio"] = min(float(updated["payment_to_balance_ratio"]) + 0.05, 0.65)
        updated["days_late"] = max(int(updated["days_late"]) - 4, 0)
        updated["account_activity_index"] = min(float(updated["account_activity_index"]) + 0.08, 1.20)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    updated = _recompute(updated)
    updated["scenario_name"] = scenario_name
    return updated
