from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STAGE_ORDER = ["stable", "vulnerable", "slipping", "critical"]
ACTION_PLAYBOOK = {
    "stable": ["Keep current limit policy", "Send financial wellness tips", "Monitor for trend breaks"],
    "vulnerable": ["Send soft reminder", "Offer budgeting nudge", "Increase monitoring frequency"],
    "slipping": ["Offer restructure option", "Trigger proactive human review", "Pause marketing credit offers"],
    "critical": ["Launch urgent outreach", "Evaluate hardship program", "Reduce exposure and review line strategy"],
}


@dataclass(frozen=True)
class CustomerAssessment:
    customer_id: str
    month: int
    distress_score: float
    distress_stage: str
    early_warning_probability: float
    drivers: list[str]
    recommended_actions: list[str]


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def compute_distress_score(row: pd.Series) -> float:
    utilization_component = row["utilization"]
    payment_component = _clip(1.0 - (row["payment_to_balance_ratio"] / 0.30), 0.0, 1.0)
    lateness_component = _clip(row["days_late"] / 14.0, 0.0, 1.0)
    cash_advance_component = _clip(row["cash_advance_ratio"] / 0.20, 0.0, 1.0)
    activity_component = _clip(1.0 - row["account_activity_index"], 0.0, 1.0)
    expense_component = _clip((row["expense_to_income_ratio"] - 0.55) / 0.40, 0.0, 1.0)

    return _clip(
        0.27 * utilization_component
        + 0.19 * payment_component
        + 0.17 * lateness_component
        + 0.14 * cash_advance_component
        + 0.11 * activity_component
        + 0.12 * expense_component,
        0.0,
        1.0,
    )


def score_to_stage(score: float) -> str:
    if score < 0.30:
        return "stable"
    if score < 0.48:
        return "vulnerable"
    if score < 0.68:
        return "slipping"
    return "critical"


def build_driver_list(row: pd.Series) -> list[str]:
    drivers: list[str] = []

    if row["utilization"] >= 0.75:
        drivers.append("credit utilization is critically high")
    elif row["utilization"] >= 0.55:
        drivers.append("credit utilization is rising into a risky zone")

    if row["payment_to_balance_ratio"] <= 0.06:
        drivers.append("payments are too small relative to outstanding balance")
    elif row["payment_to_balance_ratio"] <= 0.12:
        drivers.append("payment-to-balance ratio is weakening")

    if row["days_late"] >= 8:
        drivers.append("repayment timing is irregular and frequently late")
    elif row["days_late"] >= 3:
        drivers.append("repayment timing is starting to slip")

    if row["cash_advance_ratio"] >= 0.14:
        drivers.append("cash-advance dependence is elevated")

    if row["account_activity_index"] <= 0.65:
        drivers.append("account activity is dropping sharply")

    if row["expense_to_income_ratio"] >= 0.85:
        drivers.append("income-to-expense cushion is nearly exhausted")
    elif row["expense_to_income_ratio"] >= 0.72:
        drivers.append("income-to-expense pressure is increasing")

    if not drivers:
        drivers.append("behavior remains broadly stable")
    return drivers


def build_personalized_recommendations(row: pd.Series, stage: str) -> list[str]:
    recommendations: list[str] = []

    current_balance = float(row["revolving_balance"])
    credit_limit = max(float(row["credit_limit"]), 1.0)
    utilization = float(row["utilization"])
    payment_ratio = float(row["payment_to_balance_ratio"])
    cash_advance_ratio = float(row["cash_advance_ratio"])
    expense_ratio = float(row["expense_to_income_ratio"])
    activity_index = float(row["account_activity_index"])
    days_late = int(row["days_late"])

    target_utilization = 0.70 if stage == "critical" else 0.55 if stage == "slipping" else 0.40
    if utilization > target_utilization:
        target_balance = credit_limit * target_utilization
        paydown_needed = max(current_balance - target_balance, 0.0)
        recommendations.append(
            f"Reduce revolving balance by about ${paydown_needed:,.0f} to bring utilization from {utilization:.0%} closer to {target_utilization:.0%}."
        )

    target_payment_ratio = 0.12 if stage in {"critical", "slipping"} else 0.18
    if payment_ratio < target_payment_ratio:
        recommendations.append(
            f"Increase payment-to-balance ratio from {payment_ratio:.0%} to at least {target_payment_ratio:.0%} to slow balance stress."
        )

    if days_late >= 8:
        recommendations.append("Prioritize on-time repayment for the next 2 cycles and bring late days below 3 immediately.")
    elif days_late >= 3:
        recommendations.append("Set up autopay or reminders to stop repayment timing from slipping further.")

    if cash_advance_ratio >= 0.14:
        recommendations.append(
            f"Cut cash-advance usage from {cash_advance_ratio:.0%} to below 10%; this is a strong distress signal."
        )

    if expense_ratio >= 0.85:
        recommendations.append(
            f"Lower monthly expense pressure from {expense_ratio:.0%} of income to under 75% through budget cuts or income support."
        )
    elif expense_ratio >= 0.72:
        recommendations.append(
            f"Reduce fixed outflows so expense-to-income pressure falls below 70%; current level is {expense_ratio:.0%}."
        )

    if activity_index <= 0.65:
        recommendations.append(
            "Review engagement drop and trigger a human outreach or digital check-in before the account disengages further."
        )

    if len(recommendations) < 3:
        for action in ACTION_PLAYBOOK[stage]:
            if action not in recommendations:
                recommendations.append(action)
            if len(recommendations) >= 3:
                break

    return recommendations[:5]


def assess_customer(row: pd.Series) -> CustomerAssessment:
    score = compute_distress_score(row)
    stage = score_to_stage(score)
    probability = _clip((score * 0.75) + (0.10 if stage in {"slipping", "critical"} else 0.02), 0.01, 0.98)
    drivers = build_driver_list(row)
    actions = build_personalized_recommendations(row, stage)
    return CustomerAssessment(
        customer_id=str(row["customer_id"]),
        month=int(row["month"]),
        distress_score=round(score, 4),
        distress_stage=stage,
        early_warning_probability=round(probability, 4),
        drivers=drivers,
        recommended_actions=actions,
    )


def score_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["distress_score"] = scored.apply(compute_distress_score, axis=1)
    scored["distress_stage"] = scored["distress_score"].apply(score_to_stage)
    scored["early_warning_probability"] = scored["distress_score"].apply(
        lambda score: _clip((score * 0.75) + (0.10 if score >= 0.48 else 0.02), 0.01, 0.98)
    )
    return scored


def latest_customer_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    scored = score_dataset(frame)
    latest = scored.sort_values(["customer_id", "month"]).groupby("customer_id", as_index=False).tail(1)
    return latest.reset_index(drop=True)
