from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.home_credit_engine import assess_home_credit_customer, score_home_credit_dataset
from src.home_credit_model import HomeCreditModelBundle, add_model_scores
from src.home_credit_scenarios import _clip


ACTION_LIBRARY = {
    "paydown_support": {
        "label": "Balance paydown support",
        "description": "Directly reduce revolving utilization and improve payment headroom.",
    },
    "restructure_offer": {
        "label": "Restructure offer",
        "description": "Reshape obligations so recent lateness and payment pressure ease.",
    },
    "autopay_stabilization": {
        "label": "Autopay stabilization",
        "description": "Reduce near-term lateness through reminders, autopay, and payment discipline.",
    },
    "cash_advance_replacement": {
        "label": "Cash advance replacement",
        "description": "Replace ATM-heavy liquidity usage with a safer hardship or installment option.",
    },
    "hardship_plan": {
        "label": "Temporary hardship plan",
        "description": "Improve payment capacity for customers under immediate stress.",
    },
}
ASSUMED_LOSS_GIVEN_DEFAULT = 0.45
DEFAULT_STAGE_TARGETS = {
    "critical": 4,
    "slipping": 4,
    "vulnerable": 3,
    "stable": 1,
}


@dataclass(frozen=True)
class RankedAction:
    action_id: str
    label: str
    description: str
    expected_probability_drop: float
    expected_stage: str
    ml_probability_after: float
    blended_probability_after: float
    expected_loss_before: float
    expected_loss_after: float
    estimated_saved_loss: float
    rationale: list[str]


def apply_intervention(snapshot: pd.Series, action_id: str) -> pd.Series:
    updated = snapshot.copy()

    if action_id == "paydown_support":
        updated["cc_util_recent"] = _clip(float(updated["cc_util_recent"]) - 0.18, 0.0, 1.0)
        updated["cc_util_deterioration"] = _clip(float(updated["cc_util_deterioration"]) - 0.12, 0.0, 1.0)
        updated["cc_min_payment_ratio"] = _clip(float(updated["cc_min_payment_ratio"]) + 0.08, 0.0, 1.0)
        updated["total_debt_to_income"] = _clip(float(updated["total_debt_to_income"]) - 0.08, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) + 0.10, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) - 1.0, 0.0, 10.0)
    elif action_id == "restructure_offer":
        updated["inst_late_recent"] = _clip(float(updated["inst_late_recent"]) - 0.14, 0.0, 1.0)
        updated["inst_late_deterioration"] = _clip(float(updated["inst_late_deterioration"]) - 0.10, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) + 0.07, 0.0, 1.0)
        updated["bureau_recent_dpd"] = _clip(float(updated["bureau_recent_dpd"]) - 1.0, 0.0, 10.0)
        updated["payment_behavior_score"] = _clip(float(updated["payment_behavior_score"]) + 0.06, 0.0, 1.0)
    elif action_id == "autopay_stabilization":
        updated["inst_late_recent"] = _clip(float(updated["inst_late_recent"]) - 0.10, 0.0, 1.0)
        updated["inst_late_deterioration"] = _clip(float(updated["inst_late_deterioration"]) - 0.07, 0.0, 1.0)
        updated["cc_min_payment_ratio"] = _clip(float(updated["cc_min_payment_ratio"]) + 0.05, 0.0, 1.0)
        updated["payment_behavior_score"] = _clip(float(updated["payment_behavior_score"]) + 0.10, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) - 1.0, 0.0, 10.0)
    elif action_id == "cash_advance_replacement":
        updated["cc_drawings_atm_ratio"] = _clip(float(updated["cc_drawings_atm_ratio"]) - 0.14, 0.0, 1.0)
        updated["cc_util_recent"] = _clip(float(updated["cc_util_recent"]) - 0.06, 0.0, 1.0)
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) + 0.04, 0.0, 1.0)
        updated["financial_engagement_score"] = _clip(float(updated["financial_engagement_score"]) + 0.05, 0.0, 1.0)
    elif action_id == "hardship_plan":
        updated["monthly_payment_capacity"] = _clip(float(updated["monthly_payment_capacity"]) + 0.12, 0.0, 1.0)
        updated["total_debt_to_income"] = _clip(float(updated["total_debt_to_income"]) - 0.06, 0.0, 1.0)
        updated["inst_late_recent"] = _clip(float(updated["inst_late_recent"]) - 0.08, 0.0, 1.0)
        updated["bureau_overdue_to_debt_ratio"] = _clip(float(updated["bureau_overdue_to_debt_ratio"]) - 0.04, 0.0, 1.0)
        updated["red_flags_count"] = _clip(float(updated["red_flags_count"]) - 1.0, 0.0, 10.0)
    else:
        raise ValueError(f"Unknown action: {action_id}")

    updated["scenario_name"] = action_id
    return updated


def _build_rationale(before_row: pd.Series, after_row: pd.Series) -> list[str]:
    deltas = [
        ("recent card utilization", "cc_util_recent"),
        ("cash-advance dependence", "cc_drawings_atm_ratio"),
        ("recent installment lateness", "inst_late_recent"),
        ("payment capacity", "monthly_payment_capacity"),
        ("debt-to-income pressure", "total_debt_to_income"),
        ("payment behavior quality", "payment_behavior_score"),
    ]
    notes: list[str] = []
    for label, column in deltas:
        delta = float(after_row[column] - before_row[column])
        if abs(delta) < 0.015:
            continue
        direction = "improves" if delta < 0 and column not in {"monthly_payment_capacity", "payment_behavior_score"} else "improves"
        if column in {"monthly_payment_capacity", "payment_behavior_score"}:
            notes.append(f"{label.title()} improves by {abs(delta):.0%}.")
        else:
            notes.append(f"{label.title()} improves by {abs(delta):.0%}.")
        if len(notes) >= 2:
            break
    return notes or ["Improves a mix of repayment stability and liquidity signals."]


def rank_actions(
    scored_data: pd.DataFrame,
    model_bundle: HomeCreditModelBundle,
    baseline_row: pd.Series,
    top_n: int = 3,
) -> list[RankedAction]:
    baseline_customer_id = str(baseline_row["customer_id"])
    baseline_blended = float(baseline_row["blended_probability"])
    baseline_expected_loss = float(baseline_row["AMT_CREDIT"]) * baseline_blended * ASSUMED_LOSS_GIVEN_DEFAULT

    ranked: list[RankedAction] = []
    for action_id, metadata in ACTION_LIBRARY.items():
        candidate_raw = apply_intervention(baseline_row, action_id)
        candidate_frame = pd.concat(
            [scored_data[scored_data["customer_id"] != baseline_customer_id], pd.DataFrame([candidate_raw])],
            ignore_index=True,
        )
        rescored = add_model_scores(score_home_credit_dataset(candidate_frame), model_bundle)
        candidate_row = rescored[rescored["customer_id"] == baseline_customer_id].iloc[0].copy()
        candidate_assessment = assess_home_credit_customer(rescored, candidate_row)
        probability_drop = baseline_blended - float(candidate_row["blended_probability"])
        expected_loss_after = float(candidate_row["AMT_CREDIT"]) * float(candidate_row["blended_probability"]) * ASSUMED_LOSS_GIVEN_DEFAULT
        estimated_saved_loss = baseline_expected_loss - expected_loss_after

        ranked.append(
            RankedAction(
                action_id=action_id,
                label=str(metadata["label"]),
                description=str(metadata["description"]),
                expected_probability_drop=probability_drop,
                expected_stage=candidate_assessment.distress_stage,
                ml_probability_after=float(candidate_row["ml_default_probability"]),
                blended_probability_after=float(candidate_row["blended_probability"]),
                expected_loss_before=baseline_expected_loss,
                expected_loss_after=expected_loss_after,
                estimated_saved_loss=estimated_saved_loss,
                rationale=_build_rationale(baseline_row, candidate_row),
            )
        )

    ranked.sort(key=lambda action: action.expected_probability_drop, reverse=True)
    return ranked[:top_n]


def build_portfolio_queue(
    scored_data: pd.DataFrame,
    model_bundle: HomeCreditModelBundle,
    queue_size: int = 12,
    candidate_pool_size: int = 40,
) -> pd.DataFrame:
    stage_candidate_frames: list[pd.DataFrame] = []
    for stage_name, target_count in DEFAULT_STAGE_TARGETS.items():
        stage_slice = (
            scored_data[scored_data["blended_stage"] == stage_name]
            .sort_values(["blended_probability", "ml_default_probability"], ascending=[False, False])
            .head(max(target_count * 3, target_count))
        )
        if not stage_slice.empty:
            stage_candidate_frames.append(stage_slice)

    if stage_candidate_frames:
        high_risk = (
            pd.concat(stage_candidate_frames, ignore_index=False)
            .drop_duplicates(subset=["customer_id"])
            .sort_values(["blended_probability", "ml_default_probability"], ascending=[False, False])
        )
    else:
        high_risk = scored_data.sort_values(
            ["blended_probability", "ml_default_probability"],
            ascending=[False, False],
        )

    if len(high_risk) < candidate_pool_size:
        additional_pool = (
            scored_data.sort_values(["blended_probability", "ml_default_probability"], ascending=[False, False])
            .head(candidate_pool_size)
        )
        high_risk = (
            pd.concat([high_risk, additional_pool], ignore_index=False)
            .drop_duplicates(subset=["customer_id"])
            .sort_values(["blended_probability", "ml_default_probability"], ascending=[False, False])
        )
    else:
        high_risk = high_risk.reset_index(drop=True)

    queue_rows: list[dict[str, object]] = []
    for _, row in high_risk.iterrows():
        actions = rank_actions(scored_data, model_bundle, row, top_n=1)
        if not actions:
            continue
        best_action = actions[0]
        queue_rows.append(
            {
                "customer_id": str(row["customer_id"]),
                "baseline_stage": str(row["blended_stage"]),
                "baseline_blended_probability": float(row["blended_probability"]),
                "baseline_ml_probability": float(row["ml_default_probability"]),
                "credit_amount": float(row["AMT_CREDIT"]),
                "best_action": best_action.label,
                "expected_stage_after_action": best_action.expected_stage,
                "risk_drop": best_action.expected_probability_drop,
                "estimated_saved_loss": best_action.estimated_saved_loss,
                "priority_score": (best_action.estimated_saved_loss * 0.7)
                + (best_action.expected_probability_drop * float(row["AMT_CREDIT"]) * 0.3),
            }
        )

    queue = pd.DataFrame(queue_rows)
    if queue.empty:
        return queue

    queue = queue.sort_values(
        ["priority_score", "estimated_saved_loss", "risk_drop"],
        ascending=[False, False, False],
    )

    selected_frames: list[pd.DataFrame] = []
    selected_customer_ids: set[str] = set()
    for stage_name, target_count in DEFAULT_STAGE_TARGETS.items():
        stage_rows = (
            queue[queue["baseline_stage"] == stage_name]
            .sort_values(["priority_score", "estimated_saved_loss", "risk_drop"], ascending=[False, False, False])
            .head(target_count)
        )
        if not stage_rows.empty:
            selected_frames.append(stage_rows)
            selected_customer_ids.update(stage_rows["customer_id"].astype(str).tolist())

    selected_queue = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(columns=queue.columns)

    if len(selected_queue) < queue_size:
        remaining = queue[~queue["customer_id"].astype(str).isin(selected_customer_ids)]
        filler = remaining.head(queue_size - len(selected_queue))
        selected_queue = pd.concat([selected_queue, filler], ignore_index=True)

    selected_queue = (
        selected_queue.drop_duplicates(subset=["customer_id"])
        .sort_values(["priority_score", "estimated_saved_loss", "risk_drop"], ascending=[False, False, False])
        .head(queue_size)
        .reset_index(drop=True)
    )
    return selected_queue
