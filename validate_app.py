from __future__ import annotations

import pandas as pd

from src.home_credit_actions import build_portfolio_queue, rank_actions
from src.home_credit_engine import STAGE_ORDER, assess_home_credit_customer, score_home_credit_dataset
from src.home_credit_loader import load_home_credit_sample
from src.home_credit_model import add_model_scores, explain_model_prediction, train_home_credit_model
from src.home_credit_scenarios import apply_home_credit_scenario


SCENARIOS = [
    "none",
    "income_drop",
    "missed_payment",
    "cash_advance_spike",
    "paydown_support",
    "restructure_offer",
]


def main() -> None:
    df = load_home_credit_sample(sample_size=1200, seed=42)
    scored_base = score_home_credit_dataset(df)
    model_bundle = train_home_credit_model(scored_base, seed=42)
    scored = add_model_scores(scored_base, model_bundle)

    print("DATA", len(scored))
    print("BLENDED_STAGE_COUNTS", scored["blended_stage"].value_counts().sort_index().to_dict())

    stage_samples: list[tuple[str, pd.Series]] = []
    for stage in STAGE_ORDER:
        rows = scored[scored["blended_stage"] == stage].head(3)
        for _, row in rows.iterrows():
            stage_samples.append((stage, row.copy()))

    failures: list[str] = []
    action_issues: list[str] = []

    for _, row in stage_samples:
        customer_id = str(row["customer_id"])
        actions = rank_actions(scored, model_bundle, row, top_n=3)
        if len(actions) != 3:
            failures.append(f"action_count::{customer_id}::{len(actions)}")

        for action in actions:
            if action.estimated_saved_loss < -1e-6:
                action_issues.append(
                    f"negative_saved_loss::{customer_id}::{action.label}::{action.estimated_saved_loss:.2f}"
                )

        if len(explain_model_prediction(row, model_bundle)) == 0:
            failures.append(f"no_ml_expl::{customer_id}")

        for scenario in SCENARIOS:
            try:
                if scenario == "none":
                    scenario_row = row.copy()
                    scenario_scored = scored
                else:
                    scenario_raw = apply_home_credit_scenario(row, scenario)
                    augmented = pd.concat(
                        [
                            scored[scored["customer_id"] != customer_id],
                            pd.DataFrame([scenario_raw]),
                        ],
                        ignore_index=True,
                    )
                    scenario_scored = add_model_scores(score_home_credit_dataset(augmented), model_bundle)
                    scenario_row = scenario_scored[scenario_scored["customer_id"] == customer_id].iloc[0].copy()

                assess_home_credit_customer(scenario_scored, scenario_row)
                explain_model_prediction(scenario_row, model_bundle)
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append(f"scenario::{customer_id}::{scenario}::{type(exc).__name__}::{exc}")

    queue = build_portfolio_queue(scored, model_bundle, queue_size=12, candidate_pool_size=40)
    print("QUEUE_LEN", len(queue))
    print("QUEUE_STAGE_COUNTS", queue["baseline_stage"].value_counts().to_dict())
    print("ACTION_ISSUES", len(action_issues))
    print("FAILURES", len(failures))

    if action_issues:
        print("ACTION_ISSUE_EXAMPLES", action_issues[:5])
    if failures:
        print("FAILURE_EXAMPLES", failures[:10])
        raise SystemExit(1)

    print("PROGRAMMATIC_VALIDATION_OK")


if __name__ == "__main__":
    main()
