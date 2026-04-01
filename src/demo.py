from __future__ import annotations

import json

from src.data_generator import generate_behavior_data
from src.llm_advisor import generate_advice
from src.risk_engine import assess_customer, latest_customer_snapshot
from src.scenario_engine import apply_scenario


def print_json(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


def main() -> None:
    behavior_data = generate_behavior_data(customers=150, months=6, seed=42)
    latest_snapshot = latest_customer_snapshot(behavior_data)

    target = latest_snapshot.sort_values(["distress_score", "early_warning_probability"], ascending=False).iloc[0]
    baseline_assessment = assess_customer(target)

    baseline_payload = {
        "customer_id": baseline_assessment.customer_id,
        "month": baseline_assessment.month,
        "distress_score": baseline_assessment.distress_score,
        "distress_stage": baseline_assessment.distress_stage,
        "early_warning_probability": baseline_assessment.early_warning_probability,
        "drivers": baseline_assessment.drivers,
        "recommended_actions": baseline_assessment.recommended_actions,
    }
    baseline_advice = generate_advice(baseline_payload)

    scenario_row = apply_scenario(target, "restructure_offer")
    scenario_assessment = assess_customer(scenario_row)
    scenario_payload = {
        "customer_id": scenario_assessment.customer_id,
        "month": scenario_assessment.month,
        "distress_score": scenario_assessment.distress_score,
        "distress_stage": scenario_assessment.distress_stage,
        "early_warning_probability": scenario_assessment.early_warning_probability,
        "drivers": scenario_assessment.drivers,
        "recommended_actions": scenario_assessment.recommended_actions,
    }
    scenario_advice = generate_advice(scenario_payload)

    summary = {
        "customers_generated": int(latest_snapshot.shape[0]),
        "target_customer": baseline_assessment.customer_id,
        "baseline_stage": baseline_assessment.distress_stage,
        "baseline_probability": baseline_assessment.early_warning_probability,
        "post_restructure_stage": scenario_assessment.distress_stage,
        "post_restructure_probability": scenario_assessment.early_warning_probability,
    }

    print_json("Run Summary", summary)
    print_json("Baseline Assessment", baseline_payload)
    print_json("LLM/Fallback Advice", baseline_advice)
    print_json("Scenario Assessment", scenario_payload)
    print_json("Scenario Advice", scenario_advice)


if __name__ == "__main__":
    main()
