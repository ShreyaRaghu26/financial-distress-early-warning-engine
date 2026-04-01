from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentProfile:
    monthly_income_range: tuple[int, int]
    expense_ratio_range: tuple[float, float]
    utilization_range: tuple[float, float]
    payment_ratio_range: tuple[float, float]
    cash_advance_range: tuple[float, float]
    activity_range: tuple[float, float]
    days_late_range: tuple[int, int]
    trend_strength: float


SEGMENT_PROFILES = {
    "stable": SegmentProfile((7000, 11500), (0.42, 0.60), (0.10, 0.35), (0.18, 0.38), (0.00, 0.04), (0.90, 1.06), (0, 1), -0.010),
    "vulnerable": SegmentProfile((5200, 9200), (0.58, 0.76), (0.28, 0.58), (0.11, 0.22), (0.03, 0.10), (0.76, 0.95), (1, 4), 0.010),
    "slipping": SegmentProfile((4200, 7600), (0.70, 0.88), (0.48, 0.76), (0.05, 0.15), (0.08, 0.18), (0.56, 0.84), (3, 8), 0.024),
    "critical": SegmentProfile((3000, 6200), (0.82, 1.00), (0.68, 0.96), (0.02, 0.08), (0.16, 0.30), (0.40, 0.72), (6, 14), 0.038),
}


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def generate_behavior_data(customers: int = 250, months: int = 9, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segment_choices = rng.choice(
        ["stable", "vulnerable", "slipping", "critical"],
        size=customers,
        p=[0.44, 0.28, 0.18, 0.10],
    )

    records: list[dict[str, object]] = []

    for customer_index, segment in enumerate(segment_choices, start=1):
        profile = SEGMENT_PROFILES[str(segment)]
        monthly_income_base = float(rng.integers(*profile.monthly_income_range))
        expense_ratio_base = float(rng.uniform(*profile.expense_ratio_range))
        credit_limit = float(rng.integers(5000, 20000))
        base_utilization = float(rng.uniform(*profile.utilization_range))
        base_payment_ratio = float(rng.uniform(*profile.payment_ratio_range))
        base_cash_advance = float(rng.uniform(*profile.cash_advance_range))
        base_activity = float(rng.uniform(*profile.activity_range))
        base_days_late = int(rng.integers(profile.days_late_range[0], profile.days_late_range[1] + 1))
        seasonal_shift = float(rng.uniform(0.0, 6.28))

        for month in range(1, months + 1):
            trend = profile.trend_strength * month
            seasonal = np.sin((month / 12.0) * 6.28 + seasonal_shift)

            income = monthly_income_base * (1.0 + rng.normal(0.0, 0.03) - max(trend, 0.0) * 0.10)
            expenses = monthly_income_base * expense_ratio_base * (
                1.0 + rng.normal(0.0, 0.05) + max(trend, 0.0) * 0.22
            )
            utilization = _clip(base_utilization + trend + seasonal * 0.03 + rng.normal(0.0, 0.025), 0.02, 0.99)
            payment_ratio = _clip(base_payment_ratio - trend * 0.55 + rng.normal(0.0, 0.015), 0.01, 0.65)
            cash_advance_ratio = _clip(base_cash_advance + max(trend, 0.0) * 0.45 + rng.normal(0.0, 0.02), 0.0, 0.5)
            account_activity_index = _clip(base_activity - max(trend, 0.0) * 1.10 + seasonal * 0.04 + rng.normal(0.0, 0.03), 0.25, 1.20)
            days_late = int(round(_clip(base_days_late + max(trend, 0.0) * 110.0 + rng.normal(0.0, 1.5), 0.0, 30.0)))

            records.append(
                {
                    "customer_id": f"CUST-{customer_index:04d}",
                    "month": month,
                    "seed_segment": segment,
                    "monthly_income": round(income, 2),
                    "monthly_expenses": round(expenses, 2),
                    "credit_limit": round(credit_limit, 2),
                    "revolving_balance": round(credit_limit * utilization, 2),
                    "payment_amount": round(credit_limit * utilization * payment_ratio, 2),
                    "utilization": round(utilization, 4),
                    "payment_to_balance_ratio": round(payment_ratio, 4),
                    "cash_advance_ratio": round(cash_advance_ratio, 4),
                    "account_activity_index": round(account_activity_index, 4),
                    "days_late": days_late,
                }
            )

    frame = pd.DataFrame.from_records(records).sort_values(["customer_id", "month"]).reset_index(drop=True)
    frame["expense_to_income_ratio"] = (frame["monthly_expenses"] / frame["monthly_income"]).round(4)
    return frame
