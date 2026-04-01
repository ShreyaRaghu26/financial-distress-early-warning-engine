from __future__ import annotations

import numpy as np
import pandas as pd


HOME_CREDIT_PARQUET_PATH = "hf://datasets/jamirc/home_credit_default_risk/home_credit_train_ready.parquet"
HOME_CREDIT_SOURCE_LABEL = "Home Credit Default Risk (Hugging Face mirror of Kaggle competition data)"

SELECTED_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "total_debt_to_income",
    "monthly_payment_capacity",
    "deterioration_composite_score",
    "cc_utilization",
    "cc_util_recent",
    "cc_drawings_atm_ratio",
    "cc_min_payment_ratio",
    "inst_late_ratio_total",
    "inst_late_recent",
    "inst_late_deterioration",
    "bureau_overdue_to_debt_ratio",
    "bureau_recent_dpd",
    "bureau_dpd_ratio",
    "recent_rejection_intensity",
    "prev_recent_refused_count",
    "prev_approval_rate_6m",
    "red_flags_count",
    "positive_signals_count",
    "credit_mix_score",
    "financial_engagement_score",
    "payment_behavior_score",
]


def _clip_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.astype("float64").clip(lower=lower, upper=upper)


def load_home_credit_sample(sample_size: int = 2000, seed: int = 42) -> pd.DataFrame:
    frame = pd.read_parquet(HOME_CREDIT_PARQUET_PATH, columns=SELECTED_COLUMNS)
    if sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    for column in numeric_columns:
        median_value = frame[column].median()
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
        frame[column] = frame[column].fillna(fill_value)

    frame["customer_id"] = frame["SK_ID_CURR"].astype(str)
    frame["age_years"] = (-frame["DAYS_BIRTH"] / 365.25).round(1)
    frame["employment_years"] = ((-frame["DAYS_EMPLOYED"]).clip(lower=0) / 365.25).round(1)
    frame["monthly_income"] = (frame["AMT_INCOME_TOTAL"] / 12.0).round(2)
    frame["annuity_to_income_ratio"] = _clip_series(frame["AMT_ANNUITY"] / frame["AMT_INCOME_TOTAL"].replace(0, np.nan), 0.0, 3.0)
    frame["credit_to_income_ratio"] = _clip_series(frame["AMT_CREDIT"] / frame["AMT_INCOME_TOTAL"].replace(0, np.nan), 0.0, 20.0)
    frame["cc_util_deterioration"] = (frame["cc_util_recent"] - frame["cc_utilization"]).clip(lower=0.0, upper=1.0)

    ratio_columns = [
        "total_debt_to_income",
        "monthly_payment_capacity",
        "deterioration_composite_score",
        "cc_utilization",
        "cc_util_recent",
        "cc_util_deterioration",
        "cc_drawings_atm_ratio",
        "cc_min_payment_ratio",
        "inst_late_ratio_total",
        "inst_late_recent",
        "inst_late_deterioration",
        "bureau_overdue_to_debt_ratio",
        "bureau_dpd_ratio",
        "recent_rejection_intensity",
        "prev_approval_rate_6m",
        "credit_mix_score",
        "financial_engagement_score",
        "payment_behavior_score",
        "annuity_to_income_ratio",
    ]
    for column in ratio_columns:
        frame[column] = _clip_series(frame[column], 0.0, 1.0)

    frame["bureau_recent_dpd"] = _clip_series(frame["bureau_recent_dpd"], 0.0, 10.0)
    frame["prev_recent_refused_count"] = _clip_series(frame["prev_recent_refused_count"], 0.0, 10.0)
    frame["red_flags_count"] = _clip_series(frame["red_flags_count"], 0.0, 10.0)
    frame["positive_signals_count"] = _clip_series(frame["positive_signals_count"], 0.0, 10.0)

    frame["income_band"] = pd.qcut(
        frame["AMT_INCOME_TOTAL"].rank(method="first"),
        q=4,
        labels=["low", "mid-low", "mid-high", "high"],
    ).astype(str)
    frame["peer_group"] = (
        frame["income_band"] + " | " + frame["NAME_CONTRACT_TYPE"].astype(str) + " | " + frame["NAME_INCOME_TYPE"].astype(str)
    )
    frame["data_source"] = HOME_CREDIT_SOURCE_LABEL
    return frame.reset_index(drop=True)
