from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier


MODEL_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "age_years",
    "employment_years",
    "credit_to_income_ratio",
    "annuity_to_income_ratio",
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

FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "credit amount",
    "AMT_ANNUITY": "annuity burden",
    "age_years": "age",
    "employment_years": "employment tenure",
    "credit_to_income_ratio": "credit-to-income ratio",
    "annuity_to_income_ratio": "annuity-to-income ratio",
    "total_debt_to_income": "debt-to-income pressure",
    "monthly_payment_capacity": "payment capacity",
    "deterioration_composite_score": "deterioration composite score",
    "cc_utilization": "historical card utilization",
    "cc_util_recent": "recent card utilization",
    "cc_util_deterioration": "utilization deterioration",
    "cc_drawings_atm_ratio": "cash-advance dependence",
    "cc_min_payment_ratio": "minimum payment coverage",
    "inst_late_ratio_total": "historical installment lateness",
    "inst_late_recent": "recent installment lateness",
    "inst_late_deterioration": "installment deterioration",
    "bureau_overdue_to_debt_ratio": "bureau overdue burden",
    "bureau_recent_dpd": "recent bureau DPD",
    "bureau_dpd_ratio": "bureau DPD ratio",
    "recent_rejection_intensity": "recent rejection intensity",
    "prev_recent_refused_count": "recent refusals",
    "prev_approval_rate_6m": "recent approval rate",
    "red_flags_count": "red flags",
    "positive_signals_count": "positive signals",
    "credit_mix_score": "credit mix score",
    "financial_engagement_score": "financial engagement",
    "payment_behavior_score": "payment behavior quality",
}


@dataclass(frozen=True)
class HomeCreditModelBundle:
    model: XGBClassifier
    feature_columns: list[str]
    roc_auc: float
    average_precision: float
    train_size: int
    test_size: int
    target_rate: float


def probability_to_stage(probability: float) -> str:
    if probability < 0.20:
        return "stable"
    if probability < 0.35:
        return "vulnerable"
    if probability < 0.55:
        return "slipping"
    return "critical"


def train_home_credit_model(frame: pd.DataFrame, seed: int = 42) -> HomeCreditModelBundle:
    features = frame[MODEL_FEATURES].astype("float64")
    target = frame["TARGET"].astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=seed,
        stratify=target,
    )
    positive_rate = max(float(y_train.mean()), 1e-6)
    scale_pos_weight = max((1.0 - positive_rate) / positive_rate, 1.0)

    model = XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)

    test_predictions = model.predict_proba(x_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, test_predictions))
    avg_precision = float(average_precision_score(y_test, test_predictions))

    return HomeCreditModelBundle(
        model=model,
        feature_columns=MODEL_FEATURES,
        roc_auc=roc_auc,
        average_precision=avg_precision,
        train_size=int(len(x_train)),
        test_size=int(len(x_test)),
        target_rate=float(target.mean()),
    )


def add_model_scores(frame: pd.DataFrame, bundle: HomeCreditModelBundle) -> pd.DataFrame:
    scored = frame.copy()
    features = scored[bundle.feature_columns].astype("float64")
    scored["ml_default_probability"] = bundle.model.predict_proba(features)[:, 1]
    scored["blended_probability"] = (
        0.65 * scored["ml_default_probability"] + 0.35 * scored["early_warning_probability"]
    ).clip(0.0, 0.98)
    scored["ml_risk_stage"] = scored["ml_default_probability"].apply(probability_to_stage)
    scored["blended_stage"] = scored["blended_probability"].apply(probability_to_stage)
    return scored


def explain_model_prediction(row: pd.Series, bundle: HomeCreditModelBundle, top_n: int = 4) -> list[dict[str, float | str]]:
    feature_frame = pd.DataFrame([row[bundle.feature_columns].astype("float64")], columns=bundle.feature_columns)
    contributions = bundle.model.get_booster().predict(
        DMatrix(feature_frame, feature_names=bundle.feature_columns),
        pred_contribs=True,
    )[0]

    explanations: list[dict[str, float | str]] = []
    for feature_name, contribution in zip(bundle.feature_columns, contributions[:-1]):
        explanations.append(
            {
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name),
                "contribution": float(contribution),
                "value": float(row[feature_name]),
            }
        )

    explanations.sort(key=lambda item: item["contribution"], reverse=True)
    top_positive = [item for item in explanations if float(item["contribution"]) > 0][:top_n]
    if top_positive:
        return top_positive
    return explanations[:top_n]
