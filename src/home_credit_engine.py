from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STAGE_ORDER = ["stable", "vulnerable", "slipping", "critical"]

RISK_FEATURES = {
    "cc_util_recent": ("recent credit-card utilization", 0.18, "high"),
    "cc_util_deterioration": ("utilization deterioration", 0.12, "high"),
    "cc_drawings_atm_ratio": ("cash-advance dependence", 0.12, "high"),
    "inst_late_recent": ("recent installment lateness", 0.16, "high"),
    "inst_late_deterioration": ("installment deterioration", 0.10, "high"),
    "bureau_overdue_to_debt_ratio": ("bureau overdue burden", 0.09, "high"),
    "bureau_recent_dpd": ("recent bureau delinquency", 0.07, "high"),
    "recent_rejection_intensity": ("recent rejection intensity", 0.05, "high"),
    "red_flags_count": ("red-flag count", 0.05, "high"),
    "total_debt_to_income": ("debt-to-income pressure", 0.06, "high"),
}

PROTECTIVE_FEATURES = {
    "monthly_payment_capacity": ("payment capacity", 0.06, "low"),
    "payment_behavior_score": ("payment behavior quality", 0.05, "low"),
    "cc_min_payment_ratio": ("minimum payment coverage", 0.04, "low"),
    "financial_engagement_score": ("financial engagement", 0.03, "low"),
    "positive_signals_count": ("positive signal count", 0.02, "low"),
}


@dataclass(frozen=True)
class HomeCreditAssessment:
    customer_id: str
    distress_score: float
    distress_stage: str
    early_warning_probability: float
    risk_drivers: list[str]
    evidence: list[str]
    recommended_actions: list[str]
    peer_summary: str


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _rank_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()

    for feature_name, (_, _, direction) in {**RISK_FEATURES, **PROTECTIVE_FEATURES}.items():
        percentile_name = f"{feature_name}__percentile"
        ascending = direction == "high"
        scored[percentile_name] = scored[feature_name].rank(pct=True, ascending=ascending)

    contribution_total = 0.0
    scored["distress_score"] = 0.0
    for feature_name, (_, weight, _) in RISK_FEATURES.items():
        percentile_name = f"{feature_name}__percentile"
        scored[f"{feature_name}__contribution"] = scored[percentile_name] * weight
        scored["distress_score"] += scored[f"{feature_name}__contribution"]
        contribution_total += weight

    for feature_name, (_, weight, _) in PROTECTIVE_FEATURES.items():
        percentile_name = f"{feature_name}__percentile"
        scored[f"{feature_name}__contribution"] = scored[percentile_name] * weight
        scored["distress_score"] += scored[f"{feature_name}__contribution"]
        contribution_total += weight

    scored["distress_score"] = (scored["distress_score"] / contribution_total).clip(0.0, 1.0)
    scored["early_warning_probability"] = (0.08 + 0.84 * scored["distress_score"]).clip(0.02, 0.98)
    scored["distress_stage"] = scored["distress_score"].apply(score_to_stage)
    return scored


def score_to_stage(score: float) -> str:
    if score < 0.30:
        return "stable"
    if score < 0.48:
        return "vulnerable"
    if score < 0.68:
        return "slipping"
    return "critical"


def score_home_credit_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    return _rank_percentiles(frame)


def build_peer_subset(scored_frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    peers = scored_frame[scored_frame["peer_group"] == row["peer_group"]]
    if len(peers) < 30:
        peers = scored_frame[scored_frame["income_band"] == row["income_band"]]
    if len(peers) < 30:
        peers = scored_frame
    return peers


def _peer_percentile(peers: pd.DataFrame, row: pd.Series, feature_name: str) -> int:
    value = float(row[feature_name])
    return int(round((peers[feature_name] <= value).mean() * 100))


def _format_pct(value: float) -> str:
    return f"{value:.0%}"


def build_evidence(scored_frame: pd.DataFrame, row: pd.Series) -> tuple[list[str], str]:
    peers = build_peer_subset(scored_frame, row)
    peer_group_label = str(row["peer_group"])

    evidence: list[str] = []

    cc_recent_pct = _peer_percentile(peers, row, "cc_util_recent")
    evidence.append(
        f"Recent credit-card utilization is {_format_pct(row['cc_util_recent'])}, around the {cc_recent_pct}th percentile inside the peer group `{peer_group_label}`."
    )

    util_delta = float(row["cc_util_recent"] - row["cc_utilization"])
    if util_delta > 0.03:
        evidence.append(
            f"Utilization rose by {util_delta:.0%} versus historical behavior, which indicates worsening revolving debt pressure."
        )

    inst_pct = _peer_percentile(peers, row, "inst_late_recent")
    evidence.append(
        f"Recent installment lateness is {_format_pct(row['inst_late_recent'])}, near the {inst_pct}th peer percentile."
    )

    if float(row["inst_late_deterioration"]) > 0.03:
        evidence.append(
            f"Installment performance deteriorated by {_format_pct(row['inst_late_deterioration'])} relative to older periods."
        )

    cash_pct = _peer_percentile(peers, row, "cc_drawings_atm_ratio")
    if float(row["cc_drawings_atm_ratio"]) > 0.08:
        evidence.append(
            f"Cash-advance dependence is {_format_pct(row['cc_drawings_atm_ratio'])}, roughly the {cash_pct}th percentile among peers."
        )

    if float(row["bureau_overdue_to_debt_ratio"]) > 0.05:
        evidence.append(
            f"Bureau overdue burden is {_format_pct(row['bureau_overdue_to_debt_ratio'])}, suggesting external credit stress beyond this lender alone."
        )

    if float(row["recent_rejection_intensity"]) > 0.08:
        evidence.append(
            f"Recent rejection intensity is elevated at {_format_pct(row['recent_rejection_intensity'])}, which may indicate recent credit-seeking stress."
        )

    payment_capacity_pct = _peer_percentile(peers, row, "monthly_payment_capacity")
    evidence.append(
        f"Monthly payment capacity sits at {_format_pct(row['monthly_payment_capacity'])}, only around the {payment_capacity_pct}th percentile on a better-is-higher basis."
    )

    peer_summary = (
        f"Peer group: {peer_group_label}. "
        f"Compared against {len(peers):,} similar records using income band and contract context."
    )
    return evidence[:6], peer_summary


def build_driver_list(row: pd.Series) -> list[str]:
    contributions: list[tuple[float, str]] = []

    for feature_name, (label, _, _) in {**RISK_FEATURES, **PROTECTIVE_FEATURES}.items():
        contribution = float(row[f"{feature_name}__contribution"])
        contributions.append((contribution, label))

    contributions.sort(reverse=True, key=lambda item: item[0])
    return [label for _, label in contributions[:4]]


def build_recommendations(row: pd.Series, peers: pd.DataFrame, stage: str) -> list[str]:
    recommendations: list[str] = []

    if float(row["cc_util_recent"]) > 0.70:
        target_util = max(0.55, float(peers["cc_util_recent"].median()))
        recommendations.append(
            f"Reduce revolving exposure and target credit-card utilization closer to {_format_pct(target_util)} from the current {_format_pct(row['cc_util_recent'])}."
        )

    if float(row["inst_late_recent"]) > 0.20 or float(row["inst_late_deterioration"]) > 0.05:
        recommendations.append(
            "Prioritize repayment stabilization: move this customer to autopay, reminder, or restructure outreach before lateness compounds."
        )

    if float(row["cc_drawings_atm_ratio"]) > 0.12:
        recommendations.append(
            "Treat cash-advance usage as a liquidity warning signal and replace it with a hardship, budgeting, or installment support option."
        )

    if float(row["monthly_payment_capacity"]) < 0.35:
        recommendations.append(
            "Offer a payment plan aligned to current capacity, because available payment headroom is materially below peer levels."
        )

    if float(row["recent_rejection_intensity"]) > 0.08 or float(row["prev_recent_refused_count"]) >= 1:
        recommendations.append(
            "Avoid new marketing offers for this record and route future credit decisions through a stricter review path."
        )

    if float(row["bureau_overdue_to_debt_ratio"]) > 0.05 or float(row["bureau_recent_dpd"]) >= 1:
        recommendations.append(
            "Monitor external-credit deterioration closely; bureau stress suggests this is not an isolated account-level issue."
        )

    if len(recommendations) < 3:
        backfill = {
            "stable": [
                "Continue passive monitoring and watch for any break in payment consistency.",
                "Send personalized wellness tips instead of aggressive interventions.",
            ],
            "vulnerable": [
                "Send a soft reminder and nudge the customer toward on-time payment behavior.",
                "Increase monitoring cadence for the next statement cycle.",
            ],
            "slipping": [
                "Trigger proactive outreach before the customer becomes chronically late.",
                "Temporarily pause new exposure growth until signals stabilize.",
            ],
            "critical": [
                "Escalate to human review or hardship operations immediately.",
                "Review line strategy and active-risk containment options.",
            ],
        }[stage]
        for item in backfill:
            if item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= 4:
                break

    return recommendations[:4]


def assess_home_credit_customer(scored_frame: pd.DataFrame, customer_row: pd.Series) -> HomeCreditAssessment:
    peers = build_peer_subset(scored_frame, customer_row)
    evidence, peer_summary = build_evidence(scored_frame, customer_row)
    stage = str(customer_row["distress_stage"])
    recommendations = build_recommendations(customer_row, peers, stage)
    drivers = build_driver_list(customer_row)

    return HomeCreditAssessment(
        customer_id=str(customer_row["customer_id"]),
        distress_score=round(float(customer_row["distress_score"]), 4),
        distress_stage=stage,
        early_warning_probability=round(float(customer_row["early_warning_probability"]), 4),
        risk_drivers=drivers,
        evidence=evidence,
        recommended_actions=recommendations,
        peer_summary=peer_summary,
    )
