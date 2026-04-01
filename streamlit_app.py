from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.home_credit_actions import ASSUMED_LOSS_GIVEN_DEFAULT, build_portfolio_queue, rank_actions
from src.home_credit_engine import (
    RISK_FEATURES,
    STAGE_ORDER,
    assess_home_credit_customer,
    build_peer_subset,
    score_home_credit_dataset,
)
from src.home_credit_loader import HOME_CREDIT_SOURCE_LABEL, load_home_credit_sample
from src.home_credit_model import (
    add_model_scores,
    explain_model_prediction,
    train_home_credit_model,
)
from src.home_credit_scenarios import apply_home_credit_scenario
from src.llm_advisor import generate_advice


PAGE_TITLE = "Financial Distress Early Warning Engine"
STAGE_COLORS = {
    "stable": "#2ca25f",
    "vulnerable": "#f2c94c",
    "slipping": "#f58518",
    "critical": "#e45756",
}
SCENARIO_LABELS = {
    "none": "No scenario",
    "income_drop": "Income drops by 15%",
    "missed_payment": "Miss one payment",
    "cash_advance_spike": "Cash advance spike",
    "paydown_support": "Debt paydown support",
    "restructure_offer": "Restructure offer",
}
SNAPSHOT_FIELDS = [
    ("Annual income", "AMT_INCOME_TOTAL", "currency"),
    ("Monthly income", "monthly_income", "currency"),
    ("Credit amount", "AMT_CREDIT", "currency"),
    ("Annuity", "AMT_ANNUITY", "currency"),
    ("Debt / income", "total_debt_to_income", "percent"),
    ("Payment capacity", "monthly_payment_capacity", "percent"),
    ("Recent CC utilization", "cc_util_recent", "percent"),
    ("Historical CC utilization", "cc_utilization", "percent"),
    ("Utilization deterioration", "cc_util_deterioration", "percent"),
    ("Cash advance ratio", "cc_drawings_atm_ratio", "percent"),
    ("Recent installment lateness", "inst_late_recent", "percent"),
    ("Installment deterioration", "inst_late_deterioration", "percent"),
    ("Bureau overdue / debt", "bureau_overdue_to_debt_ratio", "percent"),
    ("Recent bureau DPD", "bureau_recent_dpd", "number"),
    ("Red flags", "red_flags_count", "number"),
    ("Positive signals", "positive_signals_count", "number"),
]


@st.cache_data(show_spinner=False)
def load_data(sample_size: int, seed: int) -> pd.DataFrame:
    base = load_home_credit_sample(sample_size=sample_size, seed=seed)
    return score_home_credit_dataset(base)


@st.cache_resource(show_spinner=False)
def load_model_bundle(sample_size: int, seed: int):
    scored = load_data(sample_size=sample_size, seed=seed)
    return train_home_credit_model(scored, seed=seed)


@st.cache_data(show_spinner=False)
def build_cached_portfolio_queue(scored_data: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    bundle = load_model_bundle(sample_size=sample_size, seed=seed)
    return build_portfolio_queue(scored_data, bundle)


def build_payload(assessment, model_explanations: list[dict[str, float | str]]) -> dict[str, object]:
    model_evidence = [
        f"ML model contribution: {item['label']} adds {float(item['contribution']):.3f} risk at current value {float(item['value']):.2f}."
        for item in model_explanations[:3]
    ]
    return {
        "customer_id": assessment.customer_id,
        "distress_score": assessment.distress_score,
        "distress_stage": assessment.distress_stage,
        "early_warning_probability": assessment.early_warning_probability,
        "drivers": assessment.risk_drivers,
        "evidence": assessment.evidence + model_evidence,
        "peer_summary": assessment.peer_summary,
        "recommended_actions": assessment.recommended_actions,
    }


def format_value(value: float, kind: str) -> str:
    if kind == "currency":
        return f"${value:,.0f}"
    if kind == "percent":
        return f"{value:.0%}"
    return f"{value:,.0f}"


def stage_delta_text(before: str, after: str) -> str:
    before_idx = STAGE_ORDER.index(before)
    after_idx = STAGE_ORDER.index(after)
    if after_idx > before_idx:
        return "Scenario worsens the expected distress stage."
    if after_idx < before_idx:
        return "Scenario improves the expected distress stage."
    return "Scenario changes severity, but not the overall stage."


def render_stage_badge(stage: str) -> None:
    st.markdown(
        f"""
        <div style="padding:10px 14px;border-radius:10px;background:{STAGE_COLORS[stage]}20;border:1px solid {STAGE_COLORS[stage]};">
            <span style="color:{STAGE_COLORS[stage]};font-weight:700;text-transform:uppercase;">{stage}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_peer_percentile_frame(scored_data: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    peers = build_peer_subset(scored_data, row)
    metrics = [
        ("Recent CC utilization", "cc_util_recent"),
        ("Utilization deterioration", "cc_util_deterioration"),
        ("Cash advance ratio", "cc_drawings_atm_ratio"),
        ("Installment lateness", "inst_late_recent"),
        ("Debt / income", "total_debt_to_income"),
        ("Payment capacity", "monthly_payment_capacity"),
    ]
    rows = []
    for label, column in metrics:
        percentile = int(round((peers[column] <= float(row[column])).mean() * 100))
        rows.append({"signal": label, "peer_percentile": percentile})
    return pd.DataFrame(rows)


def build_snapshot_frame(baseline_row: pd.Series, scenario_row: pd.Series) -> pd.DataFrame:
    records = []
    for label, column, kind in SNAPSHOT_FIELDS:
        records.append(
            {
                "metric": label,
                "baseline": format_value(float(baseline_row[column]), kind),
                "scenario": format_value(float(scenario_row[column]), kind),
            }
        )
    return pd.DataFrame(records)


def build_customer_reason_summary(assessment) -> list[str]:
    mapping = {
        "recent credit-card utilization": "You are using a large share of your available credit.",
        "utilization deterioration": "Your credit usage has been getting worse recently.",
        "cash-advance dependence": "You appear to be relying on cash advances, which often signals liquidity stress.",
        "recent installment lateness": "Recent payment timing shows missed or delayed installment behavior.",
        "installment deterioration": "Your repayment pattern has worsened compared with earlier periods.",
        "bureau overdue burden": "Other credit obligations also show overdue pressure.",
        "recent bureau delinquency": "Recent delinquency shows up in bureau behavior as well.",
        "recent rejection intensity": "Recent rejections suggest rising credit stress.",
        "red-flag count": "Several warning signals are appearing at the same time.",
        "debt-to-income pressure": "Debt pressure is high relative to income.",
        "payment capacity": "There is not much room left in your budget to absorb payments.",
        "payment behavior quality": "Payment behavior quality is weaker than healthy accounts.",
        "minimum payment coverage": "Current payments are not covering enough of the balance.",
        "financial engagement": "Account engagement suggests financial stress rather than steady usage.",
        "positive signal count": "There are not many offsetting positive signals right now.",
    }
    return [mapping.get(driver, driver) for driver in assessment.risk_drivers[:3]]


def build_customer_action_steps(ranked_actions) -> list[str]:
    steps = []
    for action in ranked_actions[:3]:
        steps.append(
            f"{action.label}: could reduce blended risk by about {action.expected_probability_drop:.0%} and avoid roughly ${action.estimated_saved_loss:,.0f} in expected loss."
        )
    return steps


def build_executive_headline(baseline_assessment, scenario_assessment, ranked_actions) -> str:
    best_action = ranked_actions[0]
    return (
        f"This customer is currently `{baseline_assessment.distress_stage}`. "
        f"The best next move is `{best_action.label}`, which is estimated to reduce blended risk by "
        f"{best_action.expected_probability_drop:.0%} and save about ${best_action.estimated_saved_loss:,.0f}."
    )


st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption(
    "A lender-friendly early warning workspace built on "
    + HOME_CREDIT_SOURCE_LABEL
    + " to spot financial stress early, recommend the next best action, and estimate business impact."
)

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    sample_size = st.slider("Portfolio sample size", min_value=400, max_value=4000, value=1400, step=200)
    customer_stage_filter = st.multiselect(
        "Customer segments to browse",
        options=STAGE_ORDER,
        default=STAGE_ORDER,
        key="customer_stage_filter_v2",
    )
    model_name = st.selectbox("LLM model", ["gpt-4.1-mini", "gpt-4o-mini"], index=0)
    api_key = st.text_input("OpenAI API key (optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    scenario_name = st.selectbox(
        "What-if scenario",
        list(SCENARIO_LABELS.keys()),
        index=1,
        format_func=lambda key: SCENARIO_LABELS[key],
    )

try:
    scored_data = load_data(sample_size=int(sample_size), seed=int(seed))
    model_bundle = load_model_bundle(sample_size=int(sample_size), seed=int(seed))
    scored_data = add_model_scores(scored_data, model_bundle)
except Exception as exc:
    st.error(f"Could not load the Home Credit dataset: {exc}")
    st.stop()

sorted_customers = scored_data.sort_values(["blended_probability", "ml_default_probability"], ascending=[False, False])
filtered_customers = sorted_customers[sorted_customers["blended_stage"].isin(customer_stage_filter)].copy()
if filtered_customers.empty:
    filtered_customers = sorted_customers.copy()
filtered_customers = filtered_customers.reset_index(drop=True)
customer_option_labels = [
    f"{row['customer_id']} | {str(row['blended_stage']).title()} | overall {float(row['blended_probability']):.0%}"
    for _, row in filtered_customers.iterrows()
]
customer_option_map = dict(zip(customer_option_labels, filtered_customers["customer_id"].tolist()))
selected_customer_label = st.selectbox(
    "Customer",
    customer_option_labels,
    index=0,
    key=f"customer_select_{'_'.join(customer_stage_filter) or 'all'}",
)
selected_customer_id = customer_option_map[selected_customer_label]
st.caption(
    f"Showing {len(filtered_customers)} customers in the selected segment(s): "
    + ", ".join(stage.title() for stage in customer_stage_filter)
)
baseline_row = sorted_customers[sorted_customers["customer_id"] == selected_customer_id].iloc[0].copy()
baseline_assessment = assess_home_credit_customer(scored_data, baseline_row)
baseline_model_explanations = explain_model_prediction(baseline_row, model_bundle)

if scenario_name == "none":
    scenario_row = baseline_row.copy()
    scenario_scored_data = scored_data
else:
    scenario_raw = apply_home_credit_scenario(baseline_row, scenario_name)
    augmented = pd.concat([scored_data.drop(index=baseline_row.name), pd.DataFrame([scenario_raw])], ignore_index=True)
    scenario_scored_data = add_model_scores(score_home_credit_dataset(augmented), model_bundle)
    scenario_row = scenario_scored_data[scenario_scored_data["customer_id"] == selected_customer_id].iloc[0].copy()
scenario_assessment = assess_home_credit_customer(scenario_scored_data, scenario_row)
scenario_model_explanations = explain_model_prediction(scenario_row, model_bundle)
ranked_actions = rank_actions(scored_data, model_bundle, baseline_row, top_n=3)
portfolio_queue = build_cached_portfolio_queue(scored_data, sample_size=int(sample_size), seed=int(seed))

baseline_payload = build_payload(baseline_assessment, baseline_model_explanations)
scenario_payload = build_payload(scenario_assessment, scenario_model_explanations)
baseline_advice = generate_advice(baseline_payload, model=model_name)
scenario_advice = generate_advice(scenario_payload, model=model_name)

stage_counts = scored_data["blended_stage"].value_counts().reindex(STAGE_ORDER, fill_value=0).reset_index()
stage_counts.columns = ["distress_stage", "customers"]
stage_chart = px.bar(
    stage_counts,
    x="distress_stage",
    y="customers",
    color="distress_stage",
    title="Portfolio Stage Distribution (Blended)",
    color_discrete_map=STAGE_COLORS,
)

comparison_frame = pd.DataFrame(
    {
        "state": ["Baseline", "Scenario"],
        "blended_probability": [
            float(baseline_row["blended_probability"]),
            float(scenario_row["blended_probability"]),
        ],
    }
)
comparison_chart = px.bar(
    comparison_frame,
    x="state",
    y="blended_probability",
    color="state",
    title="Baseline vs Scenario Blended Risk",
    color_discrete_map={"Baseline": "#4c78a8", "Scenario": "#f58518"},
)
comparison_chart.update_yaxes(tickformat=".0%")

driver_rows = []
for feature_name, (label, _, _) in RISK_FEATURES.items():
    driver_rows.append(
        {
            "driver": label,
            "contribution": float(baseline_row[f"{feature_name}__contribution"]),
        }
    )
driver_frame = pd.DataFrame(driver_rows).sort_values("contribution", ascending=False).head(6)
driver_chart = px.bar(
    driver_frame,
    x="contribution",
    y="driver",
    orientation="h",
    title="Top Driver Contributions",
    color="contribution",
    color_continuous_scale="OrRd",
)

peer_percentile_frame = build_peer_percentile_frame(scored_data, baseline_row)
peer_chart = px.bar(
    peer_percentile_frame,
    x="signal",
    y="peer_percentile",
    title="Peer Percentile Benchmark",
    color="peer_percentile",
    color_continuous_scale="YlOrRd",
)
peer_chart.update_yaxes(range=[0, 100], ticksuffix="%")

historical_recent_frame = pd.DataFrame(
    {
        "signal": ["CC utilization", "Installment lateness"],
        "historical": [float(baseline_row["cc_utilization"]), float(baseline_row["inst_late_ratio_total"])],
        "recent": [float(baseline_row["cc_util_recent"]), float(baseline_row["inst_late_recent"])],
        "scenario": [float(scenario_row["cc_util_recent"]), float(scenario_row["inst_late_recent"])],
    }
).melt(id_vars="signal", var_name="period", value_name="value")
history_chart = px.bar(
    historical_recent_frame,
    x="signal",
    y="value",
    color="period",
    barmode="group",
    title="Historical vs Recent vs Scenario Signals",
    color_discrete_map={"historical": "#7f8c8d", "recent": "#4c78a8", "scenario": "#e45756"},
)
history_chart.update_yaxes(tickformat=".0%")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
metric_col1.metric("Current risk band", baseline_assessment.distress_stage.title())
metric_col2.metric("Scenario risk band", scenario_assessment.distress_stage.title())
metric_col3.metric("Early warning score", f"{baseline_assessment.early_warning_probability:.0%}")
metric_col4.metric(
    "Expected default risk",
    f"{float(baseline_row['ml_default_probability']):.0%}",
)
metric_col5.metric(
    "Overall priority score",
    f"{float(baseline_row['blended_probability']):.0%}",
    delta=f"{float(scenario_row['blended_probability']) - float(baseline_row['blended_probability']):+.0%}",
)

st.info(f"Real data foundation: {HOME_CREDIT_SOURCE_LABEL}")
st.caption(
    f"Model health check: XGBoost on the selected sample. ROC-AUC `{model_bundle.roc_auc:.3f}`, "
    f"Average Precision `{model_bundle.average_precision:.3f}`, target rate `{model_bundle.target_rate:.1%}`. "
    f"Business impact uses assumed loss-given-default of `{ASSUMED_LOSS_GIVEN_DEFAULT:.0%}`."
)
if scenario_name == "none":
    st.info("No scenario is applied, so baseline and scenario values are intentionally identical.")

filtered_queue = pd.DataFrame()
if not portfolio_queue.empty:
    filtered_queue = portfolio_queue[
        portfolio_queue["baseline_stage"].isin(customer_stage_filter)
    ].copy()

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
overview_col1.metric("Current stage", baseline_assessment.distress_stage.title())
overview_col2.metric("Best next action", ranked_actions[0].label)
overview_col3.metric("Estimated value at stake", f"${ranked_actions[0].expected_loss_before:,.0f}")
overview_col4.metric("Potential value saved", f"${ranked_actions[0].estimated_saved_loss:,.0f}")

executive_tab, customer_tab, analyst_tab = st.tabs(["Executive View", "Customer Guidance", "Analyst View"])

with executive_tab:
    st.subheader("What The Team Should Know Today")
    st.write(build_executive_headline(baseline_assessment, scenario_assessment, ranked_actions))
    exec_col1, exec_col2 = st.columns([1.1, 1.4])
    with exec_col1:
        render_stage_badge(baseline_assessment.distress_stage)
        st.markdown("**Why this account needs attention**")
        for line in build_customer_reason_summary(baseline_assessment):
            st.write(f"- {line}")
        st.markdown("**Recommended move**")
        st.write(ranked_actions[0].description)
        st.write(
            f"If the company acts now, expected loss could move from `${ranked_actions[0].expected_loss_before:,.0f}` "
            f"to `${ranked_actions[0].expected_loss_after:,.0f}`."
        )
    with exec_col2:
        st.plotly_chart(comparison_chart, use_container_width=True, key="exec_comparison_chart")
        st.plotly_chart(stage_chart, use_container_width=True, key="exec_stage_chart")

    st.subheader("Manager Intervention Summary")
    if filtered_queue.empty:
        st.warning("No queue records match the selected global stage filter.")
    else:
        top_action = (
            filtered_queue.groupby("best_action", as_index=False)
            .agg(customers=("customer_id", "count"), saved_loss=("estimated_saved_loss", "sum"))
            .sort_values(["saved_loss", "customers"], ascending=[False, False])
            .iloc[0]
        )
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Customers to work", f"{len(filtered_queue)}")
        summary_col2.metric("Potential value protected", f"${filtered_queue['estimated_saved_loss'].sum():,.0f}")
        summary_col3.metric("Average priority improvement", f"{filtered_queue['risk_drop'].mean():+.1%}")
        summary_col4.metric("Most useful action", str(top_action["best_action"]))

        manager_col1, manager_col2 = st.columns(2)
        with manager_col1:
            action_mix = (
                filtered_queue.groupby("best_action", as_index=False)
                .agg(customers=("customer_id", "count"), saved_loss=("estimated_saved_loss", "sum"))
                .sort_values("saved_loss", ascending=False)
            )
            action_mix_chart = px.bar(
                action_mix,
                x="best_action",
                y="saved_loss",
                color="customers",
                title="Potential Value Protected By Action",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(action_mix_chart, use_container_width=True, key="exec_action_mix_chart")
        with manager_col2:
            stage_mix = (
                filtered_queue.groupby("baseline_stage", as_index=False)
                .agg(customers=("customer_id", "count"), saved_loss=("estimated_saved_loss", "sum"))
                .sort_values("saved_loss", ascending=False)
            )
            stage_mix_chart = px.bar(
                stage_mix,
                x="baseline_stage",
                y="saved_loss",
                color="baseline_stage",
                title="Potential Value Protected By Stage",
                color_discrete_map=STAGE_COLORS,
            )
            st.plotly_chart(stage_mix_chart, use_container_width=True, key="exec_stage_mix_chart")

with customer_tab:
    st.subheader("Customer-Friendly Guidance")
    render_stage_badge(baseline_assessment.distress_stage)
    st.write(
        f"This customer is currently in the `{baseline_assessment.distress_stage}` zone. "
        f"The main goal is to stabilize payments and reduce near-term stress before the situation worsens."
    )
    st.markdown("**What may be causing stress**")
    for line in build_customer_reason_summary(baseline_assessment):
        st.write(f"- {line}")
    st.markdown("**Top 3 next steps**")
    for line in build_customer_action_steps(ranked_actions):
        st.write(f"- {line}")

    customer_col1, customer_col2 = st.columns(2)
    with customer_col1:
        st.markdown("**Suggested message for customer support**")
        st.write(baseline_advice["user_message"])
    with customer_col2:
        st.markdown("**What changes under this scenario**")
        if scenario_name == "none":
            st.write("No scenario applied. Pick a scenario in the sidebar to show how this customer could improve or worsen.")
        else:
            st.write(stage_delta_text(baseline_assessment.distress_stage, scenario_assessment.distress_stage))
            st.write(scenario_advice["user_message"])

    st.markdown("**Simple customer snapshot**")
    simple_snapshot = build_snapshot_frame(baseline_row, scenario_row).iloc[:8].copy()
    st.dataframe(simple_snapshot, use_container_width=True, hide_index=True)

with analyst_tab:
    st.subheader("Detailed Risk Evidence")
    left_col, right_col = st.columns([1.1, 1.9])
    with left_col:
        st.markdown("**Current Assessment**")
        render_stage_badge(baseline_assessment.distress_stage)
        st.write(baseline_advice["summary"])
        st.write(baseline_assessment.peer_summary)
        st.markdown("**Top Drivers**")
        for driver in baseline_assessment.risk_drivers:
            st.write(f"- {driver}")
        st.markdown("**Evidence**")
        for item in baseline_assessment.evidence:
            st.write(f"- {item}")
        st.markdown("**Top ML Risk Factors**")
        for item in baseline_model_explanations:
            st.write(f"- {item['label']}: contribution {float(item['contribution']):.3f}")
        st.markdown("**Recommended Actions**")
        for action in baseline_assessment.recommended_actions:
            st.write(f"- {action}")

        st.markdown("**Scenario Outcome**")
        render_stage_badge(scenario_assessment.distress_stage)
        if scenario_name == "none":
            st.write("No scenario applied. Select a stress or support action from the sidebar to simulate a change.")
        else:
            st.write(stage_delta_text(baseline_assessment.distress_stage, scenario_assessment.distress_stage))
        st.markdown("**Scenario Evidence**")
        for item in scenario_assessment.evidence:
            st.write(f"- {item}")
        st.markdown("**Scenario ML Risk Factors**")
        for item in scenario_model_explanations:
            st.write(f"- {item['label']}: contribution {float(item['contribution']):.3f}")
        st.markdown("**Scenario Recommendations**")
        for action in scenario_assessment.recommended_actions:
            st.write(f"- {action}")

    with right_col:
        st.plotly_chart(comparison_chart, use_container_width=True, key="analyst_comparison_chart")
        st.plotly_chart(driver_chart, use_container_width=True, key="analyst_driver_chart")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(peer_chart, use_container_width=True, key="analyst_peer_chart")
        st.plotly_chart(history_chart, use_container_width=True, key="analyst_history_chart")
    with chart_col2:
        st.subheader("Customer Snapshot")
        st.dataframe(build_snapshot_frame(baseline_row, scenario_row), use_container_width=True, hide_index=True)

    st.subheader("Action Testing Lab")
    action_cols = st.columns(3)
    for action_col, action in zip(action_cols, ranked_actions):
        with action_col:
            st.markdown(f"**{action.label}**")
            st.write(action.description)
            st.metric("Expected blended risk change", f"{action.expected_probability_drop:+.1%}")
            st.metric("Estimated saved loss", f"${action.estimated_saved_loss:,.0f}")
            st.write(f"Expected stage after action: `{action.expected_stage}`")
            st.write(
                f"Expected loss: `${action.expected_loss_before:,.0f}` -> `${action.expected_loss_after:,.0f}`"
            )
            for note in action.rationale:
                st.write(f"- {note}")

    st.subheader("Portfolio Intervention Worklist")
    if portfolio_queue.empty:
        st.write("No high-risk queue could be generated for the current sample.")
    else:
        available_actions = sorted(portfolio_queue["best_action"].unique())
        selected_actions = st.multiselect(
            "Filter worklist by action",
            options=available_actions,
            default=available_actions,
            key="queue_action_filter_v2",
        )
        st.caption("This worklist follows the global customer segment filter from the sidebar.")

        analyst_queue = filtered_queue[filtered_queue["best_action"].isin(selected_actions)].copy()
        if analyst_queue.empty:
            st.warning("No queue records match the current filters.")
        else:
            queue_chart = px.bar(
                analyst_queue,
                x="customer_id",
                y="estimated_saved_loss",
                color="baseline_stage",
                title="Top Accounts By Potential Value Protected",
                color_discrete_map=STAGE_COLORS,
                hover_data=["best_action", "risk_drop"],
            )
            queue_col1, queue_col2 = st.columns([1.1, 1.6])
            with queue_col1:
                st.plotly_chart(queue_chart, use_container_width=True, key="analyst_queue_chart")
            with queue_col2:
                queue_display = analyst_queue.copy()
                queue_display["baseline_blended_probability"] = queue_display["baseline_blended_probability"].map(lambda x: f"{x:.0%}")
                queue_display["baseline_ml_probability"] = queue_display["baseline_ml_probability"].map(lambda x: f"{x:.0%}")
                queue_display["credit_amount"] = queue_display["credit_amount"].map(lambda x: f"${x:,.0f}")
                queue_display["risk_drop"] = queue_display["risk_drop"].map(lambda x: f"{x:+.1%}")
                queue_display["estimated_saved_loss"] = queue_display["estimated_saved_loss"].map(lambda x: f"${x:,.0f}")
                st.dataframe(
                    queue_display[
                        [
                            "customer_id",
                            "baseline_stage",
                            "baseline_blended_probability",
                            "baseline_ml_probability",
                            "credit_amount",
                            "best_action",
                            "expected_stage_after_action",
                            "risk_drop",
                            "estimated_saved_loss",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    st.subheader("Narrative Explanations")
    advisor_col1, advisor_col2 = st.columns(2)
    with advisor_col1:
        st.markdown("**User-facing explanation**")
        st.write(baseline_advice["user_message"])
    with advisor_col2:
        st.markdown("**Lender-facing intervention note**")
        st.write(baseline_advice["lender_message"])

    st.subheader("Why This Record Looks Risky")
    reason_frame = pd.DataFrame(
        {
            "signal": [
                "Recent utilization",
                "Utilization deterioration",
                "Cash-advance dependence",
                "Installment lateness",
                "Payment capacity",
                "Bureau overdue burden",
            ],
            "current_value": [
                f"{baseline_row['cc_util_recent']:.0%}",
                f"{baseline_row['cc_util_deterioration']:.0%}",
                f"{baseline_row['cc_drawings_atm_ratio']:.0%}",
                f"{baseline_row['inst_late_recent']:.0%}",
                f"{baseline_row['monthly_payment_capacity']:.0%}",
                f"{baseline_row['bureau_overdue_to_debt_ratio']:.0%}",
            ],
            "why_it_matters": [
                "High recent card usage can signal tightening liquidity and rising revolving stress.",
                "A worsening trend is often more informative than a high static level alone.",
                "ATM-heavy draw behavior is a classic liquidity warning sign.",
                "Recent lateness is one of the strongest predictors of near-term slippage.",
                "Low spare payment capacity means less room to absorb shocks.",
                "External bureau stress suggests the issue may extend beyond this one account.",
            ],
        }
    )
    st.dataframe(reason_frame, use_container_width=True, hide_index=True)
