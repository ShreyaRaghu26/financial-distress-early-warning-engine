# Financial Distress Early Warning Engine

A fintech decision-support application that detects early signs of borrower distress, explains why a customer is risky, recommends the next best intervention, and estimates business impact before delinquency becomes severe.

## What this project does

- Uses real `Home Credit Default Risk` data through a Hugging Face parquet mirror
- Detects early warning signals such as utilization stress, worsening installment behavior, cash-advance dependence, and external credit pressure
- Scores customers into `stable`, `vulnerable`, `slipping`, and `critical`
- Combines behavioral scoring with an `XGBoost` model for default-risk estimation
- Simulates what-if scenarios such as income drops, missed payments, restructures, and paydown support
- Recommends intervention actions and estimates potential value protected
- Provides business-friendly, customer-friendly, and analyst-friendly views in one Streamlit app

## Why this is different

This is not just a credit risk model.

It is an early warning and intervention engine that answers:

- Which customers are showing stress before default?
- Why is this customer risky right now?
- What action should the company take next?
- How much risk or expected loss could that action reduce?

## App views

The Streamlit app includes three audience-specific views:

- `Executive View`: plain-language summary, top action, and intervention impact
- `Customer Guidance`: customer-friendly explanation and next steps
- `Analyst View`: model evidence, peer benchmarking, scenario analysis, and portfolio worklist

## Core capabilities

### 1. Real-data risk detection

The app loads engineered borrower-level features from the `Home Credit Default Risk` dataset and uses them to identify signs of financial stress.

### 2. ML + rules decision layer

It combines:

- a behavioral early warning score
- an `XGBoost` default model
- a blended priority score for intervention decisions

### 3. Action center

For each customer, the app ranks actions such as:

- `Restructure offer`
- `Autopay stabilization`
- `Balance paydown support`
- `Cash advance replacement`
- `Temporary hardship plan`

Each action includes:

- expected risk reduction
- expected stage after action
- estimated saved loss

### 4. Portfolio intervention queue

The app also generates a stage-balanced worklist so teams can prioritize customers across:

- `critical`
- `slipping`
- `vulnerable`
- `stable`

instead of only reacting to the worst cases.

## Tech stack

- `Python`
- `pandas`
- `numpy`
- `Streamlit`
- `Plotly`
- `scikit-learn`
- `XGBoost`
- `OpenAI SDK`
- `Hugging Face parquet dataset access`

## Project structure

- `streamlit_app.py`: main application
- `validate_app.py`: regression-style validation sweep across scenarios and queue logic
- `src/home_credit_loader.py`: loads and prepares Home Credit data
- `src/home_credit_engine.py`: behavioral scoring, evidence building, and recommendations
- `src/home_credit_model.py`: XGBoost training, scoring, and model explanations
- `src/home_credit_scenarios.py`: borrower scenario simulation
- `src/home_credit_actions.py`: intervention ranking and portfolio queue generation
- `src/llm_advisor.py`: narrative explanation layer with fallback messaging

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python -m streamlit run streamlit_app.py
```

Open:

```bash
http://localhost:8501
```

Run validation:

```bash
python validate_app.py
```

## Example use cases

- Lender operations teams prioritizing intervention queues
- Risk teams monitoring early-stage borrower deterioration
- Product teams testing how support actions may reduce expected loss
- Recruiter/demo audiences looking for an end-to-end fintech decision system
