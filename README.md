# Financial Distress Early Warning Engine

An LLM-ready fintech project that detects early signs of customer financial distress before delinquency happens.

## What it does

- Generates realistic synthetic customer behavior data
- Detects early warning signals such as rising utilization and weakening repayments
- Classifies customers into `stable`, `vulnerable`, `slipping`, or `critical`
- Simulates interventions and stress scenarios
- Produces user-friendly action guidance with an LLM layer and a local fallback

## Tech stack

- `Python`
- `pandas` and `numpy`
- `Streamlit` for the app
- `Plotly` for interactive visuals
- `OpenAI SDK` for narrative explanations and recommendations

## Project structure

- `src/data_generator.py`: synthetic customer and monthly behavior generation
- `src/risk_engine.py`: distress scoring, classification, and action policies
- `src/scenario_engine.py`: what-if scenario simulation
- `src/llm_advisor.py`: LLM-backed explanations with a deterministic fallback
- `src/demo.py`: terminal demo for quick validation

## Run locally

```bash
python -m src.demo
```

Dashboard UI will be added next.
