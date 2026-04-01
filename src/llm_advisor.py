from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """
You are a fintech early warning assistant.
Given a customer's distress stage, early warning probability, main drivers, evidence bullets, peer summary, and recommended actions,
write a concise but insight-rich explanation.
Be practical, specific, and action-oriented.
Avoid promising approvals or making legal claims.
Return JSON with keys: summary, user_message, lender_message.
""".strip()


def _fallback_response(payload: dict[str, Any]) -> dict[str, str]:
    drivers = ", ".join(payload["drivers"][:3])
    actions = ", ".join(action.rstrip(".") for action in payload["recommended_actions"][:2])
    evidence = " ".join(payload.get("evidence", [])[:2])
    peer_summary = payload.get("peer_summary", "")
    return {
        "summary": f"{payload['customer_id']} is currently in the {payload['distress_stage']} stage with early warning probability {payload['early_warning_probability']:.0%}.",
        "user_message": (
            f"We detected early financial stress signals because {drivers}. {evidence} {peer_summary} "
            f"Best next steps: {actions}."
        ),
        "lender_message": (
            f"Intervention priority for {payload['customer_id']}: {payload['distress_stage']}. {peer_summary} "
            f"Primary actions: {actions}."
        ),
    }


def generate_advice(payload: dict[str, Any], model: str = "gpt-4.1-mini") -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_response(payload)

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.4,
    )

    raw_text = response.output_text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return _fallback_response(payload)

    required_keys = {"summary", "user_message", "lender_message"}
    if not required_keys.issubset(parsed):
        return _fallback_response(payload)
    return {
        "summary": str(parsed["summary"]),
        "user_message": str(parsed["user_message"]),
        "lender_message": str(parsed["lender_message"]),
    }
