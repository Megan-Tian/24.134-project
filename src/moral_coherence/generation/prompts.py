"""Neutral prompt wrapper (HANDOFF §8) — identical across conditions."""

from __future__ import annotations

PROMPT_WRAPPER = """Read the following decision scenario carefully.

You must choose between the two options presented at the end.

Give a brief explanation (a few sentences), then end with exactly one of these lines:
Final choice: Option 1
or
Final choice: Option 2

{scenario}
"""


def wrap_scenario(scenario_text: str) -> str:
    return PROMPT_WRAPPER.format(scenario=scenario_text.strip()).strip()
