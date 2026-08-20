"""Lightweight tests that do not require GPU / model downloads."""

from moral_coherence.analysis.coherence import continuous_coherence
from moral_coherence.generation.prompts import wrap_scenario
from moral_coherence.output.classifier import parse_choice


def test_parse_choice_final_line():
    text = "Safety matters more.\n\nFinal choice: Option 1\n"
    r = parse_choice(text, "Invest in safety at work", "Increase profit")
    assert r.choice == "A"


def test_wrap_scenario_neutral():
    text = wrap_scenario("Option 1: A.\nOption 2: B.")
    assert "J-space" not in text
    assert "taboo" not in text.lower()
    assert "Final choice: Option 1" in text
    assert "Option 1" in text


def test_parse_choice_option2():
    text = "After considering both sides, I choose Option 2: Increase profit."
    r = parse_choice(text, "Invest in safety at work", "Increase profit")
    assert r.choice == "B"
    assert r.ambiguous is False


def test_continuous_coherence_bounds():
    assert continuous_coherence(1.0, 1.0) == 1.0
    assert continuous_coherence(-1.0, 1.0) == 0.0
    assert continuous_coherence(0.0, 0.0) == 1.0
