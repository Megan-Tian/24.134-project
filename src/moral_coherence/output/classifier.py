"""Independent output-choice parsing (no J-space)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ChoiceResult:
    choice: str | None  # "A", "B", or None
    choice_confidence: float
    choice_span: list[int] | None  # char span in text
    ambiguous: bool
    decision_token_index: int | None
    decision_span: str | None
    method: str


_OPTION_PATTERNS = [
    (r"final\s+choice\s*:\s*option\s*1\b", "A"),
    (r"final\s+choice\s*:\s*option\s*2\b", "B"),
    (r"final\s+choice\s*:\s*1\b", "A"),
    (r"final\s+choice\s*:\s*2\b", "B"),
    (r"\bi\s+(?:would\s+)?(?:choose|select|pick)\s+option\s*1\b", "A"),
    (r"\bi\s+(?:would\s+)?(?:choose|select|pick)\s+option\s*2\b", "B"),
    (r"\boption\s*1\b", "A"),
    (r"\boption\s*2\b", "B"),
    (r"\boption\s*a\b", "A"),
    (r"\boption\s*b\b", "B"),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_choice(
    text: str,
    option_a: str,
    option_b: str,
    *,
    tokens: list[dict[str, Any]] | None = None,
) -> ChoiceResult:
    """Lexical extraction of final A/B choice with confidence."""
    norm = _normalize(text)
    hits: list[tuple[int, str, str]] = []  # (pos, choice, matched)

    for pat, label in _OPTION_PATTERNS:
        for m in re.finditer(pat, norm, flags=re.IGNORECASE):
            hits.append((m.start(), label, m.group(0)))

    a_key = _normalize(option_a)
    b_key = _normalize(option_b)
    # shorten to distinctive core phrases
    for phrase, label in ((a_key, "A"), (b_key, "B")):
        # use last clause after colon if present
        core = phrase.split(":")[-1].strip()
        if len(core) < 4:
            continue
        # look near the end for "choose X" / "I choose X"
        for m in re.finditer(
            rf"(?:choose|select|pick|support|favor|invest in)\s+{re.escape(core[:40])}",
            norm,
        ):
            hits.append((m.start(), label, m.group(0)))
        # bare option wording near the end of the response
        tail = norm[max(0, len(norm) - 400) :]
        idx = tail.rfind(core[:40])
        if idx >= 0 and len(core) >= 8:
            hits.append((max(0, len(norm) - 400) + idx, label, core[:40]))

    if not hits:
        return ChoiceResult(
            choice=None,
            choice_confidence=0.0,
            choice_span=None,
            ambiguous=True,
            decision_token_index=None,
            decision_span=None,
            method="lexical",
        )

    # Prefer the last explicit commitment
    hits.sort(key=lambda h: h[0])
    last_pos, last_choice, matched = hits[-1]
    # Ambiguous if both labels appear in the last 200 chars with conflicting last≠majority
    tail = norm[max(0, len(norm) - 250) :]
    a_in_tail = bool(re.search(r"\boption\s*1\b|\boption\s*a\b", tail))
    b_in_tail = bool(re.search(r"\boption\s*2\b|\boption\s*b\b", tail))
    ambiguous = a_in_tail and b_in_tail and last_choice is not None

    # Map char position roughly onto token index via cumulative decode
    decision_token_index = None
    if tokens:
        built = ""
        for t in tokens:
            built += t.get("token_text", "")
            if _normalize(built).find(matched) >= 0 or len(_normalize(built)) >= last_pos + len(
                matched
            ):
                decision_token_index = t["generation_step"]
                break
        if decision_token_index is None:
            decision_token_index = tokens[-1]["generation_step"]

    # char span in original text (best-effort, case-insensitive search)
    span = None
    m_orig = re.search(re.escape(matched), text, flags=re.IGNORECASE)
    if m_orig:
        span = [m_orig.start(), m_orig.end()]

    conf = 0.6 if ambiguous else 0.9
    if len(hits) == 1:
        conf = min(1.0, conf + 0.05)

    return ChoiceResult(
        choice=last_choice,
        choice_confidence=conf,
        choice_span=span,
        ambiguous=ambiguous,
        decision_token_index=decision_token_index,
        decision_span=matched,
        method="lexical",
    )


def choice_to_direction(choice: str | None) -> float:
    if choice == "A":
        return -1.0
    if choice == "B":
        return 1.0
    return 0.0


def continuous_output_direction(
    text: str,
    option_a: str,
    option_b: str,
    choice: ChoiceResult,
) -> float:
    """Independent continuous score in [-1, 1] without J-space.

    Uses final choice when confident; otherwise mention-count soft score.
    """
    if choice.choice and choice.choice_confidence >= 0.75 and not choice.ambiguous:
        return choice_to_direction(choice.choice)

    norm = _normalize(text)
    a = _normalize(option_a).split(":")[-1].strip()
    b = _normalize(option_b).split(":")[-1].strip()
    # count keyword hits
    ca = norm.count(a[:24]) if len(a) >= 4 else 0
    cb = norm.count(b[:24]) if len(b) >= 4 else 0
    # option markers
    ca += len(re.findall(r"\boption\s*1\b", norm))
    cb += len(re.findall(r"\boption\s*2\b", norm))
    total = ca + cb
    if total == 0:
        return choice_to_direction(choice.choice)
    return (cb - ca) / total


def result_dict(choice: ChoiceResult) -> dict[str, Any]:
    return asdict(choice)
