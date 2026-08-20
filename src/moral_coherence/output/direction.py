"""Score raw generation runs for output direction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moral_coherence.io_utils import read_jsonl, write_json
from moral_coherence.output.classifier import (
    continuous_output_direction,
    parse_choice,
    result_dict,
)


def score_run_dir(run_dir: Path) -> dict[str, Any]:
    import json

    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    tokens = read_jsonl(run_dir / "tokens.jsonl")
    text = meta["generated_text"]
    choice = parse_choice(
        text,
        meta["option_a"],
        meta["option_b"],
        tokens=tokens,
    )
    direction = continuous_output_direction(
        text, meta["option_a"], meta["option_b"], choice
    )

    # Prefix directions for no-future-leakage analyses
    prefix_dirs: list[dict[str, Any]] = []
    built = ""
    for t in tokens:
        built += t.get("token_text", "")
        step_choice = parse_choice(
            built, meta["option_a"], meta["option_b"], tokens=tokens[: t["generation_step"] + 1]
        )
        prefix_dirs.append(
            {
                "generation_step": t["generation_step"],
                "output_direction_prefix": continuous_output_direction(
                    built, meta["option_a"], meta["option_b"], step_choice
                ),
            }
        )

    out = {
        "run_id": meta["run_id"],
        "scenario_id": meta["scenario_id"],
        "condition": meta["condition"],
        "output_direction_final": direction,
        "choice": result_dict(choice),
        "prefix_directions": prefix_dirs,
    }
    write_json(run_dir / "output_scores.json", out)
    return out


def score_raw_dir(raw_dir: Path | str) -> list[dict[str, Any]]:
    raw = Path(raw_dir)
    gens = raw / "generations"
    results = []
    if not gens.exists():
        return results
    for run_dir in sorted(gens.iterdir()):
        if (run_dir / "metadata.json").exists():
            results.append(score_run_dir(run_dir))
            print(f"scored {run_dir.name}: choice={results[-1]['choice']['choice']}")
    return results
