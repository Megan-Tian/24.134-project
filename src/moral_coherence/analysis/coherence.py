"""Coherence metrics (HANDOFF §19)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from moral_coherence.io_utils import read_jsonl, write_json, write_jsonl


def continuous_coherence(j_direction: float, output_direction: float) -> float:
    return 1.0 - abs(j_direction - output_direction) / 2.0


def sign_agreement(
    j_direction: float,
    output_direction: float,
    *,
    threshold: float = 0.1,
) -> float | None:
    if abs(j_direction) < threshold or abs(output_direction) < threshold:
        return None
    return 1.0 if np.sign(j_direction) == np.sign(output_direction) else 0.0


def analyze_run(
    run_dir: Path,
    *,
    ambiguity_threshold: float = 0.1,
    output_mode: str = "eventual",
) -> dict[str, Any]:
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    jspace = read_jsonl(run_dir / "jspace.jsonl")
    scores_path = run_dir / "output_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"missing {scores_path}; run score_outputs first")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))

    final_dir = float(scores["output_direction_final"])
    choice = scores["choice"]
    decision_step = choice.get("decision_token_index")

    prefix_map = {
        p["generation_step"]: p["output_direction_prefix"]
        for p in scores.get("prefix_directions", [])
    }

    rows: list[dict[str, Any]] = []
    for row in jspace:
        step = row["generation_step"]
        if output_mode == "prefix":
            out_dir = float(prefix_map.get(step, 0.0))
        else:
            out_dir = final_dir
        coh = continuous_coherence(row["j_direction"], out_dir)
        sag = sign_agreement(
            row["j_direction"], out_dir, threshold=ambiguity_threshold
        )
        rows.append(
            {
                "run_id": meta["run_id"],
                "scenario_id": meta["scenario_id"],
                "condition": meta["condition"],
                "generation_step": step,
                "layer": row["layer"],
                "j_direction": row["j_direction"],
                "output_direction": out_dir,
                "coherence": coh,
                "sign_agreement": sag,
                "conflict": row["conflict"],
                "pre_choice": (
                    decision_step is None or step < decision_step
                ),
            }
        )

    coherences = [r["coherence"] for r in rows]
    mae = float(np.mean([abs(r["j_direction"] - r["output_direction"]) for r in rows]))
    signs = [r["sign_agreement"] for r in rows if r["sign_agreement"] is not None]
    pre = [r for r in rows if r["pre_choice"]]
    pre_coh = float(np.mean([r["coherence"] for r in pre])) if pre else float("nan")

    # best predictive layer: max mean |j_direction| pre-choice aligned with final sign
    layer_stats: dict[int, list[float]] = {}
    for r in pre:
        layer_stats.setdefault(r["layer"], []).append(r["coherence"])
    best_layer = None
    best_score = -1.0
    for layer, vals in layer_stats.items():
        m = float(np.mean(vals))
        if m > best_score:
            best_score = m
            best_layer = layer

    summary = {
        "run_id": meta["run_id"],
        "scenario_id": meta["scenario_id"],
        "condition": meta["condition"],
        "final_choice": choice.get("choice"),
        "choice_confidence": choice.get("choice_confidence"),
        "ambiguous": choice.get("ambiguous"),
        "decision_token_index": decision_step,
        "decision_span": choice.get("decision_span"),
        "output_direction_final": final_dir,
        "mean_coherence": float(np.mean(coherences)) if coherences else float("nan"),
        "pre_choice_coherence": pre_coh,
        "mae": mae,
        "sign_agreement_rate": float(np.mean(signs)) if signs else float("nan"),
        "max_conflict": float(max((r["conflict"] for r in rows), default=0.0)),
        "best_predictive_layer": best_layer,
        "best_predictive_layer_coherence": best_score if best_layer is not None else None,
        "n_steps": meta.get("n_tokens"),
        "n_layers": len({r["layer"] for r in rows}),
        "output_mode": output_mode,
        "generated_text": meta.get("generated_text"),
    }
    return {"summary": summary, "rows": rows}


def run_analysis(
    *,
    raw_dir: Path,
    processed_dir: Path,
    ambiguity_threshold: float = 0.1,
    output_mode: str = "eventual",
) -> list[dict[str, Any]]:
    gens = raw_dir / "generations"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "coherence").mkdir(parents=True, exist_ok=True)
    summaries = []
    for run_dir in sorted(gens.iterdir()):
        if not (run_dir / "metadata.json").exists():
            continue
        if not (run_dir / "output_scores.json").exists():
            print(f"skip {run_dir.name}: no output_scores.json")
            continue
        result = analyze_run(
            run_dir,
            ambiguity_threshold=ambiguity_threshold,
            output_mode=output_mode,
        )
        rid = result["summary"]["run_id"]
        write_jsonl(processed_dir / "coherence" / f"{rid}.jsonl", result["rows"])
        write_json(processed_dir / "coherence" / f"{rid}_summary.json", result["summary"])
        summaries.append(result["summary"])
        s = result["summary"]
        print(
            f"[{rid}] choice={s['final_choice']} mean_coh={s['mean_coherence']:.3f} "
            f"pre={s['pre_choice_coherence']:.3f} bestL={s['best_predictive_layer']} "
            f"max_conflict={s['max_conflict']:.3f}"
        )
    write_json(processed_dir / "coherence" / "summaries.json", summaries)
    return summaries
