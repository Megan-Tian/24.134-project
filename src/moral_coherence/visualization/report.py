"""Qualitative experiment report: J-space top-k vs generated text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from moral_coherence.generation.prompts import wrap_scenario
from moral_coherence.io_utils import (
    get_scenario,
    load_scenarios,
    load_stimulus,
    load_yaml,
    read_jsonl,
    repo_root,
)
from moral_coherence.jspace.adapter import JSpaceAdapter


def _top_k_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> list[str]:
    idx = logits.topk(k).indices.tolist()
    return [tokenizer.decode([i]).replace("\n", "\\n") for i in idx]


def _select_layers(available: Sequence[int], n: int = 4) -> list[int]:
    layers = list(available)
    if len(layers) <= n:
        return layers
    picks = [
        layers[0],
        layers[len(layers) // 4],
        layers[len(layers) // 2],
        layers[(3 * len(layers)) // 4],
        layers[-1],
    ]
    out: list[int] = []
    for layer in picks:
        if layer not in out:
            out.append(layer)
    return out[:n] if len(out) > n else out


def _select_steps(
    n_steps: int,
    decision_step: int | None,
    *,
    max_points: int = 4,
) -> list[int]:
    if n_steps <= 0:
        return []
    candidates = [
        0,
        max(0, n_steps // 4),
        max(0, n_steps // 2),
        max(0, (3 * n_steps) // 4),
        n_steps - 1,
    ]
    if decision_step is not None and 0 <= decision_step < n_steps:
        candidates.append(int(decision_step))
        if decision_step > 0:
            candidates.append(int(decision_step) - 1)
    steps = sorted({s for s in candidates if 0 <= s < n_steps})
    if len(steps) <= max_points:
        return steps
    preferred: list[int] = [steps[0]]
    mid = steps[len(steps) // 2]
    if decision_step is not None and 0 <= decision_step < n_steps:
        preferred.append(int(decision_step))
    preferred.append(mid)
    preferred.append(steps[-1])
    q = max(0, n_steps // 4)
    if q not in preferred:
        preferred.insert(1, q)
    out: list[int] = []
    for s in preferred:
        if s in steps and s not in out:
            out.append(s)
        if len(out) >= max_points:
            break
    return sorted(out)


def _direction_label(d: float, option_a: str, option_b: str) -> str:
    if d < -0.15:
        return f"toward A ({option_a[:40]})"
    if d > 0.15:
        return f"toward B ({option_b[:40]})"
    return "near-neutral"


@torch.no_grad()
def _readouts_for_run(
    adapter: JSpaceAdapter,
    *,
    prompt_body: str,
    tokens: list[dict[str, Any]],
    layers: Sequence[int],
    steps: Sequence[int],
    jspace_by_key: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute lens top-k at selected (step, layer) on the prefix ending at that step."""
    chat = adapter.format_chat_prompt(prompt_body)
    prompt_ids = adapter.encode(chat, max_length=2048)
    rows: list[dict[str, Any]] = []
    for step in steps:
        gen_ids = [t["token_id"] for t in tokens if t["generation_step"] <= step]
        if not gen_ids:
            continue
        extra = torch.tensor([gen_ids], device=prompt_ids.device)
        input_ids = torch.cat([prompt_ids, extra], dim=1)
        pos = input_ids.shape[1] - 1
        lens = adapter.readout_at_position(input_ids, layers=layers, position=pos)
        tok_row = next(t for t in tokens if t["generation_step"] == step)
        gen_tok = tok_row.get("token_text") or tok_row.get("token_str") or ""
        for layer in layers:
            js = jspace_by_key.get((step, layer), {})
            j_dir = js.get("j_direction", js.get("j_dir"))
            rows.append(
                {
                    "generation_step": step,
                    "layer": layer,
                    "generated_token": gen_tok,
                    "top_k": _top_k_tokens(lens[layer], adapter.tokenizer, k=5),
                    "j_direction": j_dir,
                    "score_a": js.get("score_a"),
                    "score_b": js.get("score_b"),
                }
            )
    return rows


def _run_section(
    *,
    meta: dict[str, Any],
    scores: dict[str, Any],
    summary: dict[str, Any] | None,
    readouts: list[dict[str, Any]],
    heatmap_rel: str | None,
) -> str:
    choice = scores.get("choice", {})
    lines: list[str] = []
    lines.append(f"## {meta['scenario_id']} (`{meta.get('condition', '')}`)")
    lines.append("")
    lines.append(f"- **run_id:** `{meta['run_id']}`")
    lines.append(
        f"- **options:** A = {meta.get('option_a')} · B = {meta.get('option_b')}"
    )
    lines.append(
        f"- **parsed choice:** `{choice.get('choice')}` "
        f"(conf={choice.get('choice_confidence')}, "
        f"ambiguous={choice.get('ambiguous')}, "
        f"decision_step={choice.get('decision_token_index')})"
    )
    if summary:
        lines.append(
            f"- **coherence:** mean={summary.get('mean_coherence'):.3f}, "
            f"pre-choice={summary.get('pre_choice_coherence'):.3f}, "
            f"best_layer={summary.get('best_predictive_layer')}, "
            f"max_conflict={summary.get('max_conflict'):.3f}"
        )
        out_dir = float(summary.get("output_direction_final", 0.0))
        lines.append(
            f"- **output direction:** {out_dir:+.3f} "
            f"({_direction_label(out_dir, meta.get('option_a', 'A'), meta.get('option_b', 'B'))})"
        )
    if heatmap_rel:
        lines.append(f"- **heatmap:** `{heatmap_rel}`")
    lines.append("")
    lines.append("### Generated text")
    lines.append("")
    lines.append("```text")
    lines.append((meta.get("generated_text") or "").strip())
    lines.append("```")
    lines.append("")
    lines.append("### J-space readouts vs generated token")
    lines.append("")
    lines.append(
        "At each selected generation step, the **generated token** is what the "
        "model actually emitted; **J-space top-5** is the Jacobian-lens verbalization "
        "of the residual at that layer/position (what the activation is disposed to say)."
    )
    lines.append("")

    by_step: dict[int, list[dict[str, Any]]] = {}
    for r in readouts:
        by_step.setdefault(r["generation_step"], []).append(r)

    for step in sorted(by_step):
        gen_tok = by_step[step][0]["generated_token"]
        lines.append(f"#### Step {step} — generated `{gen_tok!r}`")
        lines.append("")
        lines.append("| layer | j_direction | top-5 lens tokens |")
        lines.append("|------:|------------:|-------------------|")
        for r in sorted(by_step[step], key=lambda x: x["layer"]):
            jd = r.get("j_direction")
            jd_s = f"{jd:+.3f}" if jd is not None else "—"
            tops = ", ".join(f"`{t}`" for t in r["top_k"])
            lines.append(f"| {r['layer']} | {jd_s} | {tops} |")
        lines.append("")

    lines.append("### Qualitative note")
    lines.append("")
    if choice.get("choice") == "A":
        lines.append(
            "Model text committed to **Option A**. Compare whether mid/late-layer "
            "j_direction is negative (A) before the decision step, and whether lens "
            "top-5 tokens mention A-related concepts versus B-related ones."
        )
    elif choice.get("choice") == "B":
        lines.append(
            "Model text committed to **Option B**. Compare whether mid/late-layer "
            "j_direction is positive (B) before the decision step, and whether lens "
            "top-5 tokens track B-related concepts."
        )
    else:
        lines.append(
            "Choice parser did not find a clear Option 1/2 commitment — treat "
            "output_direction as soft/ambiguous and lean on the raw text + lens tokens."
        )
    lines.append("")
    return "\n".join(lines)


def write_qualitative_report(
    *,
    raw_dir: Path,
    processed_dir: Path,
    figures_dir: Path,
    report_path: Path,
    model_name: str,
    models_yaml: Path,
    scenarios_path: Path,
    concepts_path: Path,
    select_layers: Sequence[int] | None = None,
) -> Path:
    """Write markdown report with J-space top-k vs output for latest run per scenario."""
    del concepts_path  # reserved for future concept-definition calibration notes
    root = repo_root()
    models_cfg = load_yaml(models_yaml)["models"]
    mcfg = models_cfg[model_name]
    scenarios = load_scenarios(scenarios_path)

    gens = raw_dir / "generations"
    run_dirs = sorted(
        [p for p in gens.iterdir() if (p / "metadata.json").exists()],
        key=lambda p: json.loads((p / "metadata.json").read_text())["scenario_id"],
    )
    if not run_dirs:
        raise FileNotFoundError(f"no runs in {gens}")

    latest_by_scenario: dict[str, Path] = {}
    for run_dir in run_dirs:
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        sid = meta["scenario_id"]
        prev = latest_by_scenario.get(sid)
        if prev is None or run_dir.stat().st_mtime >= prev.stat().st_mtime:
            latest_by_scenario[sid] = run_dir

    print("Loading model for qualitative J-space readouts…")
    adapter = JSpaceAdapter(
        model_name,
        lens_repo=mcfg["lens_repo"],
        lens_revision=mcfg["lens_revision"],
        lens_file=mcfg["lens_file"],
        dtype=mcfg.get("dtype", "bfloat16"),
    )
    layers = (
        list(select_layers)
        if select_layers
        else _select_layers(adapter.available_layers())
    )

    header = [
        "# Moral coherence experiment — qualitative report",
        "",
        f"- **model:** `{model_name}`",
        f"- **lens layers sampled:** {layers}",
        f"- **n scenarios (latest run each):** {len(latest_by_scenario)}",
        "",
        "J-space = Jacobian lens verbalization of residual-stream states. "
        "It is **not** claimed as privileged access to beliefs; the question is "
        "whether these readouts cohere with the model's eventual text choice.",
        "",
        "---",
        "",
    ]
    sections: list[str] = []

    for sid in sorted(latest_by_scenario):
        run_dir = latest_by_scenario[sid]
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        scores_path = run_dir / "output_scores.json"
        if not scores_path.exists():
            print(f"skip {run_dir.name}: no output_scores.json")
            continue
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        summary_path = processed_dir / "coherence" / f"{meta['run_id']}_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else None
        )
        tokens = read_jsonl(run_dir / "tokens.jsonl")
        jspace = read_jsonl(run_dir / "jspace.jsonl")
        jspace_by_key = {(r["generation_step"], r["layer"]): r for r in jspace}

        scenario = get_scenario(scenarios, meta["scenario_id"])
        stimulus = load_stimulus(root, scenario)
        prompt_body = wrap_scenario(stimulus)

        decision = scores.get("choice", {}).get("decision_token_index")
        steps = _select_steps(len(tokens), decision)
        print(f"  readouts {meta['scenario_id']} steps={steps} layers={layers}")
        readouts = _readouts_for_run(
            adapter,
            prompt_body=prompt_body,
            tokens=tokens,
            layers=layers,
            steps=steps,
            jspace_by_key=jspace_by_key,
        )

        heat = figures_dir / "layer_time" / f"{meta['run_id']}_coherence.png"
        if not heat.exists():
            alt = figures_dir / f"{meta['run_id']}_coherence.png"
            heat = alt if alt.exists() else heat
        heat_rel = str(heat.relative_to(root)) if heat.exists() else None

        sections.append(
            _run_section(
                meta=meta,
                scores=scores,
                summary=summary,
                readouts=readouts,
                heatmap_rel=heat_rel,
            )
        )
        sections.append("---\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(header + sections), encoding="utf-8")
    print(f"wrote {report_path}")
    return report_path
