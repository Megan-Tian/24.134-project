#!/usr/bin/env python3
"""Export precomputed experiment results into compact JSON for the static website.

The website is a static replay of already-computed runs (GitHub Pages cannot run
the model). This script flattens results/raw + results/processed into one JSON per
scenario under web/data/, plus an index.json.

By default it also loads the model once to recompute the Jacobian-lens top-k
verbalization at every generation step (for a subset of layers). Pass --no-model
to skip that and emit direction-only readouts (fast, no GPU / no download).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from moral_coherence.generation.prompts import wrap_scenario
from moral_coherence.io_utils import (
    get_scenario,
    load_scenarios,
    load_stimulus,
    load_yaml,
    read_jsonl,
    repo_root,
)


def _latest_run_per_scenario(gens: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for run_dir in gens.iterdir():
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sid = meta["scenario_id"]
        prev = latest.get(sid)
        if prev is None or run_dir.stat().st_mtime >= prev.stat().st_mtime:
            latest[sid] = run_dir
    return latest


def _pick_topk_layers(layers: Sequence[int], n: int = 6) -> list[int]:
    layers = sorted(layers)
    if len(layers) <= n:
        return list(layers)
    # evenly spaced across depth, always include first and last
    idxs = [round(i * (len(layers) - 1) / (n - 1)) for i in range(n)]
    picks = sorted({layers[i] for i in idxs})
    return picks


def _all_position_readouts(
    adapter: Any,
    input_ids,
    *,
    layers: Sequence[int],
    positions: Sequence[int],
) -> dict[int, dict[int, Any]]:
    """One forward pass; return {position: {layer: lens_logits}}.

    Greedy decoding is causal, so a single forward over prompt+generation
    reproduces the per-position residuals used during incremental generation.
    """
    import torch
    from jlens.hooks import ActivationRecorder

    layers = list(layers)
    final_layer = adapter.model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(adapter.hf_model.device)

    with torch.no_grad():
        with ActivationRecorder(adapter.model.layers, at=record_at) as recorder:
            adapter.model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

    out: dict[int, dict[int, Any]] = {}
    for pos in positions:
        per_layer: dict[int, Any] = {}
        for layer in layers:
            residual = activations[layer][0, pos].float()
            transported = adapter.lens.transport(residual, layer)
            per_layer[layer] = adapter.model.unembed(transported).float().cpu()
        out[pos] = per_layer
    return out


def _top_k_tokens(logits, tokenizer, k: int = 6) -> list[str]:
    idx = logits.topk(k).indices.tolist()
    return [tokenizer.decode([i]) for i in idx]


def _build_scenario_json(
    *,
    run_dir: Path,
    scenario: dict[str, Any],
    processed_dir: Path,
    root: Path,
    adapter: Any | None,
    topk_layers: list[int] | None,
    topk_k: int,
) -> dict[str, Any]:
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    prompt = json.loads((run_dir / "prompt.txt.json").read_text(encoding="utf-8"))[
        "prompt"
    ]
    tokens = read_jsonl(run_dir / "tokens.jsonl")
    jspace = read_jsonl(run_dir / "jspace.jsonl")
    scores_path = run_dir / "output_scores.json"
    scores = (
        json.loads(scores_path.read_text(encoding="utf-8"))
        if scores_path.exists()
        else {}
    )
    run_id = meta["run_id"]
    summary_path = processed_dir / "coherence" / f"{run_id}_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )

    all_layers = sorted({r["layer"] for r in jspace})
    n_steps = len(tokens)

    # index jspace by (step, layer)
    js_by_key: dict[tuple[int, int], dict[str, Any]] = {
        (r["generation_step"], r["layer"]): r for r in jspace
    }

    # output_direction prefix per step
    prefix = {
        d["generation_step"]: d["output_direction_prefix"]
        for d in scores.get("prefix_directions", [])
    }

    # optional lens top-k for a subset of layers, recomputed with the model
    topk_by_step: dict[int, dict[int, list[str]]] = {}
    if adapter is not None and topk_layers:
        stimulus = load_stimulus(root, scenario)
        chat = adapter.format_chat_prompt(wrap_scenario(stimulus))
        prompt_ids = adapter.encode(chat, max_length=2048)
        prompt_len = prompt_ids.shape[1]
        import torch

        gen_ids = [t["token_id"] for t in tokens]
        input_ids = torch.cat(
            [prompt_ids, torch.tensor([gen_ids], device=prompt_ids.device)], dim=1
        )
        # readout for step s is taken at position prompt_len - 1 + s
        positions = [prompt_len - 1 + s for s in range(n_steps)]
        pos_readouts = _all_position_readouts(
            adapter, input_ids, layers=topk_layers, positions=positions
        )
        for s in range(n_steps):
            pos = prompt_len - 1 + s
            per_layer = pos_readouts.get(pos, {})
            topk_by_step[s] = {
                layer: _top_k_tokens(per_layer[layer], adapter.tokenizer, k=topk_k)
                for layer in topk_layers
                if layer in per_layer
            }

    steps: list[dict[str, Any]] = []
    for t in tokens:
        s = t["generation_step"]
        j_dir = {}
        for layer in all_layers:
            row = js_by_key.get((s, layer))
            if row is not None:
                j_dir[str(layer)] = round(float(row["j_direction"]), 4)
        step_obj: dict[str, Any] = {
            "step": s,
            "token": t.get("token_text", ""),
            "j_direction": j_dir,
            "output_direction_prefix": prefix.get(s, 0.0),
        }
        if s in topk_by_step:
            step_obj["topk"] = {
                str(layer): toks for layer, toks in topk_by_step[s].items()
            }
        steps.append(step_obj)

    choice = scores.get("choice", {})
    final = {
        "choice": choice.get("choice") or summary.get("final_choice"),
        "choice_confidence": choice.get("choice_confidence"),
        "ambiguous": choice.get("ambiguous"),
        "decision_step": choice.get("decision_token_index")
        if choice.get("decision_token_index") is not None
        else summary.get("decision_token_index"),
        "output_direction_final": scores.get("output_direction_final")
        or summary.get("output_direction_final"),
        "mean_coherence": summary.get("mean_coherence"),
        "pre_choice_coherence": summary.get("pre_choice_coherence"),
        "best_layer": summary.get("best_predictive_layer"),
        "max_conflict": summary.get("max_conflict"),
        "sign_agreement_rate": summary.get("sign_agreement_rate"),
    }

    return {
        "scenario_id": meta["scenario_id"],
        "run_id": run_id,
        "condition": meta.get("condition"),
        "experiment": scenario.get("experiment"),
        "domain": scenario.get("domain"),
        "source": scenario.get("source"),
        "model": meta.get("model_name"),
        "option_a": meta.get("option_a"),
        "option_b": meta.get("option_b"),
        "value_a": meta.get("value_a"),
        "value_b": meta.get("value_b"),
        "sacredness_a": scenario.get("sacredness_a"),
        "sacredness_b": scenario.get("sacredness_b"),
        "concept_set_a": meta.get("concept_set_a"),
        "concept_set_b": meta.get("concept_set_b"),
        "prompt": prompt,
        "generated_text": meta.get("generated_text", ""),
        "n_steps": n_steps,
        "layers": all_layers,
        "topk_layers": topk_layers or [],
        "final": final,
        "steps": steps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    p.add_argument("--analysis", type=Path, default=Path("configs/analysis.yaml"))
    p.add_argument("--out", type=Path, default=Path("web/data"))
    p.add_argument(
        "--no-model",
        action="store_true",
        help="skip lens top-k recompute (direction-only readouts, no GPU)",
    )
    p.add_argument("--topk-layers", type=int, default=6)
    p.add_argument("--topk-k", type=int, default=6)
    args = p.parse_args()

    root = repo_root()
    exp_cfg = load_yaml(root / args.config)
    ana_cfg = load_yaml(root / args.analysis)
    raw_dir = root / ana_cfg.get("raw_dir", "results/raw")
    processed_dir = root / ana_cfg.get("processed_dir", "results/processed")
    scenarios = load_scenarios(root / exp_cfg["stimuli_path"])

    gens = raw_dir / "generations"
    latest = _latest_run_per_scenario(gens)
    if not latest:
        raise FileNotFoundError(f"no runs found in {gens}")

    adapter = None
    topk_layers: list[int] | None = None
    if not args.no_model:
        from moral_coherence.jspace.adapter import JSpaceAdapter

        model_name = exp_cfg["model"]
        mcfg = load_yaml(root / "configs" / "models.yaml")["models"][model_name]
        print(f"Loading model {model_name} for lens top-k…")
        adapter = JSpaceAdapter(
            model_name,
            lens_repo=mcfg["lens_repo"],
            lens_revision=mcfg["lens_revision"],
            lens_file=mcfg["lens_file"],
            dtype=mcfg.get("dtype", "bfloat16"),
        )
        topk_layers = _pick_topk_layers(adapter.available_layers(), args.topk_layers)
        print(f"top-k layers: {topk_layers}")

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    for sid in sorted(latest):
        run_dir = latest[sid]
        scenario = get_scenario(scenarios, sid)
        print(f"exporting {sid} ({run_dir.name})…")
        data = _build_scenario_json(
            run_dir=run_dir,
            scenario=scenario,
            processed_dir=processed_dir,
            root=root,
            adapter=adapter,
            topk_layers=topk_layers,
            topk_k=args.topk_k,
        )
        (out_dir / f"{sid}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        index.append(
            {
                "scenario_id": sid,
                "condition": data["condition"],
                "experiment": data["experiment"],
                "domain": data["domain"],
                "option_a": data["option_a"],
                "option_b": data["option_b"],
                "value_a": data["value_a"],
                "value_b": data["value_b"],
                "final_choice": data["final"].get("choice"),
                "mean_coherence": data["final"].get("mean_coherence"),
                "n_steps": data["n_steps"],
                "has_topk": bool(data["topk_layers"]),
            }
        )

    index.sort(key=lambda d: (d["experiment"] or 0, d["condition"] or "", d["scenario_id"]))
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "model": exp_cfg["model"],
                "n_experiments": len(index),
                "experiments": index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(index)} scenario file(s) + index.json to {out_dir}")


if __name__ == "__main__":
    main()
