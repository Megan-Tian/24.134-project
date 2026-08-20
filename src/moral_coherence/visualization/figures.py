"""Trajectory / heatmap figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moral_coherence.io_utils import read_jsonl


def coherence_heatmap(
    rows: list[dict],
    *,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    dpi: int = 150,
) -> Path:
    if not rows:
        raise ValueError("no rows to plot")
    layers = sorted({r["layer"] for r in rows})
    steps = sorted({r["generation_step"] for r in rows})
    layer_index = {l: i for i, l in enumerate(layers)}
    step_index = {s: i for i, s in enumerate(steps)}
    grid = np.full((len(layers), len(steps)), np.nan)
    for r in rows:
        grid[layer_index[r["layer"]], step_index[r["generation_step"]]] = r["coherence"]

    fig, ax = plt.subplots(figsize=(max(6, len(steps) * 0.08), max(4, len(layers) * 0.15)))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("generation step")
    ax.set_ylabel("layer")
    ax.set_title(title)
    # tick sparsely
    if len(steps) > 20:
        xt = np.linspace(0, len(steps) - 1, 8, dtype=int)
        ax.set_xticks(xt)
        ax.set_xticklabels([steps[i] for i in xt])
    else:
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
    if len(layers) > 20:
        yt = np.linspace(0, len(layers) - 1, 10, dtype=int)
        ax.set_yticks(yt)
        ax.set_yticklabels([layers[i] for i in yt])
    else:
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
    fig.colorbar(im, ax=ax, label="coherence")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def make_figures(
    *,
    processed_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    cmap: str = "viridis",
    dpi: int = 150,
) -> list[Path]:
    coh_dir = processed_dir / "coherence"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "layer_time").mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summaries_path = coh_dir / "summaries.json"
    if not summaries_path.exists():
        print("no summaries.json — run analysis first")
        return []
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))

    # write CSV summary table
    import csv

    csv_path = tables_dir / "coherence.csv"
    fields = [
        "run_id",
        "scenario_id",
        "condition",
        "final_choice",
        "decision_token_index",
        "mean_coherence",
        "pre_choice_coherence",
        "mae",
        "sign_agreement_rate",
        "max_conflict",
        "best_predictive_layer",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    paths: list[Path] = [csv_path]
    for s in summaries:
        rid = s["run_id"]
        rows = read_jsonl(coh_dir / f"{rid}.jsonl")
        title = (
            f"{s['scenario_id']} ({s['condition']}) run={rid} "
            f"choice={s['final_choice']}"
        )
        out = figures_dir / "layer_time" / f"{rid}_coherence.png"
        coherence_heatmap(rows, title=title, out_path=out, cmap=cmap, dpi=dpi)
        paths.append(out)
        print(f"wrote {out}")

        # terminal-friendly summary line already printed in analysis; echo CSV path
    print(f"wrote {csv_path}")
    return paths
