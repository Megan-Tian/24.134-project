#!/usr/bin/env python3
"""Make layer×time coherence heatmaps, summary CSV, and qualitative .md report."""

from __future__ import annotations

import argparse
from pathlib import Path

from moral_coherence.io_utils import load_yaml, repo_root
from moral_coherence.visualization.figures import make_figures
from moral_coherence.visualization.report import write_qualitative_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=None, help="processed dir")
    p.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    p.add_argument(
        "--skip-report",
        action="store_true",
        help="skip qualitative markdown report (no model reload)",
    )
    args = p.parse_args()
    root = repo_root()
    cfg = load_yaml(args.config if args.config.is_absolute() else root / args.config)
    processed = args.input or (root / cfg.get("processed_dir", "results/processed"))
    heat = cfg.get("heatmap", {})
    paths = make_figures(
        processed_dir=Path(processed),
        figures_dir=root / cfg.get("figures_dir", "results/figures"),
        tables_dir=root / cfg.get("tables_dir", "results/tables"),
        cmap=heat.get("cmap", "viridis"),
        dpi=int(heat.get("dpi", 150)),
    )
    print(f"wrote {len(paths)} figure/table artifact(s)")

    qcfg = cfg.get("qualitative_report", {}) or {}
    if not args.skip_report and qcfg.get("enabled", True):
        exp_cfg = load_yaml(root / "configs" / "experiment.yaml")
        write_qualitative_report(
            raw_dir=root / cfg.get("raw_dir", "results/raw"),
            processed_dir=Path(processed),
            figures_dir=root / cfg.get("figures_dir", "results/figures"),
            report_path=root
            / cfg.get("reports_dir", "results/reports")
            / "experiment_summary.md",
            model_name=exp_cfg["model"],
            models_yaml=root / "configs" / "models.yaml",
            scenarios_path=root / exp_cfg["stimuli_path"],
            concepts_path=root / exp_cfg["concepts_path"],
            select_layers=qcfg.get("layers"),
        )


if __name__ == "__main__":
    main()
