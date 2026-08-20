#!/usr/bin/env python3
"""Make layer×time coherence heatmaps and summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from moral_coherence.io_utils import load_yaml, repo_root
from moral_coherence.visualization.figures import make_figures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=None, help="processed dir")
    p.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
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
    print(f"wrote {len(paths)} artifact(s)")


if __name__ == "__main__":
    main()
