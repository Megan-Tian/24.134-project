#!/usr/bin/env python3
"""Compute J-space / output coherence trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from moral_coherence.analysis.coherence import run_analysis
from moral_coherence.io_utils import load_yaml, repo_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    args = p.parse_args()
    root = repo_root()
    cfg = load_yaml(args.config if args.config.is_absolute() else root / args.config)
    summaries = run_analysis(
        raw_dir=root / cfg.get("raw_dir", "results/raw"),
        processed_dir=root / cfg.get("processed_dir", "results/processed"),
        ambiguity_threshold=float(cfg.get("ambiguity_threshold", 0.1)),
        output_mode=str(cfg.get("output_mode", "eventual")),
    )
    print(f"analyzed {len(summaries)} run(s)")


if __name__ == "__main__":
    main()
