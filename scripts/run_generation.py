#!/usr/bin/env python3
"""Run generation + J-space capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from moral_coherence.generation.runner import run_experiment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="experiment config YAML",
    )
    args = p.parse_args()
    written = run_experiment(args.config)
    print(f"wrote {len(written)} run(s)")


if __name__ == "__main__":
    main()
