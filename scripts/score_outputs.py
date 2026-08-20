#!/usr/bin/env python3
"""Score generated text for independent output moral direction."""

from __future__ import annotations

import argparse
from pathlib import Path

from moral_coherence.output.direction import score_raw_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("results/raw/"))
    args = p.parse_args()
    results = score_raw_dir(args.input)
    print(f"scored {len(results)} run(s)")


if __name__ == "__main__":
    main()
