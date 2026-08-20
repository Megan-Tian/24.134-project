#!/usr/bin/env python3
"""Verify / re-export J-space rows from raw generations (already captured at gen time)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moral_coherence.io_utils import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("results/raw/"))
    args = p.parse_args()
    gens = args.input / "generations"
    n = 0
    for run_dir in sorted(gens.iterdir()) if gens.exists() else []:
        jpath = run_dir / "jspace.jsonl"
        if not jpath.exists():
            print(f"MISSING jspace: {run_dir}")
            continue
        rows = read_jsonl(jpath)
        steps = {r["generation_step"] for r in rows}
        layers = {r["layer"] for r in rows}
        tokens = read_jsonl(run_dir / "tokens.jsonl")
        ok = steps == {t["generation_step"] for t in tokens} or steps.issubset(
            {t["generation_step"] for t in tokens}
        )
        print(
            f"{run_dir.name}: steps={len(steps)} layers={len(layers)} "
            f"rows={len(rows)} aligned={ok}"
        )
        # mirror into processed for convenience
        out = args.input.parent / "processed" / "jspace" / f"{run_dir.name}.jsonl"
        write_jsonl(out, rows)
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        (out.parent / f"{run_dir.name}_meta.json").write_text(
            json.dumps({"run_id": meta["run_id"], "n_rows": len(rows)}, indent=2),
            encoding="utf-8",
        )
        n += 1
    print(f"verified {n} run(s)")


if __name__ == "__main__":
    main()
