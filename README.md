# Moral Coherence Experiment Package
## Setup

```bash
# Submodule (already pinned)
git submodule update --init --recursive

# Install experiment package + editable jlens
pip install -e ./jacobian-lens
pip install -e .
```

## First milestone run

```bash
python scripts/run_generation.py --config configs/experiment.yaml
python scripts/score_outputs.py --input results/raw/
python scripts/run_analysis.py --config configs/analysis.yaml
python scripts/make_figures.py --input results/processed/
```

Default config: `Qwen/Qwen3.5-0.8B` × `exp1_taboo_safety_at_work` × one greedy seed.

## Layout

- `experiments/` — raw paper stimulus wording
- `stimuli/` — scenario metadata + concept vocabularies
- `src/moral_coherence/` — pipeline code
- `scripts/` — CLI entry points
- `results/` — gitignored outputs
