# exp-ethics

Experiential Ethics project: J-space / moral coherence experiments.

## Layout

- `experiments/` — stimulus text from the sacred-values decision paper
- `HANDOFF.md` — experiment design / implementation handoff
- `factory-experiment/` — early CEO factory probe
- `jacobian-lens/` — **git submodule** (Anthropic Jacobian lens + local walkthrough)

## Submodule

`jacobian-lens` is pinned to a specific commit (see `.gitmodules` / `git submodule status`).

```bash
git submodule update --init --recursive
```

The pinned commit includes a local `walkthrough.py` on branch `local/walkthrough`. That commit is **not** on `github.com/anthropics/jacobian-lens`; to clone this project elsewhere, push the submodule branch to your own fork and point `.gitmodules` at that fork.

## PDFs

Course PDFs stay local and are gitignored (`*.pdf`).
