# Moral Coherence — web demo

A static site that lets an external audience explore the J-space moral-coherence
experiments. A visitor picks a dilemma, advances the model's response token by
token, and watches the **J-space readout** (what each layer's residual stream is
leaning toward, Option A ◀ ▶ Option B) update in real time.

Because GitHub Pages can only serve static files, the site does **not** run the
model in the browser. Instead it replays *precomputed* runs from `results/`,
flattened into small JSON files under `web/data/`.

## Files

- `index.html`, `styles.css`, `app.js` — the (build-free) single-page app
- `data/index.json` — list of experiments shown on the landing page
- `data/<scenario_id>.json` — per-experiment replay data (tokens + per-layer
  J-space directions + optional lens top-k verbalizations + final coherence)

## Regenerate the data

From the repo root, after runs exist in `results/`:

```bash
export PYTHONPATH=src:${PYTHONPATH:-}

# Fast: direction-only readouts (no GPU, no model download)
python3 scripts/export_web_data.py --no-model

# Rich: also recompute the Jacobian-lens top-k tokens per step
# (loads Qwen3.5-0.8B + the pre-fitted lens once; needs the ML env)
python3 scripts/export_web_data.py
```

The rich mode adds a `topk` block per step for a handful of evenly spaced layers;
the site renders it automatically when present (`has_topk` in `index.json`).

## Run locally

```bash
cd web
python3 -m http.server 8137
# open http://127.0.0.1:8137
```

Serve over HTTP (not `file://`) so `fetch()` can load the JSON.

## Deploy to GitHub Pages

Paths are relative and routing is hash-based, so the site works from a project
subpath (`https://<user>.github.io/<repo>/`).

A workflow at `.github/workflows/deploy-pages.yml` publishes the `web/` folder
automatically on push to `main`. To enable it: repo **Settings → Pages → Build
and deployment → Source: GitHub Actions**.

Make sure `web/data/*.json` is committed (it is not covered by the root
`.gitignore`, which only ignores `results/`).
