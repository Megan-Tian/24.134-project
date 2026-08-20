"""Incremental generation with per-step J-space capture."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

import torch

from moral_coherence.generation.prompts import wrap_scenario
from moral_coherence.io_utils import (
    get_scenario,
    git_commit,
    load_scenarios,
    load_stimulus,
    load_yaml,
    repo_root,
    sha256_text,
    write_json,
    write_jsonl,
)
from moral_coherence.jspace.adapter import JSpaceAdapter


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_with_jspace(
    adapter: JSpaceAdapter,
    prompt_text: str,
    *,
    concept_ids,
    layers: Sequence[int] | None,
    max_new_tokens: int,
    max_seq_len: int,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy/sampled decode with J-space directions at each generation step.

    Returns (generated_text, token_rows, jspace_rows).
    """
    layers = list(layers) if layers is not None else adapter.available_layers()
    chat_prompt = adapter.format_chat_prompt(prompt_text)
    input_ids = adapter.encode(chat_prompt, max_length=max_seq_len)
    prompt_len = input_ids.shape[1]
    generated_ids: list[int] = []
    token_rows: list[dict[str, Any]] = []
    jspace_rows: list[dict[str, Any]] = []

    eos_id = adapter.tokenizer.eos_token_id
    # Some chat models use <|im_end|> etc.
    stop_ids = {eos_id} if eos_id is not None else set()
    for tid in (
        adapter.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        adapter.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ):
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)

    for step in range(max_new_tokens):
        # Lens readout at current last prompt/gen token (pre-next-token state)
        pos = input_ids.shape[1] - 1
        lens_logits = adapter.readout_at_position(
            input_ids, layers=layers, position=pos
        )

        # Next-token from model (final residual / lm head via HF)
        out = adapter.hf_model(input_ids=input_ids)
        next_logits = out.logits[0, -1].float()

        if do_sample and temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                pick = torch.multinomial(sorted_probs, 1).item()
                next_id = int(sorted_idx[pick].item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())
        else:
            next_id = int(torch.argmax(next_logits).item())

        token_text = adapter.tokenizer.decode([next_id])
        generated_ids.append(next_id)
        token_rows.append(
            {
                "generation_step": step,
                "token_id": next_id,
                "token_text": token_text,
            }
        )

        for layer, logits in lens_logits.items():
            direction, score_a, score_b = adapter.direction_from_logits(
                logits, concept_ids
            )
            conflict = min(abs(score_a), abs(score_b))
            jspace_rows.append(
                {
                    "generation_step": step,
                    "layer": layer,
                    "j_direction": direction,
                    "score_a": score_a,
                    "score_b": score_b,
                    "conflict": conflict,
                    "token_id": next_id,
                    "token_text": token_text,
                }
            )

        next_tensor = torch.tensor([[next_id]], device=input_ids.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

        if next_id in stop_ids:
            break
        if input_ids.shape[1] >= max_seq_len:
            break
        if torch.cuda.is_available() and step % 32 == 31:
            torch.cuda.empty_cache()

    text = adapter.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, token_rows, jspace_rows


def run_experiment(config_path: Path | str) -> list[Path]:
    """Run configured scenarios/seeds; write immutable raw artifacts. Returns run dirs."""
    root = repo_root()
    cfg = load_yaml(config_path)
    models_cfg = load_yaml(root / "configs" / "models.yaml")["models"]
    model_name = cfg["model"]
    if model_name not in models_cfg:
        raise KeyError(f"model {model_name} not in configs/models.yaml")
    mcfg = models_cfg[model_name]

    scenarios = load_scenarios(root / cfg["stimuli_path"])
    concepts = load_yaml(root / cfg["concepts_path"])

    adapter = JSpaceAdapter(
        model_name,
        lens_repo=mcfg["lens_repo"],
        lens_revision=mcfg["lens_revision"],
        lens_file=mcfg["lens_file"],
        dtype=mcfg.get("dtype", "bfloat16"),
    )
    layers = cfg.get("layers")
    out_root = root / cfg.get("output_dir", "results/raw")
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for scenario_id in cfg["scenario_ids"]:
        scenario = get_scenario(scenarios, scenario_id)
        stimulus = load_stimulus(root, scenario)
        prompt_body = wrap_scenario(stimulus)
        phrases_a = concepts[scenario["concept_set_a"]]
        phrases_b = concepts[scenario["concept_set_b"]]
        concept_ids = adapter.concept_token_ids(phrases_a, phrases_b)

        for seed in cfg.get("seeds", [0]):
            _set_seed(int(seed))
            run_id = uuid.uuid4().hex[:12]
            text, token_rows, jspace_rows = generate_with_jspace(
                adapter,
                prompt_body,
                concept_ids=concept_ids,
                layers=layers,
                max_new_tokens=int(cfg.get("max_new_tokens", 256)),
                max_seq_len=int(cfg.get("max_seq_len", 2048)),
                do_sample=bool(cfg.get("do_sample", False)),
                temperature=float(cfg.get("temperature", 0.0)),
                top_p=float(cfg.get("top_p", 1.0)),
            )

            for row in token_rows:
                row.update(
                    {
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                    }
                )
            for row in jspace_rows:
                row.update(
                    {
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                    }
                )

            run_dir = out_root / "generations" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "run_id": run_id,
                "model_name": model_name,
                "model_revision": getattr(
                    adapter.hf_model.config, "_name_or_path", model_name
                ),
                "scenario_id": scenario_id,
                "condition": scenario["condition"],
                "seed": int(seed),
                "temperature": float(cfg.get("temperature", 0.0)),
                "top_p": float(cfg.get("top_p", 1.0)),
                "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
                "do_sample": bool(cfg.get("do_sample", False)),
                "prompt_hash": sha256_text(prompt_body),
                "stimulus_hash": sha256_text(stimulus),
                "git_commit": git_commit(root),
                "layers": list(layers) if layers else adapter.available_layers(),
                "option_a": scenario["option_a"],
                "option_b": scenario["option_b"],
                "value_a": scenario["value_a"],
                "value_b": scenario["value_b"],
                "concept_set_a": scenario["concept_set_a"],
                "concept_set_b": scenario["concept_set_b"],
                "concept_token_ids_a": concept_ids.set_a,
                "concept_token_ids_b": concept_ids.set_b,
                "generated_text": text,
                "n_tokens": len(token_rows),
            }
            write_json(run_dir / "metadata.json", meta)
            write_json(run_dir / "prompt.txt.json", {"prompt": prompt_body})
            write_jsonl(run_dir / "tokens.jsonl", token_rows)
            write_jsonl(run_dir / "jspace.jsonl", jspace_rows)
            # also mirror metadata index
            write_json(out_root / "metadata" / f"{run_id}.json", meta)
            written.append(run_dir)
            print(
                f"[{run_id}] {scenario_id} seed={seed} "
                f"tokens={len(token_rows)} layers={len(meta['layers'])}"
            )
            print(f"  choice preview: {text[:200]!r}…")
    return written
