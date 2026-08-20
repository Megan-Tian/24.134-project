"""Thin adapter over jlens — do not rewrite Jacobian lens internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

import jlens
from jlens.hooks import ActivationRecorder


def _hub_cached(fn, *args, label: str, **kwargs):
    try:
        return fn(*args, local_files_only=True, **kwargs)
    except (OSError, LocalEntryNotFoundError):
        print(f"Cache miss for {label}; downloading…")
        return fn(*args, local_files_only=False, **kwargs)


@dataclass
class ConceptTokenIds:
    """Token ids used to score concept sets in lens logits."""

    set_a: list[int]
    set_b: list[int]
    phrases_a: list[str]
    phrases_b: list[str]


class JSpaceAdapter:
    """Isolate experiment code from jlens / HF layout details."""

    def __init__(
        self,
        model_name: str,
        *,
        lens_repo: str,
        lens_revision: str,
        lens_file: str,
        dtype: str = "bfloat16",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        torch_dtype = getattr(torch, dtype)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        jlens.configure_logging()
        self.hf_model = _hub_cached(
            transformers.AutoModelForCausalLM.from_pretrained,
            model_name,
            label=model_name,
            dtype=torch_dtype,
        ).to(device)
        self.tokenizer = _hub_cached(
            transformers.AutoTokenizer.from_pretrained,
            model_name,
            label=model_name,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = jlens.from_hf(self.hf_model, self.tokenizer)
        lens_dir = _hub_cached(
            snapshot_download,
            lens_repo,
            label=f"{lens_repo}/{lens_file}",
            allow_patterns=[lens_file],
            revision=lens_revision,
        )
        self.lens = jlens.JacobianLens.from_pretrained(lens_dir, filename=lens_file)
        self.device = device

    def available_layers(self) -> list[int]:
        return list(self.lens.source_layers)

    def concept_token_ids(
        self,
        phrases_a: Sequence[str],
        phrases_b: Sequence[str],
    ) -> ConceptTokenIds:
        def last_ids(phrases: Sequence[str]) -> list[int]:
            ids: list[int] = []
            for phrase in phrases:
                toks = self.tokenizer.encode(phrase, add_special_tokens=False)
                if toks:
                    ids.append(toks[-1])
            # unique, preserve order
            seen: set[int] = set()
            out: list[int] = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            return out

        return ConceptTokenIds(
            set_a=last_ids(phrases_a),
            set_b=last_ids(phrases_b),
            phrases_a=list(phrases_a),
            phrases_b=list(phrases_b),
        )

    @staticmethod
    def score_concept_mass(logits: torch.Tensor, token_ids: Sequence[int]) -> float:
        """Log-sum-exp over concept token logits (scalar)."""
        if not token_ids:
            return 0.0
        vals = logits[list(token_ids)].float()
        return float(torch.logsumexp(vals, dim=0).item())

    def direction_from_logits(
        self,
        logits: torch.Tensor,
        concept_ids: ConceptTokenIds,
    ) -> tuple[float, float, float]:
        """Return (j_direction, score_a, score_b) in approx [-1, +1] for direction."""
        score_a = self.score_concept_mass(logits, concept_ids.set_a)
        score_b = self.score_concept_mass(logits, concept_ids.set_b)
        # tanh keeps roughly [-1, 1] without hard clipping
        direction = float(torch.tanh(torch.tensor(score_b - score_a)).item())
        return direction, score_a, score_b

    @torch.no_grad()
    def readout_at_position(
        self,
        input_ids: torch.Tensor,
        *,
        layers: Sequence[int] | None = None,
        position: int = -1,
    ) -> dict[int, torch.Tensor]:
        """Lens logits at one position for each layer: {layer: [vocab]}."""
        layers = list(layers) if layers is not None else self.available_layers()
        unknown = set(layers) - set(self.lens.source_layers)
        if unknown:
            raise ValueError(f"layers not in fitted lens: {sorted(unknown)}")

        final_layer = self.model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.hf_model.device)

        with ActivationRecorder(self.model.layers, at=record_at) as recorder:
            self.model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        seq_len = activations[final_layer].shape[1]
        pos = position if position >= 0 else seq_len + position
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = activations[layer][0, pos].float()
            transported = self.lens.transport(residual, layer)
            out[layer] = self.model.unembed(transported).float().cpu()
        return out

    def format_chat_prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3 thinking flag when supported
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, text: str, *, max_length: int = 2048) -> torch.Tensor:
        return self.model.encode(text, max_length=max_length)
