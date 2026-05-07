#!/usr/bin/env python3
"""
Logit-lens inspection for saved ARA direction vectors.

The ARA runner saves one unit vector per selected transformer layer. This script
maps those vectors into vocabulary-logit space using the model's tied unembedding
matrix, so the directions can be inspected as token preferences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DIRECTIONS = PROJECT_ROOT / "outputs" / "ara_qwen_pilot40_s2" / "qwen_ara_directions.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ara_qwen_pilot40_s2" / "logit_lens"

PROBE_TERMS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    " correct",
    " incorrect",
    " wrong",
    " sure",
    " unsure",
    " answer",
    " actually",
    " professor",
    " taught",
    " community",
    " consensus",
    " agree",
    " majority",
    " embarrassed",
    " bet",
    " reconsider",
    " change",
    " final",
    " yes",
    " no",
    " sorry",
    " cannot",
    " I",
    " You",
]


def load_indexed_tensor(model_path: Path, tensor_name: str) -> torch.Tensor:
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    filename = index["weight_map"].get(tensor_name)
    if not filename:
        raise KeyError(f"Tensor {tensor_name!r} not found in {index_path}")
    tensors = load_file(model_path / filename, device="cpu")
    return tensors[tensor_name]


def rms_norm(vector: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = vector.pow(2).mean()
    return vector * torch.rsqrt(variance + eps) * weight


def top_tokens(
    scores: torch.Tensor,
    tokenizer: Any,
    k: int,
    largest: bool = True,
) -> list[dict[str, Any]]:
    values, ids = torch.topk(scores, k=k, largest=largest)
    rows = []
    for value, token_id in zip(values.tolist(), ids.tolist()):
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        rows.append({"token_id": token_id, "token": token, "score": float(value)})
    return rows


def first_token_ids(tokenizer: Any, terms: list[str]) -> dict[str, int]:
    ids = {}
    for term in terms:
        encoded = tokenizer(term, add_special_tokens=False)["input_ids"]
        if encoded:
            ids[term] = int(encoded[0])
    return ids


def option_token_scores(
    scores: torch.Tensor,
    tokenizer: Any,
) -> dict[str, dict[str, Any]]:
    result = {}
    for option in ["A", "B", "C", "D", "E"]:
        variants = [option, f" {option}", f"({option}", f"({option})"]
        variant_rows = []
        for variant in variants:
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if not encoded:
                continue
            token_id = int(encoded[0])
            variant_rows.append(
                {
                    "variant": variant,
                    "token_id": token_id,
                    "token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "score": float(scores[token_id]),
                }
            )
        result[option] = {
            "best_score": max(row["score"] for row in variant_rows),
            "variants": sorted(variant_rows, key=lambda row: row["score"], reverse=True),
        }
    return result


def probe_token_scores(
    scores: torch.Tensor,
    tokenizer: Any,
    terms: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for term, token_id in first_token_ids(tokenizer, terms).items():
        rows.append(
            {
                "term": term,
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False),
                "score": float(scores[token_id]),
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def analyze_layer(
    layer: int,
    direction: torch.Tensor,
    unembedding: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    direction = direction.float()
    direct_scores = torch.mv(unembedding, direction)
    normed_scores = torch.mv(unembedding, rms_norm(direction, norm_weight, eps))

    return {
        "layer": layer,
        "direction_norm": float(direction.norm()),
        "direct_positive_top": top_tokens(direct_scores, tokenizer, top_k, largest=True),
        "direct_negative_top": top_tokens(direct_scores, tokenizer, top_k, largest=False),
        "normalized_positive_top": top_tokens(normed_scores, tokenizer, top_k, largest=True),
        "normalized_negative_top": top_tokens(normed_scores, tokenizer, top_k, largest=False),
        "direct_option_scores": option_token_scores(direct_scores, tokenizer),
        "normalized_option_scores": option_token_scores(normed_scores, tokenizer),
        "direct_probe_scores": probe_token_scores(direct_scores, tokenizer, PROBE_TERMS),
        "normalized_probe_scores": probe_token_scores(normed_scores, tokenizer, PROBE_TERMS),
    }


def tokens_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| rank | token | token_id | score |", "| ---: | --- | ---: | ---: |"]
    for idx, row in enumerate(rows, 1):
        token = row["token"].replace("\n", "\\n").replace("|", "\\|")
        lines.append(f"| {idx} | `{token}` | {row['token_id']} | {row['score']:.4f} |")
    return "\n".join(lines)


def options_table(scores: dict[str, dict[str, Any]]) -> str:
    rows = sorted(scores.items(), key=lambda item: item[1]["best_score"], reverse=True)
    lines = ["| option | best_score | best_variant | token_id |", "| --- | ---: | --- | ---: |"]
    for option, data in rows:
        best = data["variants"][0]
        variant = best["variant"].replace("|", "\\|")
        lines.append(f"| {option} | {data['best_score']:.4f} | `{variant}` | {best['token_id']} |")
    return "\n".join(lines)


def probes_table(rows: list[dict[str, Any]], limit: int = 12) -> str:
    lines = ["| term | token | token_id | score |", "| --- | --- | ---: | ---: |"]
    for row in rows[:limit]:
        term = row["term"].replace("|", "\\|")
        token = row["token"].replace("\n", "\\n").replace("|", "\\|")
        lines.append(f"| `{term}` | `{token}` | {row['token_id']} | {row['score']:.4f} |")
    return "\n".join(lines)


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# ARA Direction Logit Lens",
        "",
        "This report maps each saved ARA direction into vocabulary space.",
        "",
        "- `direct`: `logits = W_U direction`, the linear logit contribution of the unit direction.",
        "- `normalized`: `logits = W_U RMSNorm(direction)`, a closer match to a standard final-layer logit lens.",
        "- `negative_top`: tokens favored by `-direction`; these are the logits most reduced by the positive direction.",
        "",
    ]
    for layer_result in results["layers"]:
        layer = layer_result["layer"]
        lines.extend(
            [
                f"## Layer {layer}",
                "",
                "### Direct Positive Top Tokens",
                "",
                tokens_table(layer_result["direct_positive_top"]),
                "",
                "### Direct Negative Top Tokens",
                "",
                tokens_table(layer_result["direct_negative_top"]),
                "",
                "### Normalized Positive Top Tokens",
                "",
                tokens_table(layer_result["normalized_positive_top"]),
                "",
                "### Normalized Negative Top Tokens",
                "",
                tokens_table(layer_result["normalized_negative_top"]),
                "",
                "### Direct Option Scores",
                "",
                options_table(layer_result["direct_option_scores"]),
                "",
                "### Probe Terms, Direct",
                "",
                probes_table(layer_result["direct_probe_scores"]),
                "",
                "### Probe Terms, Normalized",
                "",
                probes_table(layer_result["normalized_probe_scores"]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    directions = torch.load(args.directions, map_location="cpu")

    unembedding = load_indexed_tensor(args.model, "model.embed_tokens.weight").float()
    norm_weight = load_indexed_tensor(args.model, "model.norm.weight").float()

    results = {
        "model": str(args.model),
        "directions": str(args.directions),
        "top_k": args.top_k,
        "lens_definitions": {
            "direct": "logits = unembedding @ direction",
            "normalized": "logits = unembedding @ RMSNorm(direction)",
        },
        "layers": [],
    }
    for layer in sorted(directions):
        print(f"analyzing layer {layer}")
        results["layers"].append(
            analyze_layer(
                layer=layer,
                direction=directions[layer],
                unembedding=unembedding,
                norm_weight=norm_weight,
                eps=float(getattr(config, "rms_norm_eps", 1e-6)),
                tokenizer=tokenizer,
                top_k=args.top_k,
            )
        )

    json_path = args.output_dir / "qwen_ara_direction_logit_lens.json"
    md_path = args.output_dir / "qwen_ara_direction_logit_lens.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
