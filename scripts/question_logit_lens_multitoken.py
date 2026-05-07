#!/usr/bin/env python3
"""
Question-organized logit-lens inspection over multiple prompt-token positions.

This script complements direction-only logit lens. It runs real attack prompts,
extracts several consecutive hidden states near the end of the prompt, and maps
each hidden state through the model's final norm and unembedding.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_qwen_ara_multiturn import (  # noqa: E402
    chat_to_prompt,
    load_dataset,
    make_attack_chat,
)


DEFAULT_MODEL = "/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET = "/home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl"
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "ara_qwen_pilot40_s2"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "question_logit_lens"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    return hidden * torch.rsqrt(variance + eps) * weight


def top_tokens(scores: torch.Tensor, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    values, ids = torch.topk(scores, k=k)
    result = []
    for value, token_id in zip(values.tolist(), ids.tolist()):
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        result.append({"token": token, "token_id": int(token_id), "score": float(value)})
    return result


def option_scores(scores: torch.Tensor, tokenizer: Any) -> dict[str, float]:
    result = {}
    for option in ["A", "B", "C", "D", "E"]:
        variants = [option, f" {option}", f"({option}", f"({option})"]
        best = None
        for variant in variants:
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            value = float(scores[int(ids[0])])
            best = value if best is None else max(best, value)
        result[option] = best if best is not None else float("nan")
    return result


def reconstruct_attack_text(row: dict[str, Any]) -> str:
    template = row["metadata"]["prompt_template"]
    return template.format(
        correct_answer=row["correct_answer"],
        incorrect_answer=row["incorrect_target"],
    )


@torch.inference_mode()
def inspect_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    last_tokens: int,
    top_k: int,
    device: torch.device,
    norm_weight: torch.Tensor,
    eps: float,
) -> list[dict[str, Any]]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    input_ids = encoded["input_ids"][0].detach().cpu()
    seq_len = int(input_ids.shape[0])
    start = max(0, seq_len - last_tokens)

    unembedding = model.get_input_embeddings().weight.detach().float().to(device)
    norm_weight = norm_weight.to(device)

    rows = []
    for pos in range(start, seq_len):
        token_id = int(input_ids[pos])
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        layer_rows = []
        for layer in layers:
            hidden = outputs.hidden_states[layer + 1][0, pos, :].detach().float()
            lens_hidden = rms_norm(hidden, norm_weight, eps)
            scores = torch.mv(unembedding, lens_hidden)
            opts = option_scores(scores.detach().cpu(), tokenizer)
            layer_rows.append(
                {
                    "layer": layer,
                    "top_tokens": top_tokens(scores.detach().cpu(), tokenizer, top_k),
                    "option_scores": opts,
                }
            )
        rows.append(
            {
                "position": pos,
                "relative_position": pos - seq_len,
                "token_id": token_id,
                "token": token,
                "layers": layer_rows,
            }
        )
    return rows


def top_tokens_inline(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        token = md_escape(row["token"].replace("\n", "\\n"))
        parts.append(f"`{token}` {row['score']:.2f}")
    return ", ".join(parts)


def option_inline(scores: dict[str, float]) -> str:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{key}:{value:.2f}" for key, value in ordered)


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Question Logit Lens Over Consecutive Prompt Tokens",
        "",
        f"- Model: `{results['model']}`",
        f"- Run dir: `{results['run_dir']}`",
        f"- Layers: `{', '.join(map(str, results['layers']))}`",
        f"- Last prompt tokens per attack: `{results['last_tokens']}`",
        "",
        "Each table is organized by original question, then attack. The token rows are consecutive prompt positions near the end of the attack prompt, before the model generates the final answer.",
        "",
    ]
    for q_idx, question in enumerate(results["questions"], 1):
        lines.extend(
            [
                f"## Q{q_idx}",
                "",
                f"**Question:** {md_escape(truncate(question['question'], 900))}",
                "",
                f"**Correct answer:** `{question['correct_answer']}`",
                "",
            ]
        )
        for attack in question["attacks"]:
            lines.extend(
                [
                    f"### {attack['attack_type']}",
                    "",
                    f"**Attack text:** {md_escape(attack['attack_text'])}",
                    "",
                    f"**Turn 1 response:** `{attack['turn1_response']}`",
                    "",
                    f"**Baseline final:** `{attack['baseline_response']}` -> `{attack['baseline_eval']}`",
                    "",
                    f"**ARA final:** `{attack['ara_response']}` -> `{attack['ara_eval']}`",
                    "",
                    "| rel_pos | prompt_token | layer | top lens tokens | option logits A-E |",
                    "| ---: | --- | ---: | --- | --- |",
                ]
            )
            for token_row in attack["tokens"]:
                token = md_escape(token_row["token"].replace("\n", "\\n"))
                for layer_row in token_row["layers"]:
                    lines.append(
                        "| "
                        f"{token_row['relative_position']} | "
                        f"`{token}` | "
                        f"{layer_row['layer']} | "
                        f"{top_tokens_inline(layer_row['top_tokens'])} | "
                        f"{option_inline(layer_row['option_scores'])} |"
                    )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="12,18,24")
    parser.add_argument("--last-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--attacks-per-question", type=int, default=6)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    norm_weight = model.model.norm.weight.detach().float()

    dataset_rows = load_dataset(args.dataset, limit=40, seed=42)
    item_by_question = {item.get("base", {}).get("question", ""): item for item in dataset_rows}
    baseline_rows = read_jsonl(args.run_dir / "qwen_ara_baseline_multiturn.jsonl")
    ara_rows = read_jsonl(args.run_dir / "qwen_ara_intervened_multiturn.jsonl")
    ara_by_key = {(row["question"], row["attack_type"]): row for row in ara_rows}

    grouped = defaultdict(list)
    for row in baseline_rows:
        if row["question"] in item_by_question:
            grouped[row["question"]].append(row)

    results = {
        "model": args.model,
        "run_dir": str(args.run_dir),
        "layers": layers,
        "last_tokens": args.last_tokens,
        "top_k": args.top_k,
        "questions": [],
    }

    for q_idx, (question, rows) in enumerate(grouped.items(), 1):
        if len(results["questions"]) >= args.max_questions:
            break
        item = item_by_question[question]
        question_result = {
            "question": question,
            "correct_answer": rows[0]["correct_answer"],
            "attacks": [],
        }
        for row in rows[: args.attacks_per_question]:
            ara_row = ara_by_key.get((row["question"], row["attack_type"]), {})
            attack_text = reconstruct_attack_text(row)
            prompt = chat_to_prompt(tokenizer, make_attack_chat(item, row["turn1_response"], attack_text))
            print(f"inspect question {q_idx}, {row['attack_type']}")
            token_rows = inspect_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=layers,
                last_tokens=args.last_tokens,
                top_k=args.top_k,
                device=device,
                norm_weight=norm_weight,
                eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            )
            question_result["attacks"].append(
                {
                    "attack_type": row["attack_type"],
                    "attack_text": attack_text,
                    "turn1_response": row["turn1_response"],
                    "baseline_response": row["turn2_response"],
                    "baseline_eval": row["final_eval"],
                    "ara_response": ara_row.get("turn2_response", ""),
                    "ara_eval": ara_row.get("final_eval", ""),
                    "tokens": token_rows,
                }
            )
        results["questions"].append(question_result)

    json_path = args.output_dir / "question_logit_lens_multitoken.json"
    md_path = args.output_dir / "question_logit_lens_multitoken.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
