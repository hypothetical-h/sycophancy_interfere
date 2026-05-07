#!/usr/bin/env python3
"""
Qwen multi-turn sycophancy evaluation with arbitrary-rank activation ablation.

Compared with the earlier rank-1 runner, this script estimates one rank-k
subspace per selected layer from multiple consecutive prompt-token hidden-state
deltas, then removes the whole subspace during attacked generation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_qwen_ara_multiturn import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    build_attack_tasks,
    chat_to_prompt,
    evaluate_attacks,
    extract_answer_letter,
    load_dataset,
    make_attack_chat,
    make_base_chat,
    summarize,
    write_jsonl,
    write_summary,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ara_rank_qwen"


@torch.inference_mode()
def hidden_window_by_layer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layers: list[int],
    device: torch.device,
    token_window: int,
) -> dict[int, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    seq_len = encoded["input_ids"].shape[1]
    width = seq_len if token_window <= 0 else min(token_window, seq_len)
    return {
        layer: outputs.hidden_states[layer + 1][0, -width:, :].detach().float().cpu()
        for layer in layers
    }


def estimate_ara_subspaces(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    layers: list[int],
    device: torch.device,
    calibration_trials: int,
    token_window: int,
    rank: int,
) -> dict[int, torch.Tensor]:
    deltas = {layer: [] for layer in layers}
    for idx, task in enumerate(tasks[:calibration_trials], 1):
        base_prompt = chat_to_prompt(
            tokenizer,
            make_base_chat(task["item"]) + [{"role": "assistant", "content": task["turn1_response"]}],
        )
        attack_prompt = chat_to_prompt(
            tokenizer,
            make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]),
        )
        base_h = hidden_window_by_layer(model, tokenizer, base_prompt, layers, device, token_window)
        attack_h = hidden_window_by_layer(model, tokenizer, attack_prompt, layers, device, token_window)
        for layer in layers:
            width = min(base_h[layer].shape[0], attack_h[layer].shape[0])
            deltas[layer].append(attack_h[layer][-width:] - base_h[layer][-width:])
        if idx % 10 == 0:
            print(f"calibration hidden-window deltas: {idx}/{min(calibration_trials, len(tasks))}")

    subspaces = {}
    for layer, windows in deltas.items():
        matrix = torch.cat(windows, dim=0)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        basis = vh[: min(rank, vh.shape[0])]
        basis = torch.nn.functional.normalize(basis, dim=1)
        subspaces[layer] = basis
        kept = singular_values[: basis.shape[0]]
        total_energy = singular_values.pow(2).sum().clamp_min(1e-12)
        kept_energy = kept.pow(2).sum()
        print(
            f"layer {layer}: matrix={tuple(matrix.shape)}, "
            f"rank={basis.shape[0]}, explained_energy={float(kept_energy / total_energy):.4f}"
        )
    return subspaces


class RankSubspaceAblator:
    def __init__(self, model: Any, subspaces: dict[int, torch.Tensor], strength: float, device: torch.device) -> None:
        self.handles = []
        self.model = model
        dtype = next(model.parameters()).dtype
        self.subspaces = {
            layer: basis.to(device=device, dtype=dtype)
            for layer, basis in subspaces.items()
        }
        self.strength = strength

    def __enter__(self) -> "RankSubspaceAblator":
        layers = self.model.model.layers
        for layer_idx, basis in self.subspaces.items():
            self.handles.append(layers[layer_idx].register_forward_hook(self._make_hook(basis)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def _make_hook(self, basis: torch.Tensor):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            projection = torch.matmul(torch.matmul(hidden, basis.transpose(-1, -2)), basis)
            hidden = hidden - self.strength * projection
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook


def evaluate_rank_ara_attacks(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    max_new_tokens: int,
    device: torch.device,
    subspaces: dict[int, torch.Tensor],
    ablation_strength: float,
) -> list[dict[str, Any]]:
    rows = []
    with RankSubspaceAblator(model, subspaces, ablation_strength, device):
        for idx, task in enumerate(tasks, 1):
            prompt = chat_to_prompt(
                tokenizer,
                make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]),
            )
            from run_qwen_ara_multiturn import generate_text

            turn2_response = generate_text(model, tokenizer, prompt, max_new_tokens, device)
            pred = extract_answer_letter(turn2_response)
            if not pred:
                final_eval = "format_error"
            elif pred == task["correct_answer"]:
                final_eval = "correct"
            else:
                final_eval = "wrong"
            rows.append(
                {
                    "question": task["item"].get("base", {}).get("question", ""),
                    "correct_answer": task["correct_answer"],
                    "incorrect_target": task["incorrect_target"],
                    "attack_type": task["attack_type"],
                    "turn1_correct": True,
                    "turn1_response": task["turn1_response"],
                    "turn2_response": turn2_response,
                    "turn2_predicted": pred,
                    "final_eval": final_eval,
                    "metadata": {
                        "prompt_template": task["prompt_template"],
                        "intervention": "rank_ara",
                    },
                }
            )
            if idx % 25 == 0:
                print(f"rank-ara attack eval: {idx}/{len(tasks)}")
    return rows


def write_rank_report(path: Path, comparison: dict[str, Any], metadata: dict[str, Any]) -> None:
    transition_by_attack = comparison.get("transitions", {}).get("by_attack", {})
    lines = [
        "# Rank-k ARA Multiturn Report",
        "",
        "This run estimates arbitrary-rank subspaces from consecutive hidden-state deltas.",
        "",
        "Configuration:",
        "",
        f"- model: `{metadata['model']}`",
        f"- dataset: `{metadata['dataset']}`",
        f"- limit: `{metadata['limit']}`",
        f"- layers: `{metadata['layers']}`",
        f"- rank: `{metadata['rank']}`",
        f"- token_window: `{metadata['token_window']}`",
        f"- calibration_trials: `{metadata['calibration_trials']}`",
        f"- ablation_strength: `{metadata['ablation_strength']}`",
        "",
        "Summary:",
        "",
        f"- baseline flip-rate: `{comparison['baseline']['trial_flip_rate']:.2%}` "
        f"({comparison['baseline']['flipped_trials']}/{comparison['baseline']['valid_attack_trials']})",
        f"- rank-ARA flip-rate: `{comparison['ara']['trial_flip_rate']:.2%}` "
        f"({comparison['ara']['flipped_trials']}/{comparison['ara']['valid_attack_trials']})",
        f"- delta trial flip-rate: `{comparison['delta_trial_flip_rate']:.2%}`",
        "",
        "| Attack | Baseline | Rank-ARA | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    baseline_by_attack = comparison["baseline"]["by_attack"]
    ara_by_attack = comparison["ara"]["by_attack"]
    for attack in sorted(baseline_by_attack):
        base = baseline_by_attack[attack]["flip_rate"]
        ara = ara_by_attack.get(attack, {}).get("flip_rate", 0.0)
        lines.append(f"| {attack} | {base:.2%} | {ara:.2%} | {ara - base:+.2%} |")
    if transition_by_attack:
        lines.extend(
            [
                "",
                "Transitions:",
                "",
                "| Attack | total | wrong->correct | correct->wrong | same wrong | same correct | net gain |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for attack in sorted(transition_by_attack):
            row = transition_by_attack[attack]
            lines.append(
                f"| {attack} | {row['total']} | {row['wrong_to_correct']} | "
                f"{row['correct_to_wrong']} | {row['same_wrong']} | {row['same_correct']} | "
                f"{row['wrong_to_correct'] - row['correct_to_wrong']:+d} |"
            )
        row = comparison["transitions"]["total"]
        lines.append(
            f"| ALL | {row['total']} | {row['wrong_to_correct']} | "
            f"{row['correct_to_wrong']} | {row['same_wrong']} | {row['same_correct']} | "
            f"{row['wrong_to_correct'] - row['correct_to_wrong']:+d} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_transitions(
    baseline_rows: list[dict[str, Any]],
    ara_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ara_by_key = {(row["question"], row["attack_type"]): row for row in ara_rows}
    by_attack: dict[str, dict[str, int]] = {}
    total = {
        "total": 0,
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "same_wrong": 0,
        "same_correct": 0,
    }
    for baseline in baseline_rows:
        ara = ara_by_key[(baseline["question"], baseline["attack_type"])]
        row = by_attack.setdefault(
            baseline["attack_type"],
            {
                "total": 0,
                "wrong_to_correct": 0,
                "correct_to_wrong": 0,
                "same_wrong": 0,
                "same_correct": 0,
            },
        )
        baseline_wrong = baseline["final_eval"] == "wrong"
        ara_wrong = ara["final_eval"] == "wrong"
        baseline_correct = baseline["final_eval"] == "correct"
        ara_correct = ara["final_eval"] == "correct"
        updates = {
            "total": 1,
            "wrong_to_correct": int(baseline_wrong and ara_correct),
            "correct_to_wrong": int(baseline_correct and ara_wrong),
            "same_wrong": int(baseline_wrong and ara_wrong),
            "same_correct": int(baseline_correct and ara_correct),
        }
        for key, value in updates.items():
            row[key] += value
            total[key] += value
    return {"by_attack": by_attack, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20, help="Number of source questions to sample. Use 0 for full dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="12,18,24")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--calibration-trials", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ablation-strength", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    limit = None if args.limit == 0 else args.limit
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    rows = load_dataset(args.dataset, limit, args.seed)
    print(f"loaded source questions: {len(rows)}")

    from run_qwen_ara_multiturn import generate_text

    base_results = []
    for idx, item in enumerate(rows, 1):
        correct = item.get("base", {}).get("correct_answer", item.get("base", {}).get("correct_letter", ""))
        correct = item.get("base", {}).get("correct_letter", correct)
        prompt = chat_to_prompt(tokenizer, make_base_chat(item))
        response = generate_text(model, tokenizer, prompt, args.max_new_tokens, device)
        pred = extract_answer_letter(response)
        base_results.append(
            {
                "question": item.get("base", {}).get("question", ""),
                "correct_answer": correct,
                "turn1_response": response,
                "turn1_predicted": pred,
                "turn1_correct": pred == correct,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "qwen_rank_ara_base_results.jsonl", base_results)

    tasks = build_attack_tasks(rows, base_results, args.seed)
    print(f"valid base questions: {sum(r['turn1_correct'] for r in base_results)}")
    print(f"attack trials: {len(tasks)}")

    baseline_rows = evaluate_attacks(model, tokenizer, tasks, args.max_new_tokens, device)
    baseline_summary = summarize(baseline_rows)
    write_jsonl(args.output_dir / "qwen_rank_ara_baseline_multiturn.jsonl", baseline_rows)

    metadata = {
        "model": args.model,
        "dataset": str(args.dataset),
        "limit": args.limit,
        "layers": layers,
        "rank": args.rank,
        "token_window": args.token_window,
        "calibration_trials": min(args.calibration_trials, len(tasks)),
        "ablation_strength": args.ablation_strength,
    }
    write_summary(args.output_dir / "qwen_rank_ara_baseline_summary.txt", baseline_summary, metadata | {"ablation_strength": args.ablation_strength})

    if tasks:
        subspaces = estimate_ara_subspaces(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            layers=layers,
            device=device,
            calibration_trials=min(args.calibration_trials, len(tasks)),
            token_window=args.token_window,
            rank=args.rank,
        )
        torch.save(
            {
                "subspaces": subspaces,
                "metadata": metadata,
            },
            args.output_dir / "qwen_rank_ara_subspaces.pt",
        )
        ara_rows = evaluate_rank_ara_attacks(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_new_tokens=args.max_new_tokens,
            device=device,
            subspaces=subspaces,
            ablation_strength=args.ablation_strength,
        )
    else:
        ara_rows = []

    ara_summary = summarize(ara_rows)
    write_jsonl(args.output_dir / "qwen_rank_ara_intervened_multiturn.jsonl", ara_rows)
    write_summary(args.output_dir / "qwen_rank_ara_intervened_summary.txt", ara_summary, metadata | {"ablation_strength": args.ablation_strength})

    comparison = {
        "baseline": baseline_summary,
        "ara": ara_summary,
        "delta_trial_flip_rate": ara_summary["trial_flip_rate"] - baseline_summary["trial_flip_rate"],
        "delta_question_flip_rate": ara_summary["question_flip_rate"] - baseline_summary["question_flip_rate"],
        "metadata": metadata,
        "transitions": summarize_transitions(baseline_rows, ara_rows),
    }
    (args.output_dir / "qwen_rank_ara_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_rank_report(args.output_dir / "qwen_rank_ara_report.md", comparison, metadata)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
