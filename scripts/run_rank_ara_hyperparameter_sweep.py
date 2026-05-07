#!/usr/bin/env python3
"""
Chained hyperparameter sweep for rank-k ARA.

The sweep follows workflow.md:
1. Sweep layer groups with fixed alpha/rank/token_window.
2. Fix the best layer group and sweep alpha.
3. Fix best layers+alpha and sweep rank.
4. Fix best layers+alpha+rank and sweep token_window.

For every parameter combination, the script records final-answer transitions,
valid rate, baseline/intervened correct-vs-wrong logit margins, and net gain.
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
sys.path.insert(0, str(PROJECT_ROOT / "logit_lens"))

from run_qwen_ara_multiturn import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    build_attack_tasks,
    chat_to_prompt,
    extract_answer_letter,
    generate_text,
    load_dataset,
    make_attack_chat,
    make_base_chat,
    summarize,
    write_jsonl,
)
from run_qwen_ara_rank_multiturn import (  # noqa: E402
    RankSubspaceAblator,
    estimate_ara_subspaces,
    summarize_transitions,
)
from run_layer_selection_experiment import option_token_ids  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rank_ara_hyper_sweep_limit40"


def parse_layers(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_layer_groups(value: str) -> list[list[int]]:
    return [parse_layers(group) for group in value.split(";") if group.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def combo_name(stage: str, layers: list[int], alpha: float, rank: int, token_window: int) -> str:
    layer_text = "-".join(str(layer) for layer in layers)
    alpha_text = str(alpha).replace(".", "p")
    return f"{stage}__L{layer_text}__a{alpha_text}__r{rank}__tw{token_window}"


@torch.inference_mode()
def next_token_margin(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    option_ids: dict[str, int],
    device: torch.device,
) -> float | None:
    correct = task["correct_answer"]
    wrong = task["incorrect_target"]
    if correct not in option_ids or wrong not in option_ids:
        return None
    prompt = chat_to_prompt(tokenizer, make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]))
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    logits = model(**encoded, use_cache=False).logits[0, -1].detach().float()
    return float(logits[option_ids[correct]] - logits[option_ids[wrong]])


def evaluate_rank_ara_with_margins(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    max_new_tokens: int,
    device: torch.device,
    subspaces: dict[int, torch.Tensor],
    alpha: float,
    option_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], list[float | None]]:
    rows = []
    margins = []
    with RankSubspaceAblator(model, subspaces, alpha, device):
        for idx, task in enumerate(tasks, 1):
            margins.append(next_token_margin(model, tokenizer, task, option_ids, device))
            prompt = chat_to_prompt(tokenizer, make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]))
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
            if idx % 50 == 0:
                print(f"combo eval: {idx}/{len(tasks)}")
    return rows, margins


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_combo(
    stage: str,
    params: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    ara_rows: list[dict[str, Any]],
    baseline_margins: list[float | None],
    intervened_margins: list[float | None],
) -> dict[str, Any]:
    transitions = summarize_transitions(baseline_rows, ara_rows)
    total = transitions["total"]
    valid_count = sum(1 for row in ara_rows if row["final_eval"] != "format_error")
    paired = [
        (base, ara)
        for base, ara in zip(baseline_margins, intervened_margins)
        if base is not None and ara is not None
    ]
    baseline_margin_values = [base for base, _ in paired]
    intervened_margin_values = [ara for _, ara in paired]
    delta_margin_values = [ara - base for base, ara in paired]
    baseline_summary = summarize(baseline_rows)
    ara_summary = summarize(ara_rows)
    net_gain = total["wrong_to_correct"] - total["correct_to_wrong"]
    return {
        "stage": stage,
        "params": params,
        "baseline_flip_rate": baseline_summary["trial_flip_rate"],
        "ara_flip_rate": ara_summary["trial_flip_rate"],
        "delta_flip_rate": ara_summary["trial_flip_rate"] - baseline_summary["trial_flip_rate"],
        "valid_rate": valid_count / len(ara_rows) if ara_rows else 0.0,
        "wrong_to_correct": total["wrong_to_correct"],
        "correct_to_wrong": total["correct_to_wrong"],
        "same_wrong": total["same_wrong"],
        "same_correct": total["same_correct"],
        "net_gain": net_gain,
        "mean_baseline_margin": safe_mean(baseline_margin_values),
        "mean_intervened_margin": safe_mean(intervened_margin_values),
        "mean_delta_margin": safe_mean(delta_margin_values),
        "transition_total": total,
        "transition_by_attack": transitions["by_attack"],
    }


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        row["net_gain"],
        row["mean_delta_margin"],
        -row["correct_to_wrong"],
        row["valid_rate"],
    )


def write_markdown_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Rank-k ARA Hyperparameter Sweep",
        "",
        "Selection rule: sort by `net_gain`, then `mean_delta_margin`, then fewer `correct_to_wrong`, then `valid_rate`.",
        "",
        "Configuration:",
        "",
    ]
    for key, value in results["config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            f"- valid_base_questions: `{results['valid_base_questions']}`",
            f"- attack_trials: `{results['attack_trial_count']}`",
            "",
        ]
    )
    for stage in ["layers", "alpha", "rank", "token_window"]:
        rows = results["stages"].get(stage, [])
        lines.extend(
            [
                f"## {stage}",
                "",
                "| rank | layers | alpha | subspace rank | token_window | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | baseline flip | ARA flip |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, row in enumerate(sorted(rows, key=sort_key, reverse=True), 1):
            params = row["params"]
            lines.append(
                f"| {idx} | `{params['layers']}` | {params['alpha']:.2f} | {params['rank']} | "
                f"{params['token_window']} | {row['net_gain']:+d} | {row['wrong_to_correct']} | "
                f"{row['correct_to_wrong']} | {row['valid_rate']:.2%} | {row['mean_delta_margin']:.4f} | "
                f"{row['baseline_flip_rate']:.2%} | {row['ara_flip_rate']:.2%} |"
            )
        lines.append("")
        if rows:
            best = max(rows, key=sort_key)
            lines.append(f"Best `{stage}` params: `{best['params']}`")
            lines.append("")
    lines.extend(
        [
            "## Final Choice",
            "",
            f"`{results['final_choice']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--target-correct", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layer-groups",
        default="22,23,24;24,25,26;26,27,28;28,29,30;31,32,33;32,33,34;33,34,35;24;27;30;33;35",
    )
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--ranks", default="1,2,4,8,16")
    parser.add_argument("--token-windows", default="1,4,8,16,32")
    parser.add_argument("--default-alpha", type=float, default=1.0)
    parser.add_argument("--default-rank", type=int, default=4)
    parser.add_argument("--default-token-window", type=int, default=16)
    parser.add_argument("--calibration-trials", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layer_groups = parse_layer_groups(args.layer_groups)
    alphas = parse_floats(args.alphas)
    ranks = parse_ints(args.ranks)
    token_windows = parse_ints(args.token_windows)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    option_ids = option_token_ids(tokenizer, ["A", "B", "C", "D", "E"])

    limit = None if args.limit == 0 else args.limit
    rows = load_dataset(args.dataset, limit, args.seed)
    print(f"loaded source questions: {len(rows)}")

    base_results = []
    scanned_rows = []
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
        scanned_rows.append(item)
        if idx % 20 == 0:
            print(f"base eval: {idx}/{len(rows)}")
        if args.target_correct and sum(result["turn1_correct"] for result in base_results) >= args.target_correct:
            break
    write_jsonl(args.output_dir / "base_results.jsonl", base_results)

    tasks = build_attack_tasks(scanned_rows, base_results, args.seed)
    print(f"valid base questions: {sum(result['turn1_correct'] for result in base_results)}")
    print(f"attack trials: {len(tasks)}")

    baseline_rows = []
    baseline_margins = []
    for idx, task in enumerate(tasks, 1):
        baseline_margins.append(next_token_margin(model, tokenizer, task, option_ids, device))
        prompt = chat_to_prompt(tokenizer, make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]))
        turn2_response = generate_text(model, tokenizer, prompt, args.max_new_tokens, device)
        pred = extract_answer_letter(turn2_response)
        if not pred:
            final_eval = "format_error"
        elif pred == task["correct_answer"]:
            final_eval = "correct"
        else:
            final_eval = "wrong"
        baseline_rows.append(
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
                "metadata": {"prompt_template": task["prompt_template"], "intervention": "none"},
            }
        )
        if idx % 50 == 0:
            print(f"baseline attack eval: {idx}/{len(tasks)}")
    write_jsonl(args.output_dir / "baseline_attacks.jsonl", baseline_rows)

    stages: dict[str, list[dict[str, Any]]] = {"layers": [], "alpha": [], "rank": [], "token_window": []}
    subspace_cache: dict[tuple[tuple[int, ...], int, int], dict[int, torch.Tensor]] = {}

    def get_subspaces(layers: list[int], rank: int, token_window: int) -> dict[int, torch.Tensor]:
        key = (tuple(layers), rank, token_window)
        if key not in subspace_cache:
            print(f"estimating subspaces: layers={layers}, rank={rank}, token_window={token_window}")
            subspace_cache[key] = estimate_ara_subspaces(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                layers=layers,
                device=device,
                calibration_trials=min(args.calibration_trials, len(tasks)),
                token_window=token_window,
                rank=rank,
            )
        return subspace_cache[key]

    def run_combo(stage: str, layers: list[int], alpha: float, rank: int, token_window: int) -> dict[str, Any]:
        print(f"running combo: stage={stage}, layers={layers}, alpha={alpha}, rank={rank}, token_window={token_window}")
        subspaces = get_subspaces(layers, rank, token_window)
        ara_rows, intervened_margins = evaluate_rank_ara_with_margins(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_new_tokens=args.max_new_tokens,
            device=device,
            subspaces=subspaces,
            alpha=alpha,
            option_ids=option_ids,
        )
        params = {"layers": layers, "alpha": alpha, "rank": rank, "token_window": token_window}
        summary = summarize_combo(stage, params, baseline_rows, ara_rows, baseline_margins, intervened_margins)
        name = combo_name(stage, layers, alpha, rank, token_window)
        combo_dir = args.output_dir / "combos" / name
        combo_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(combo_dir / "intervened_attacks.jsonl", ara_rows)
        (combo_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stages[stage].append(summary)
        (args.output_dir / "sweep_results_partial.json").write_text(
            json.dumps({"stages": stages}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "params": params,
                    "net_gain": summary["net_gain"],
                    "mean_delta_margin": summary["mean_delta_margin"],
                    "correct_to_wrong": summary["correct_to_wrong"],
                    "valid_rate": summary["valid_rate"],
                },
                ensure_ascii=False,
            )
        )
        return summary

    for layers in layer_groups:
        run_combo("layers", layers, args.default_alpha, args.default_rank, args.default_token_window)
    best_layers = max(stages["layers"], key=sort_key)["params"]["layers"]

    for alpha in alphas:
        run_combo("alpha", best_layers, alpha, args.default_rank, args.default_token_window)
    best_alpha = max(stages["alpha"], key=sort_key)["params"]["alpha"]

    for rank in ranks:
        run_combo("rank", best_layers, best_alpha, rank, args.default_token_window)
    best_rank = max(stages["rank"], key=sort_key)["params"]["rank"]

    for token_window in token_windows:
        run_combo("token_window", best_layers, best_alpha, best_rank, token_window)
    best_token_window = max(stages["token_window"], key=sort_key)["params"]["token_window"]

    final_choice = {
        "layers": best_layers,
        "alpha": best_alpha,
        "rank": best_rank,
        "token_window": best_token_window,
    }
    results = {
        "config": {
            "model": args.model,
            "dataset": str(args.dataset),
            "limit": args.limit,
            "target_correct": args.target_correct,
            "source_questions_scanned": len(scanned_rows),
            "seed": args.seed,
            "layer_groups": layer_groups,
            "alphas": alphas,
            "ranks": ranks,
            "token_windows": token_windows,
            "default_alpha": args.default_alpha,
            "default_rank": args.default_rank,
            "default_token_window": args.default_token_window,
            "calibration_trials": min(args.calibration_trials, len(tasks)),
            "max_new_tokens": args.max_new_tokens,
        },
        "valid_base_questions": sum(result["turn1_correct"] for result in base_results),
        "attack_trial_count": len(tasks),
        "baseline_summary": summarize(baseline_rows),
        "stages": stages,
        "final_choice": final_choice,
    }
    (args.output_dir / "sweep_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(args.output_dir / "sweep_report.md", results)
    print(json.dumps({"output_dir": str(args.output_dir), "final_choice": final_choice}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
