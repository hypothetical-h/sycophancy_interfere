#!/usr/bin/env python3
"""
Detector-executor ARA.

Detector: uses current attack prompt option-logit gap to decide intervention
strength per sample:

    score = logit(user_wrong) - logit(correct)
    gate = 1[score > threshold]

Executor: always applies the layer-35 subspace U_35 when gate is on:

    h_35' = h_35 - alpha * gate * U_35 U_35^T h_35
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


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "logit_lens"))

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
from run_qwen_ara_rank_multiturn import estimate_ara_subspaces, summarize_transitions  # noqa: E402
from run_layer_selection_experiment import option_token_ids  # noqa: E402
from run_rank_ara_hyperparameter_sweep import next_token_margin, safe_mean, sort_key  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "derived" / "outputs" / "detector_executor_q200"


class GatedRankSubspaceAblator:
    def __init__(self, model: Any, subspaces: dict[int, torch.Tensor], alpha: float, gate: float, device: torch.device) -> None:
        self.model = model
        self.alpha = alpha
        self.gate = gate
        self.handles = []
        dtype = next(model.parameters()).dtype
        self.subspaces = {layer: basis.to(device=device, dtype=dtype) for layer, basis in subspaces.items()}

    def __enter__(self) -> "GatedRankSubspaceAblator":
        for layer, basis in self.subspaces.items():
            self.handles.append(self.model.model.layers[layer].register_forward_hook(self._make_hook(basis)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def _make_hook(self, basis: torch.Tensor):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            projection = torch.matmul(torch.matmul(hidden, basis.transpose(-1, -2)), basis)
            hidden = hidden - self.alpha * self.gate * projection
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook


@torch.inference_mode()
def option_gap(
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    option_ids: dict[str, int],
    device: torch.device,
) -> float | None:
    margin = next_token_margin(model, tokenizer, task, option_ids, device)
    if margin is None:
        return None
    return -margin


def evaluate_gated(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    max_new_tokens: int,
    device: torch.device,
    subspaces: dict[int, torch.Tensor],
    alpha: float,
    threshold: float,
    option_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], list[float | None], list[float], list[float | None]]:
    rows = []
    margins = []
    gates = []
    scores = []
    for idx, task in enumerate(tasks, 1):
        score = option_gap(model, tokenizer, task, option_ids, device)
        gate = 1.0 if score is not None and score > threshold else 0.0
        scores.append(score)
        gates.append(gate)
        prompt = chat_to_prompt(tokenizer, make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]))
        with GatedRankSubspaceAblator(model, subspaces, alpha, gate, device):
            margins.append(next_token_margin(model, tokenizer, task, option_ids, device))
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
                "detector_score": score,
                "gate": gate,
                "metadata": {"intervention": "detector_executor_ara", "threshold": threshold},
            }
        )
        if idx % 50 == 0:
            print(f"gated eval threshold={threshold}: {idx}/{len(tasks)}")
    return rows, margins, gates, scores


def summarize_combo(
    threshold: float,
    baseline_rows: list[dict[str, Any]],
    ara_rows: list[dict[str, Any]],
    baseline_margins: list[float | None],
    intervened_margins: list[float | None],
    gates: list[float],
    scores: list[float | None],
) -> dict[str, Any]:
    transitions = summarize_transitions(baseline_rows, ara_rows)
    total = transitions["total"]
    paired = [(b, a) for b, a in zip(baseline_margins, intervened_margins) if b is not None and a is not None]
    delta_margin_values = [a - b for b, a in paired]
    baseline_summary = summarize(baseline_rows)
    ara_summary = summarize(ara_rows)
    gated_scores = [score for score, gate in zip(scores, gates) if score is not None and gate > 0]
    all_scores = [score for score in scores if score is not None]
    return {
        "params": {"threshold": threshold},
        "baseline_flip_rate": baseline_summary["trial_flip_rate"],
        "ara_flip_rate": ara_summary["trial_flip_rate"],
        "delta_flip_rate": ara_summary["trial_flip_rate"] - baseline_summary["trial_flip_rate"],
        "valid_rate": sum(row["final_eval"] != "format_error" for row in ara_rows) / len(ara_rows),
        "wrong_to_correct": total["wrong_to_correct"],
        "correct_to_wrong": total["correct_to_wrong"],
        "same_wrong": total["same_wrong"],
        "same_correct": total["same_correct"],
        "net_gain": total["wrong_to_correct"] - total["correct_to_wrong"],
        "mean_delta_margin": safe_mean(delta_margin_values),
        "gate_rate": safe_mean(gates),
        "mean_score": safe_mean(all_scores),
        "mean_gated_score": safe_mean(gated_scores),
        "transition_total": total,
        "transition_by_attack": transitions["by_attack"],
    }


def write_report(path: Path, results: dict[str, Any]) -> None:
    rows = sorted(results["results"], key=sort_key, reverse=True)
    lines = [
        "# Detector-Executor ARA Report",
        "",
        "Detector: `score = logit(user_wrong) - logit(correct)` on the attack prompt.",
        "Executor: gated layer-35 rank-k ARA using `U_35`.",
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
            f"- baseline flip-rate: `{results['baseline_summary']['trial_flip_rate']:.2%}`",
            "",
            "| rank | threshold | gate rate | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | ARA flip |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, 1):
        lines.append(
            f"| {idx} | {row['params']['threshold']:.2f} | {row['gate_rate']:.2%} | {row['net_gain']:+d} | "
            f"{row['wrong_to_correct']} | {row['correct_to_wrong']} | {row['valid_rate']:.2%} | "
            f"{row['mean_delta_margin']:.4f} | {row['ara_flip_rate']:.2%} |"
        )
    if rows:
        lines.append("")
        lines.append(f"Best params by selection rule: `{rows[0]['params']}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=650)
    parser.add_argument("--target-correct", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--executor-layer", type=int, default=35)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--calibration-trials", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--thresholds", default="-4,-2,-1,0,1,2,4")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]

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

    rows = load_dataset(args.dataset, args.limit, args.seed)
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
        if idx % 50 == 0:
            print(f"base scan: source={idx}, correct={sum(r['turn1_correct'] for r in base_results)}/{args.target_correct}")
        if args.target_correct and sum(r["turn1_correct"] for r in base_results) >= args.target_correct:
            break
    write_jsonl(args.output_dir / "base_results.jsonl", base_results)
    tasks = build_attack_tasks(scanned_rows, base_results, args.seed)
    print(f"valid base questions: {sum(r['turn1_correct'] for r in base_results)}")
    print(f"attack trials: {len(tasks)}")

    baseline_rows = []
    baseline_margins = []
    for idx, task in enumerate(tasks, 1):
        baseline_margins.append(next_token_margin(model, tokenizer, task, option_ids, device))
        prompt = chat_to_prompt(tokenizer, make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]))
        response = generate_text(model, tokenizer, prompt, args.max_new_tokens, device)
        pred = extract_answer_letter(response)
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
                "turn2_response": response,
                "turn2_predicted": pred,
                "final_eval": final_eval,
                "metadata": {"intervention": "none"},
            }
        )
        if idx % 50 == 0:
            print(f"baseline attack eval: {idx}/{len(tasks)}")
    write_jsonl(args.output_dir / "baseline_attacks.jsonl", baseline_rows)

    subspaces = estimate_ara_subspaces(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        layers=[args.executor_layer],
        device=device,
        calibration_trials=min(args.calibration_trials, len(tasks)),
        token_window=args.token_window,
        rank=args.rank,
    )

    results_rows = []
    for threshold in thresholds:
        ara_rows, intervened_margins, gates, scores = evaluate_gated(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_new_tokens=args.max_new_tokens,
            device=device,
            subspaces=subspaces,
            alpha=args.alpha,
            threshold=threshold,
            option_ids=option_ids,
        )
        summary = summarize_combo(threshold, baseline_rows, ara_rows, baseline_margins, intervened_margins, gates, scores)
        combo_dir = args.output_dir / "combos" / f"threshold_{str(threshold).replace('-', 'm').replace('.', 'p')}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(combo_dir / "intervened_attacks.jsonl", ara_rows)
        (combo_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results_rows.append(summary)
        (args.output_dir / "detector_executor_results_partial.json").write_text(
            json.dumps({"results": results_rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))

    results = {
        "config": {
            "model": args.model,
            "dataset": str(args.dataset),
            "limit": args.limit,
            "target_correct": args.target_correct,
            "source_questions_scanned": len(scanned_rows),
            "seed": args.seed,
            "executor_layer": args.executor_layer,
            "rank": args.rank,
            "token_window": args.token_window,
            "calibration_trials": min(args.calibration_trials, len(tasks)),
            "alpha": args.alpha,
            "thresholds": thresholds,
            "max_new_tokens": args.max_new_tokens,
        },
        "valid_base_questions": sum(r["turn1_correct"] for r in base_results),
        "attack_trial_count": len(tasks),
        "baseline_summary": summarize(baseline_rows),
        "results": results_rows,
    }
    (args.output_dir / "detector_executor_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "detector_executor_report.md", results)
    print(json.dumps({"output_dir": str(args.output_dir), "best": sorted(results_rows, key=sort_key, reverse=True)[0]["params"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
