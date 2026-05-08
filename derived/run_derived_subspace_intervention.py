#!/usr/bin/env python3
"""
Estimate ARA subspaces from one set of layers and apply them at another layer.

This tests whether layers that contain a strong/consistent sycophancy direction
can be used as the subspace-generating layers, while the actual intervention is
applied at a later readout-sensitive layer.
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
from run_qwen_ara_rank_multiturn import (  # noqa: E402
    RankSubspaceAblator,
    hidden_window_by_layer,
    summarize_transitions,
)
from run_layer_selection_experiment import option_token_ids  # noqa: E402
from run_rank_ara_hyperparameter_sweep import (  # noqa: E402
    evaluate_rank_ara_with_margins,
    next_token_margin,
    parse_layer_groups,
    parse_layers,
    safe_mean,
    sort_key,
)


DEFAULT_OUTPUT_DIR = ROOT / "derived" / "outputs" / "derived_subspace_q200"


def estimate_derived_subspace(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    derive_layers: list[int],
    target_layers: list[int],
    device: torch.device,
    calibration_trials: int,
    token_window: int,
    rank: int,
) -> dict[int, torch.Tensor]:
    deltas = []
    for idx, task in enumerate(tasks[:calibration_trials], 1):
        base_prompt = chat_to_prompt(
            tokenizer,
            make_base_chat(task["item"]) + [{"role": "assistant", "content": task["turn1_response"]}],
        )
        attack_prompt = chat_to_prompt(tokenizer, make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]))
        base_h = hidden_window_by_layer(model, tokenizer, base_prompt, derive_layers, device, token_window)
        attack_h = hidden_window_by_layer(model, tokenizer, attack_prompt, derive_layers, device, token_window)
        for layer in derive_layers:
            width = min(base_h[layer].shape[0], attack_h[layer].shape[0])
            deltas.append(attack_h[layer][-width:] - base_h[layer][-width:])
        if idx % 10 == 0:
            print(f"derived calibration: {idx}/{calibration_trials}")
    matrix = torch.cat(deltas, dim=0)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = torch.nn.functional.normalize(vh[: min(rank, vh.shape[0])], dim=1)
    energy = singular_values.pow(2)
    explained = float(energy[: basis.shape[0]].sum() / energy.sum().clamp_min(1e-12))
    print(
        f"derive_layers={derive_layers} -> target_layers={target_layers}: "
        f"matrix={tuple(matrix.shape)}, rank={basis.shape[0]}, explained_energy={explained:.4f}"
    )
    return {target_layer: basis for target_layer in target_layers}


def summarize_combo(
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
    delta_margin_values = [ara - base for base, ara in paired]
    baseline_summary = summarize(baseline_rows)
    ara_summary = summarize(ara_rows)
    return {
        "params": params,
        "baseline_flip_rate": baseline_summary["trial_flip_rate"],
        "ara_flip_rate": ara_summary["trial_flip_rate"],
        "delta_flip_rate": ara_summary["trial_flip_rate"] - baseline_summary["trial_flip_rate"],
        "valid_rate": valid_count / len(ara_rows) if ara_rows else 0.0,
        "wrong_to_correct": total["wrong_to_correct"],
        "correct_to_wrong": total["correct_to_wrong"],
        "same_wrong": total["same_wrong"],
        "same_correct": total["same_correct"],
        "net_gain": total["wrong_to_correct"] - total["correct_to_wrong"],
        "mean_delta_margin": safe_mean(delta_margin_values),
        "transition_total": total,
        "transition_by_attack": transitions["by_attack"],
    }


def write_report(path: Path, results: dict[str, Any]) -> None:
    rows = sorted(results["results"], key=sort_key, reverse=True)
    lines = [
        "# Derived Subspace Intervention Report",
        "",
        "Question: can we derive an ARA subspace from one layer or layer group, then apply it at a different readout-sensitive layer?",
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
            "| rank | derive layers | target layers | alpha | subspace rank | token_window | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | ARA flip |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, 1):
        params = row["params"]
        lines.append(
            f"| {idx} | `{params['derive_layers']}` | `{params['target_layers']}` | {params['alpha']:.2f} | "
            f"{params['rank']} | {params['token_window']} | {row['net_gain']:+d} | {row['wrong_to_correct']} | "
            f"{row['correct_to_wrong']} | {row['valid_rate']:.2%} | {row['mean_delta_margin']:.4f} | {row['ara_flip_rate']:.2%} |"
        )
    if rows:
        best = rows[0]
        lines.extend(["", f"Best params by selection rule: `{best['params']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=650)
    parser.add_argument("--target-correct", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--derive-layer-groups", default="31;32;33;31,32,33;35")
    parser.add_argument("--target-layers", default="35")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--calibration-trials", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    derive_groups = parse_layer_groups(args.derive_layer_groups)
    target_layers = parse_layers(args.target_layers)
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
                "metadata": {"prompt_template": task["prompt_template"], "intervention": "none"},
            }
        )
        if idx % 50 == 0:
            print(f"baseline attack eval: {idx}/{len(tasks)}")
    write_jsonl(args.output_dir / "baseline_attacks.jsonl", baseline_rows)

    combo_results = []
    for derive_layers in derive_groups:
        subspaces = estimate_derived_subspace(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            derive_layers=derive_layers,
            target_layers=target_layers,
            device=device,
            calibration_trials=min(args.calibration_trials, len(tasks)),
            token_window=args.token_window,
            rank=args.rank,
        )
        ara_rows, intervened_margins = evaluate_rank_ara_with_margins(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_new_tokens=args.max_new_tokens,
            device=device,
            subspaces=subspaces,
            alpha=args.alpha,
            option_ids=option_ids,
        )
        params = {
            "derive_layers": derive_layers,
            "target_layers": target_layers,
            "alpha": args.alpha,
            "rank": args.rank,
            "token_window": args.token_window,
        }
        summary = summarize_combo(params, baseline_rows, ara_rows, baseline_margins, intervened_margins)
        name = "derive_" + "-".join(str(layer) for layer in derive_layers) + "__target_" + "-".join(str(layer) for layer in target_layers)
        combo_dir = args.output_dir / "combos" / name
        combo_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(combo_dir / "intervened_attacks.jsonl", ara_rows)
        (combo_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        combo_results.append(summary)
        (args.output_dir / "derived_results_partial.json").write_text(
            json.dumps({"results": combo_results}, ensure_ascii=False, indent=2) + "\n",
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
            "derive_layer_groups": derive_groups,
            "target_layers": target_layers,
            "rank": args.rank,
            "token_window": args.token_window,
            "calibration_trials": min(args.calibration_trials, len(tasks)),
            "alpha": args.alpha,
            "max_new_tokens": args.max_new_tokens,
        },
        "valid_base_questions": sum(r["turn1_correct"] for r in base_results),
        "attack_trial_count": len(tasks),
        "baseline_summary": summarize(baseline_rows),
        "results": combo_results,
    }
    (args.output_dir / "derived_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "derived_report.md", results)
    print(json.dumps({"output_dir": str(args.output_dir), "best": sorted(combo_results, key=sort_key, reverse=True)[0]["params"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
