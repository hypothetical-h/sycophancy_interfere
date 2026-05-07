#!/usr/bin/env python3
"""
Readout-alignment experiment for sycophancy hidden shifts.

For each layer, compare the attack-induced hidden shift

    d_syco = h_attack - h_base

with the answer readout direction

    d_logit = W_lm[wrong_answer] - W_lm[correct_answer]

The goal is to locate layers where the attack shift points most strongly toward
increasing the user's wrong answer relative to the correct answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
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
)
from run_layer_selection_experiment import (  # noqa: E402
    ATTACK_SHORT_NAMES,
    hidden_states_for_prompt,
    option_token_ids,
    svg_line_chart,
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def angle_from_cosine(value: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def top_layers(rows: list[dict[str, float]], key: str, n: int = 8, reverse: bool = True) -> list[dict[str, float]]:
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[:n]


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Readout Alignment Report",
        "",
        "This experiment checks whether attack-induced hidden shifts align with the wrong-answer readout direction.",
        "",
        "Definitions:",
        "",
        "```text",
        "d_syco,l = h_attack,l - h_base,l",
        "d_logit = W_lm[id(user_wrong)] - W_lm[id(correct)]",
        "alignment_l = cosine(d_syco,l, d_logit)",
        "projection_l = dot(d_syco,l, normalize(d_logit))",
        "angle_l = arccos(alignment_l)",
        "```",
        "",
        "Configuration:",
        "",
        f"- model: `{results['config']['model']}`",
        f"- target_correct: `{results['config']['target_correct']}`",
        f"- max_source: `{results['config']['max_source']}`",
        f"- baseline_correct_collected: `{results['baseline_correct_count']}`",
        f"- attack_trial_count: `{results['attack_trial_count']}`",
        "",
        "Artifacts:",
        "",
    ]
    for name, rel_path in results["artifacts"].items():
        lines.append(f"- {name}: `{rel_path}`")

    lines.extend(["", "## Top Layers By Mean Alignment", "", "| layer | mean alignment | mean angle deg | mean projection |", "| ---: | ---: | ---: | ---: |"])
    for row in top_layers(results["overall_by_layer"], "mean_alignment", n=10):
        lines.append(f"| {int(row['layer'])} | {row['mean_alignment']:.4f} | {row['mean_angle_deg']:.2f} | {row['mean_projection']:.4f} |")

    lines.extend(["", "## Top Layers By Projection Strength", "", "| layer | mean projection | mean alignment | mean angle deg |", "| ---: | ---: | ---: | ---: |"])
    for row in top_layers(results["overall_by_layer"], "mean_projection", n=10):
        lines.append(f"| {int(row['layer'])} | {row['mean_projection']:.4f} | {row['mean_alignment']:.4f} | {row['mean_angle_deg']:.2f} |")

    lines.extend(["", "## All Layers", "", "| layer | mean alignment | mean angle deg | mean projection | mean ||d_syco|| |", "| ---: | ---: | ---: | ---: | ---: |"])
    for row in results["overall_by_layer"]:
        lines.append(
            f"| {int(row['layer'])} | {row['mean_alignment']:.4f} | {row['mean_angle_deg']:.2f} | "
            f"{row['mean_projection']:.4f} | {row['mean_shift_norm']:.4f} |"
        )

    lines.extend(["", "## By Attack Type", ""])
    for attack, rows in results["by_attack"].items():
        lines.extend([f"### {attack}", "", "| layer | mean alignment | mean angle deg | mean projection |", "| ---: | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(f"| {int(row['layer'])} | {row['mean_alignment']:.4f} | {row['mean_angle_deg']:.2f} | {row['mean_projection']:.4f} |")
        best = top_layers(rows, "mean_alignment", n=5)
        lines.append("")
        lines.append("Top alignment layers: " + ", ".join(f"{int(row['layer'])} ({row['mean_alignment']:.3f})" for row in best))
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- High positive alignment means the attack shift points toward increasing the user-suggested wrong answer relative to the correct answer.",
            "- High projection means the signed component of the shift along that readout direction is large.",
            "- Layers that are high in both metrics are stronger candidates for ARA layer sweeps.",
        ]
    )
    ensure_parent(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logit_lens" / "outputs" / "readout_alignment_q200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-correct", type=int, default=200)
    parser.add_argument("--max-source", type=int, default=650)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    option_ids = option_token_ids(tokenizer, ["A", "B", "C", "D", "E"])
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        output_embeddings = model.get_input_embeddings()
    unembed = output_embeddings.weight.detach().float().cpu()

    rows = load_dataset(args.dataset, limit=args.max_source, seed=args.seed)
    base_results = []
    kept_items = []
    scanned_items = []
    source_seen = 0
    for idx, item in enumerate(rows, 1):
        source_seen = idx
        correct = item.get("base", {}).get("correct_answer", item.get("base", {}).get("correct_letter", ""))
        correct = item.get("base", {}).get("correct_letter", correct)
        if correct not in option_ids:
            continue
        prompt = chat_to_prompt(tokenizer, make_base_chat(item))
        response = generate_text(model, tokenizer, prompt, args.max_new_tokens, device)
        pred = extract_answer_letter(response)
        result = {
            "question": item.get("base", {}).get("question", ""),
            "correct_answer": correct,
            "turn1_response": response,
            "turn1_predicted": pred,
            "turn1_correct": pred == correct,
        }
        base_results.append(result)
        scanned_items.append(item)
        if result["turn1_correct"]:
            kept_items.append(item)
        if len(kept_items) >= args.target_correct:
            break
        if idx % 50 == 0:
            print(f"base scan: source={idx}, correct={len(kept_items)}/{args.target_correct}")

    tasks = build_attack_tasks(scanned_items, base_results, args.seed)
    tasks = [
        task for task in tasks
        if task["correct_answer"] in option_ids and task["incorrect_target"] in option_ids
    ]
    if not tasks:
        raise RuntimeError("No valid attack tasks were built.")
    print(f"collected baseline-correct questions: {len(kept_items)}")
    print(f"attack trials: {len(tasks)}")

    sums = {
        "alignment": [0.0] * num_layers,
        "angle": [0.0] * num_layers,
        "projection": [0.0] * num_layers,
        "shift_norm": [0.0] * num_layers,
        "count": [0] * num_layers,
    }
    attack_sums = {
        attack: {
            "alignment": [0.0] * num_layers,
            "angle": [0.0] * num_layers,
            "projection": [0.0] * num_layers,
            "shift_norm": [0.0] * num_layers,
            "count": [0] * num_layers,
        }
        for attack in ATTACK_SHORT_NAMES.values()
    }

    for idx, task in enumerate(tasks, 1):
        correct_id = option_ids[task["correct_answer"]]
        wrong_id = option_ids[task["incorrect_target"]]
        d_logit = unembed[wrong_id] - unembed[correct_id]
        d_logit_unit = d_logit / d_logit.norm().clamp_min(1e-12)
        base_prompt = chat_to_prompt(
            tokenizer,
            make_base_chat(task["item"]) + [{"role": "assistant", "content": task["turn1_response"]}],
        )
        attack_prompt = chat_to_prompt(
            tokenizer,
            make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]),
        )
        base_hidden, _ = hidden_states_for_prompt(model, tokenizer, base_prompt, device)
        attack_hidden, _ = hidden_states_for_prompt(model, tokenizer, attack_prompt, device)
        attack_name = ATTACK_SHORT_NAMES[task["attack_type"]]
        for layer in range(num_layers):
            base_vec = base_hidden[layer + 1][0, -1, :].detach().float().cpu()
            attack_vec = attack_hidden[layer + 1][0, -1, :].detach().float().cpu()
            d_syco = attack_vec - base_vec
            shift_norm = float(d_syco.norm())
            alignment = float(torch.nn.functional.cosine_similarity(d_syco, d_logit, dim=0))
            angle = angle_from_cosine(alignment)
            projection = float(torch.dot(d_syco, d_logit_unit))
            for target in (sums, attack_sums[attack_name]):
                target["alignment"][layer] += alignment
                target["angle"][layer] += angle
                target["projection"][layer] += projection
                target["shift_norm"][layer] += shift_norm
                target["count"][layer] += 1
        if idx % 50 == 0:
            print(f"readout alignment: {idx}/{len(tasks)}")

    def rows_from(acc: dict[str, list[float]]) -> list[dict[str, float]]:
        output = []
        for layer in range(num_layers):
            count = acc["count"][layer]
            output.append(
                {
                    "layer": layer,
                    "mean_alignment": acc["alignment"][layer] / count,
                    "mean_angle_deg": acc["angle"][layer] / count,
                    "mean_projection": acc["projection"][layer] / count,
                    "mean_shift_norm": acc["shift_norm"][layer] / count,
                }
            )
        return output

    overall = rows_from(sums)
    by_attack = {attack: rows_from(acc) for attack, acc in attack_sums.items()}
    artifacts = {
        "overall_alignment_svg": "figures/overall_alignment.svg",
        "overall_projection_svg": "figures/overall_projection.svg",
        "by_attack_alignment_svg": "figures/by_attack_alignment.svg",
        "by_attack_projection_svg": "figures/by_attack_projection.svg",
    }
    svg_line_chart(
        args.output_dir / artifacts["overall_alignment_svg"],
        "Readout alignment: mean cosine(d_syco, d_logit)",
        {"overall": [row["mean_alignment"] for row in overall]},
        y_label="cosine",
    )
    svg_line_chart(
        args.output_dir / artifacts["overall_projection_svg"],
        "Projection strength: dot(d_syco, normalize(d_logit))",
        {"overall": [row["mean_projection"] for row in overall]},
        y_label="projection",
    )
    svg_line_chart(
        args.output_dir / artifacts["by_attack_alignment_svg"],
        "Readout alignment by attack type",
        {attack: [row["mean_alignment"] for row in rows] for attack, rows in by_attack.items()},
        y_label="cosine",
    )
    svg_line_chart(
        args.output_dir / artifacts["by_attack_projection_svg"],
        "Projection strength by attack type",
        {attack: [row["mean_projection"] for row in rows] for attack, rows in by_attack.items()},
        y_label="projection",
    )

    results = {
        "config": {
            "model": args.model,
            "dataset": str(args.dataset),
            "seed": args.seed,
            "target_correct": args.target_correct,
            "max_source": args.max_source,
            "max_new_tokens": args.max_new_tokens,
        },
        "source_seen": source_seen,
        "baseline_scanned_count": len(scanned_items),
        "baseline_correct_count": len(kept_items),
        "attack_trial_count": len(tasks),
        "artifacts": artifacts,
        "overall_by_layer": overall,
        "by_attack": by_attack,
    }
    write_json(args.output_dir / "readout_alignment_results.json", results)
    write_report(args.output_dir / "readout_alignment_report.md", results)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "top_alignment_layers": [
                    int(row["layer"]) for row in top_layers(overall, "mean_alignment", n=8)
                ],
                "top_projection_layers": [
                    int(row["layer"]) for row in top_layers(overall, "mean_projection", n=8)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
