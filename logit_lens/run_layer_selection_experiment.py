#!/usr/bin/env python3
"""
Layer-selection experiment for sycophancy intervention using logit lens.

The experiment:
1. Collects questions where the model's first answer is correct.
2. Builds the six existing multi-turn attack prompts.
3. Computes layer-wise logit margins for correct vs user-suggested wrong answers.
4. Locates layers with large attack-induced margin drops and representation shifts.
5. Estimates rank-k ARA subspaces on selected layers.
6. Measures pre/post-intervention logit margins.
7. Compares per-attack subspaces via principal-angle singular values.
8. Writes JSON, Markdown, and SVG visualizations under logit_lens/outputs.
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

ATTACK_SHORT_NAMES = {
    "MT_Attack_1_Doubt": "Doubt",
    "MT_Attack_2_Trap": "Trap",
    "MT_Attack_3_Authority": "Authority",
    "MT_Attack_4_Gaslighting": "Gaslighting",
    "MT_Attack_5_Majority": "Majority",
    "MT_Attack_6_Emotional": "Emotional",
}


class DeviceAwareRankSubspaceAblator:
    def __init__(self, model: Any, subspaces: dict[int, torch.Tensor], strength: float) -> None:
        self.model = model
        self.subspaces = subspaces
        self.strength = strength
        self.handles = []
        self._cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def __enter__(self) -> "DeviceAwareRankSubspaceAblator":
        for layer_idx, basis in self.subspaces.items():
            self.handles.append(self.model.model.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx, basis)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def _basis_for(self, layer_idx: int, basis: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        key = (layer_idx, hidden.device, hidden.dtype)
        if key not in self._cache:
            self._cache[key] = basis.to(device=hidden.device, dtype=hidden.dtype)
        return self._cache[key]

    def _make_hook(self, layer_idx: int, basis: torch.Tensor):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            local_basis = self._basis_for(layer_idx, basis, hidden)
            projection = torch.matmul(torch.matmul(hidden, local_basis.transpose(-1, -2)), local_basis)
            hidden = hidden - self.strength * projection
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook


def option_token_ids(tokenizer: Any, letters: list[str]) -> dict[str, int]:
    result = {}
    for letter in letters:
        variants = [f" {letter}", letter, f"({letter}", f"({letter})"]
        encoded = [(variant, tokenizer(variant, add_special_tokens=False)["input_ids"]) for variant in variants]
        encoded = [(variant, ids) for variant, ids in encoded if ids]
        # Prefer the space-prefixed token when available because generation follows a prompt.
        result[letter] = int(encoded[0][1][0])
    return result


def layer_device(model: Any) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def hidden_states_for_prompt(model: Any, tokenizer: Any, prompt: str, device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    return list(outputs.hidden_states), encoded["input_ids"][0].detach().cpu()


def final_position_margins(
    hidden_states: list[torch.Tensor],
    model: Any,
    correct_id: int,
    wrong_id: int,
) -> list[float]:
    unembed = model.get_input_embeddings().weight.detach()
    norm = model.model.norm
    margins = []
    for layer_hidden in hidden_states[1:]:
        h = layer_hidden[0, -1:, :]
        logits = torch.matmul(norm(h), unembed.t())
        logits = logits[0]
        margins.append(float((logits[correct_id] - logits[wrong_id]).detach().cpu()))
    return margins


def final_position_option_logits(
    hidden_states: list[torch.Tensor],
    model: Any,
    option_ids: dict[str, int],
) -> list[dict[str, float]]:
    unembed = model.get_input_embeddings().weight.detach()
    norm = model.model.norm
    rows = []
    for layer_hidden in hidden_states[1:]:
        h = layer_hidden[0, -1:, :]
        logits = torch.matmul(norm(h), unembed.t())
        logits = logits[0]
        rows.append({letter: float(logits[token_id].detach().cpu()) for letter, token_id in option_ids.items()})
    return rows


def hidden_window(hidden_states: list[torch.Tensor], layer: int, token_window: int) -> torch.Tensor:
    h = hidden_states[layer + 1][0]
    width = h.shape[0] if token_window <= 0 else min(token_window, h.shape[0])
    return h[-width:].detach().float().cpu()


def estimate_subspace(matrix: torch.Tensor, rank: int) -> dict[str, Any]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = torch.nn.functional.normalize(vh[: min(rank, vh.shape[0])], dim=1)
    energy = singular_values.pow(2)
    explained = energy[: basis.shape[0]].sum() / energy.sum().clamp_min(1e-12)
    return {
        "basis": basis,
        "explained_energy": float(explained),
        "singular_values": [float(value) for value in singular_values[: min(10, singular_values.shape[0])]],
    }


def projection_ratio(matrix: torch.Tensor, basis: torch.Tensor) -> float:
    projection = matrix @ basis.t() @ basis
    numerator = projection.norm(dim=1)
    denominator = matrix.norm(dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean())


def subspace_similarity(basis_a: torch.Tensor, basis_b: torch.Tensor) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(basis_a @ basis_b.t()).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(singular_values))
    return {
        "mean_cosine": float(singular_values.mean()),
        "min_cosine": float(singular_values.min()),
        "max_cosine": float(singular_values.max()),
        "min_angle_deg": float(angles.min()),
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg": float(angles.max()),
        "singular_values": [float(value) for value in singular_values],
        "principal_angles_deg": [float(value) for value in angles],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def svg_line_chart(path: Path, title: str, series: dict[str, list[float]], y_label: str = "") -> None:
    width, height = 980, 460
    margin = 60
    all_values = [value for values in series.values() for value in values]
    y_min, y_max = min(all_values), max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad
    n = max(len(values) for values in series.values())

    def x_pos(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, n - 1)

    def y_pos(v: float) -> float:
        return height - margin - (v - y_min) * (height - 2 * margin) / (y_max - y_min)

    colors = ["#175e8a", "#c2410c", "#166534", "#7c3aed", "#b45309"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{margin}" y="32" font-family="monospace" font-size="20" fill="#111">{title}</text>',
        f'<text x="{margin}" y="{height - 18}" font-family="monospace" font-size="12" fill="#444">layer</text>',
        f'<text x="16" y="{margin}" font-family="monospace" font-size="12" fill="#444">{y_label}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    zero_y = y_pos(0)
    if margin <= zero_y <= height - margin:
        parts.append(f'<line x1="{margin}" y1="{zero_y}" x2="{width-margin}" y2="{zero_y}" stroke="#999" stroke-dasharray="4 4"/>')
    for i in range(0, n, max(1, n // 12)):
        parts.append(f'<text x="{x_pos(i)-6}" y="{height-margin+18}" font-family="monospace" font-size="10" fill="#444">{i}</text>')
    for idx, (name, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        legend_y = 58 + idx * 20
        parts.append(f'<rect x="{width-margin-190}" y="{legend_y-10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{width-margin-172}" y="{legend_y}" font-family="monospace" font-size="12" fill="#111">{name}</text>')
    parts.append("</svg>")
    ensure_parent(path)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, title: str, labels: list[str], matrix: list[list[float]]) -> None:
    cell = 58
    margin_left = 150
    margin_top = 80
    width = margin_left + cell * len(labels) + 40
    height = margin_top + cell * len(labels) + 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="24" y="32" font-family="monospace" font-size="18" fill="#111">{title}</text>',
    ]
    for i, label in enumerate(labels):
        short = label[:11]
        parts.append(f'<text x="{margin_left + i * cell + 6}" y="{margin_top - 12}" font-family="monospace" font-size="10" fill="#333" transform="rotate(-35 {margin_left + i * cell + 6},{margin_top - 12})">{short}</text>')
        parts.append(f'<text x="12" y="{margin_top + i * cell + 34}" font-family="monospace" font-size="11" fill="#333">{short}</text>')
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            value = max(0.0, min(1.0, value))
            red = int(255 - 150 * value)
            green = int(245 - 70 * value)
            blue = int(235 - 160 * value)
            x = margin_left + c * cell
            y = margin_top + r * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="rgb({red},{green},{blue})" stroke="#fff"/>')
            parts.append(f'<text x="{x+11}" y="{y+33}" font-family="monospace" font-size="12" fill="#111">{value:.2f}</text>')
    parts.append("</svg>")
    ensure_parent(path)
    path.write_text("\n".join(parts), encoding="utf-8")


def normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def select_layers(margin_drop: list[float], delta_energy: list[float], count: int) -> list[int]:
    # Larger margin_drop means attack pushed correct-vs-wrong margin downward.
    md = normalize(margin_drop)
    de = normalize(delta_energy)
    scores = [a + b for a, b in zip(md, de)]
    return sorted(sorted(range(len(scores)), key=lambda layer: scores[layer], reverse=True)[:count])


def write_report(path: Path, results: dict[str, Any]) -> None:
    selected = results["selected_layers"]
    lines = [
        "# Logit Lens Layer Selection Report",
        "",
        "Configuration:",
        "",
        f"- model: `{results['config']['model']}`",
        f"- target_correct: `{results['config']['target_correct']}`",
        f"- source_seen: `{results['source_seen']}`",
        f"- baseline_correct_collected: `{results['baseline_correct_count']}`",
        f"- attack_trials: `{results['attack_trial_count']}`",
        f"- rank: `{results['config']['rank']}`",
        f"- token_window: `{results['config']['token_window']}`",
        f"- selected_layers: `{selected}`",
        "",
        "Artifacts:",
        "",
    ]
    for name, rel_path in results["artifacts"].items():
        lines.append(f"- {name}: `{rel_path}`")
    lines.extend(
        [
            "",
            "## Module 1: Layer-wise Logit Margin",
            "",
            "Margin is `logit(correct) - logit(user_wrong)`. Positive means the layer favors the correct answer over the user's wrong answer.",
            "",
            "| Layer | base margin | attack margin | attack-base |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for layer, row in enumerate(results["module1"]["by_layer"]):
        lines.append(f"| {layer} | {row['base_margin']:.4f} | {row['attack_margin']:.4f} | {row['attack_minus_base']:.4f} |")
    lines.extend(
        [
            "",
            "## Module 2: Intervention Margin",
            "",
            "| Layer | baseline attack margin | intervened attack margin | delta |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for layer, row in enumerate(results["module2"]["by_layer"]):
        lines.append(f"| {layer} | {row['baseline_attack_margin']:.4f} | {row['intervened_attack_margin']:.4f} | {row['delta_margin']:.4f} |")
    lines.extend(
        [
            "",
            "## Module 3: Layer-wise Delta Energy",
            "",
            "| Layer | mean ||delta h|| | mean cosine(base, attack) |",
            "| ---: | ---: | ---: |",
        ]
    )
    for layer, row in enumerate(results["module3"]["by_layer"]):
        lines.append(f"| {layer} | {row['delta_energy']:.4f} | {row['cosine']:.4f} |")
    lines.extend(
        [
            "",
            "## Module 4: Subspace Quality",
            "",
            "| Layer | explained energy | projection norm ratio |",
            "| ---: | ---: | ---: |",
        ]
    )
    for layer in selected:
        row = results["module4"]["shared_subspaces"][str(layer)]
        lines.append(f"| {layer} | {row['explained_energy']:.4f} | {row['projection_norm_ratio']:.4f} |")
    lines.extend(
        [
            "",
            "Pairwise per-attack subspace similarity is reported as mean singular value of `U_attack_a @ U_attack_b.T`. Higher means more aligned subspaces.",
            "",
        ]
    )
    for layer in selected:
        lines.append(f"### Layer {layer} Attack Subspace Similarity")
        lines.append("")
        labels = results["module4"]["attack_labels"]
        matrix = results["module4"]["attack_similarity"][str(layer)]
        lines.append("| attack | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(labels)) + " |")
        for label, row in zip(labels, matrix):
            lines.append("| " + label + " | " + " | ".join(f"{value:.3f}" for value in row) + " |")
        lines.append("")
        lines.append("Principal angle spectrum by attack pair:")
        lines.append("")
        lines.append("| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        spectrum = results["module4"]["attack_similarity_spectrum"][str(layer)]
        for pair in spectrum:
            sv = ", ".join(f"{value:.3f}" for value in pair["singular_values"])
            angles = ", ".join(f"{value:.1f}" for value in pair["principal_angles_deg"])
            lines.append(
                f"| {pair['attack_a']} vs {pair['attack_b']} | "
                f"`[{sv}]` | `[{angles}]` | {pair['mean_cosine']:.3f} | {pair['max_angle_deg']:.1f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logit_lens" / "outputs" / "layer_selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-correct", type=int, default=200)
    parser.add_argument("--max-source", type=int, default=650)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--token-window", type=int, default=8)
    parser.add_argument("--select-layer-count", type=int, default=3)
    parser.add_argument("--calibration-trials", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ablation-strength", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto", choices=["auto", "single"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        device = layer_device(model)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    option_ids = option_token_ids(tokenizer, ["A", "B", "C", "D", "E"])

    rows = load_dataset(args.dataset, limit=args.max_source, seed=args.seed)
    base_results = []
    kept_items = []
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
        if result["turn1_correct"]:
            kept_items.append(item)
        if len(kept_items) >= args.target_correct:
            break
        if idx % 50 == 0:
            print(f"base scan: source={idx}, correct={len(kept_items)}/{args.target_correct}")

    # build_attack_tasks expects base_results aligned with rows; filter to scanned rows only.
    scanned_rows = rows[:source_seen]
    tasks = build_attack_tasks(scanned_rows, base_results, args.seed)
    tasks = [
        task for task in tasks
        if task["correct_answer"] in option_ids and task["incorrect_target"] in option_ids
    ]
    print(f"collected baseline-correct questions: {len(kept_items)}")
    print(f"attack trials: {len(tasks)}")
    if not tasks:
        raise RuntimeError(
            "No attack tasks were built because no first-turn-correct questions were collected. "
            "Increase --max-source or use --device-map single if sharded generation is producing unparsable answers."
        )

    base_margin_sums = [0.0] * num_layers
    attack_margin_sums = [0.0] * num_layers
    delta_norm_sums = [0.0] * num_layers
    cosine_sums = [0.0] * num_layers
    layer_counts = [0] * num_layers
    matrices_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    matrices_by_layer_attack: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    records = []

    max_calibration = min(args.calibration_trials, len(tasks))
    for idx, task in enumerate(tasks, 1):
        correct_id = option_ids[task["correct_answer"]]
        wrong_id = option_ids[task["incorrect_target"]]
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
        base_margins = final_position_margins(base_hidden, model, correct_id, wrong_id)
        attack_margins = final_position_margins(attack_hidden, model, correct_id, wrong_id)
        attack_option_logits = final_position_option_logits(attack_hidden, model, option_ids)

        for layer in range(num_layers):
            base_margin_sums[layer] += base_margins[layer]
            attack_margin_sums[layer] += attack_margins[layer]
            base_vec = base_hidden[layer + 1][0, -1, :].detach().float().cpu()
            attack_vec = attack_hidden[layer + 1][0, -1, :].detach().float().cpu()
            delta = attack_vec - base_vec
            delta_norm_sums[layer] += float(delta.norm())
            cosine_sums[layer] += float(torch.nn.functional.cosine_similarity(base_vec, attack_vec, dim=0))
            layer_counts[layer] += 1

        if idx <= max_calibration:
            for layer in range(num_layers):
                base_window = hidden_window(base_hidden, layer, args.token_window)
                attack_window = hidden_window(attack_hidden, layer, args.token_window)
                width = min(base_window.shape[0], attack_window.shape[0])
                delta_window = attack_window[-width:] - base_window[-width:]
                matrices_by_layer[layer].append(delta_window)
                matrices_by_layer_attack[layer][task["attack_type"]].append(delta_window)

        records.append(
            {
                "question": task["item"].get("base", {}).get("question", ""),
                "attack_type": task["attack_type"],
                "correct_answer": task["correct_answer"],
                "incorrect_target": task["incorrect_target"],
                "attack_margins": attack_margins,
                "base_margins": base_margins,
                "attack_option_logits": attack_option_logits,
            }
        )
        if idx % 50 == 0:
            print(f"logit lens collection: {idx}/{len(tasks)}")

    base_margin = [base_margin_sums[layer] / layer_counts[layer] for layer in range(num_layers)]
    attack_margin = [attack_margin_sums[layer] / layer_counts[layer] for layer in range(num_layers)]
    margin_drop = [base_margin[layer] - attack_margin[layer] for layer in range(num_layers)]
    delta_energy = [delta_norm_sums[layer] / layer_counts[layer] for layer in range(num_layers)]
    cosine = [cosine_sums[layer] / layer_counts[layer] for layer in range(num_layers)]
    selected_layers = select_layers(margin_drop, delta_energy, args.select_layer_count)
    print(f"selected layers: {selected_layers}")

    shared_subspaces = {}
    per_attack_subspaces: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    module4_shared = {}
    attack_labels = [key for key in sorted(ATTACK_SHORT_NAMES)]
    for layer in selected_layers:
        matrix = torch.cat(matrices_by_layer[layer], dim=0)
        shared = estimate_subspace(matrix, args.rank)
        shared_subspaces[layer] = shared["basis"]
        module4_shared[str(layer)] = {
            "explained_energy": shared["explained_energy"],
            "projection_norm_ratio": projection_ratio(matrix, shared["basis"]),
            "singular_values": shared["singular_values"],
            "matrix_shape": list(matrix.shape),
        }
        for attack_type, windows in matrices_by_layer_attack[layer].items():
            attack_matrix = torch.cat(windows, dim=0)
            per_attack_subspaces[layer][attack_type] = estimate_subspace(attack_matrix, args.rank)["basis"]

    baseline_intervention_margin_sums = [0.0] * num_layers
    intervened_margin_sums = [0.0] * num_layers
    intervention_counts = [0] * num_layers
    with DeviceAwareRankSubspaceAblator(model, shared_subspaces, args.ablation_strength):
        for idx, task in enumerate(tasks, 1):
            correct_id = option_ids[task["correct_answer"]]
            wrong_id = option_ids[task["incorrect_target"]]
            attack_prompt = chat_to_prompt(
                tokenizer,
                make_attack_chat(task["item"], task["turn1_response"], task["attack_text"]),
            )
            hidden, _ = hidden_states_for_prompt(model, tokenizer, attack_prompt, device)
            intervened_margins = final_position_margins(hidden, model, correct_id, wrong_id)
            # Reuse stored attack margins in the same task order.
            baseline_margins = records[idx - 1]["attack_margins"]
            for layer in range(num_layers):
                baseline_intervention_margin_sums[layer] += baseline_margins[layer]
                intervened_margin_sums[layer] += intervened_margins[layer]
                intervention_counts[layer] += 1
            if idx % 50 == 0:
                print(f"intervention margin collection: {idx}/{len(tasks)}")

    baseline_intervention_margin = [
        baseline_intervention_margin_sums[layer] / intervention_counts[layer] for layer in range(num_layers)
    ]
    intervened_margin = [intervened_margin_sums[layer] / intervention_counts[layer] for layer in range(num_layers)]
    delta_margin = [intervened_margin[layer] - baseline_intervention_margin[layer] for layer in range(num_layers)]

    attack_similarity: dict[str, list[list[float]]] = {}
    attack_similarity_spectrum: dict[str, list[dict[str, Any]]] = {}
    readable_attack_labels = [ATTACK_SHORT_NAMES[key] for key in attack_labels]
    for layer in selected_layers:
        matrix = []
        spectrum = []
        for attack_a in attack_labels:
            row = []
            for attack_b in attack_labels:
                sim = subspace_similarity(per_attack_subspaces[layer][attack_a], per_attack_subspaces[layer][attack_b])
                row.append(sim["mean_cosine"])
                if attack_a < attack_b:
                    spectrum.append(
                        {
                            "attack_a": ATTACK_SHORT_NAMES[attack_a],
                            "attack_b": ATTACK_SHORT_NAMES[attack_b],
                            **sim,
                        }
                    )
            matrix.append(row)
        attack_similarity[str(layer)] = matrix
        attack_similarity_spectrum[str(layer)] = spectrum

    artifacts = {
        "module1_margin_svg": "figures/module1_layer_margin.svg",
        "module2_intervention_margin_svg": "figures/module2_intervention_margin.svg",
        "module3_delta_energy_svg": "figures/module3_delta_energy.svg",
    }
    svg_line_chart(
        args.output_dir / artifacts["module1_margin_svg"],
        "Module 1: layer-wise correct-vs-wrong margin",
        {
            "base": base_margin,
            "attack": attack_margin,
            "base-attack": margin_drop,
        },
        y_label="margin",
    )
    svg_line_chart(
        args.output_dir / artifacts["module2_intervention_margin_svg"],
        "Module 2: intervention pre/post attack margin",
        {
            "baseline attack": baseline_intervention_margin,
            "intervened attack": intervened_margin,
            "delta": delta_margin,
        },
        y_label="margin",
    )
    svg_line_chart(
        args.output_dir / artifacts["module3_delta_energy_svg"],
        "Module 3: attack-base representation shift",
        {
            "||delta h||": delta_energy,
            "cosine(base,attack)": cosine,
        },
        y_label="value",
    )
    for layer in selected_layers:
        name = f"figures/module4_attack_subspace_similarity_layer_{layer}.svg"
        artifacts[f"module4_similarity_layer_{layer}_svg"] = name
        svg_heatmap(
            args.output_dir / name,
            f"Module 4: attack subspace similarity, layer {layer}",
            readable_attack_labels,
            attack_similarity[str(layer)],
        )

    results = {
        "config": {
            "model": args.model,
            "dataset": str(args.dataset),
            "seed": args.seed,
            "target_correct": args.target_correct,
            "max_source": args.max_source,
            "rank": args.rank,
            "token_window": args.token_window,
            "select_layer_count": args.select_layer_count,
            "calibration_trials": max_calibration,
            "ablation_strength": args.ablation_strength,
            "device_map": args.device_map,
        },
        "source_seen": source_seen,
        "baseline_correct_count": len(kept_items),
        "attack_trial_count": len(tasks),
        "selected_layers": selected_layers,
        "artifacts": artifacts,
        "module1": {
            "by_layer": [
                {
                    "base_margin": base_margin[layer],
                    "attack_margin": attack_margin[layer],
                    "attack_minus_base": attack_margin[layer] - base_margin[layer],
                    "base_minus_attack": margin_drop[layer],
                }
                for layer in range(num_layers)
            ]
        },
        "module2": {
            "by_layer": [
                {
                    "baseline_attack_margin": baseline_intervention_margin[layer],
                    "intervened_attack_margin": intervened_margin[layer],
                    "delta_margin": delta_margin[layer],
                }
                for layer in range(num_layers)
            ]
        },
        "module3": {
            "by_layer": [
                {
                    "delta_energy": delta_energy[layer],
                    "cosine": cosine[layer],
                }
                for layer in range(num_layers)
            ]
        },
        "module4": {
            "shared_subspaces": module4_shared,
            "attack_labels": readable_attack_labels,
            "attack_similarity": attack_similarity,
            "attack_similarity_spectrum": attack_similarity_spectrum,
        },
    }
    write_json(args.output_dir / "layer_selection_results.json", results)
    torch.save(
        {
            "selected_layers": selected_layers,
            "shared_subspaces": shared_subspaces,
            "config": results["config"],
        },
        args.output_dir / "selected_rank_ara_subspaces.pt",
    )
    write_report(args.output_dir / "layer_selection_report.md", results)
    print(json.dumps({"selected_layers": selected_layers, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
