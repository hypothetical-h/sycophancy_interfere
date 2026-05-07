#!/usr/bin/env python3
"""Generate SVG visualizations for rank-k ARA sweep results."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


COLORS = {
    "net_gain": "#175e8a",
    "wrong_to_correct": "#166534",
    "correct_to_wrong": "#b91c1c",
    "mean_delta_margin": "#7c3aed",
    "baseline_flip_rate": "#525252",
    "ara_flip_rate": "#c2410c",
}


def label_params(params: dict[str, Any]) -> str:
    layers = "-".join(str(layer) for layer in params["layers"])
    return f"L{layers} a{params['alpha']} r{params['rank']} tw{params['token_window']}"


def stage_label(stage: str, params: dict[str, Any]) -> str:
    if stage == "layers":
        return "[" + ",".join(str(layer) for layer in params["layers"]) + "]"
    if stage == "alpha":
        return str(params["alpha"])
    if stage == "rank":
        return str(params["rank"])
    if stage == "token_window":
        return str(params["token_window"])
    return label_params(params)


def scale(values: list[float], lo: float, hi: float, height: float, top: float) -> list[float]:
    if hi == lo:
        hi += 1
        lo -= 1
    return [top + (hi - value) * height / (hi - lo) for value in values]


def svg_grouped_bars(path: Path, stage: str, title: str, rows: list[dict[str, Any]], metrics: list[str], y_label: str) -> None:
    width = max(980, 150 + 92 * len(rows))
    height = 520
    left, right, top, bottom = 80, 30, 58, 135
    plot_w = width - left - right
    plot_h = height - top - bottom
    labels = [stage_label(stage, row["params"]) for row in rows]
    values = [[float(row[metric]) for row in rows] for metric in metrics]
    flat = [value for metric_values in values for value in metric_values]
    lo = min(0.0, min(flat))
    hi = max(0.0, max(flat))
    pad = (hi - lo) * 0.08 or 1.0
    lo -= pad
    hi += pad
    zero_y = scale([0.0], lo, hi, plot_h, top)[0]
    group_w = plot_w / max(1, len(rows))
    bar_w = min(22, group_w / (len(metrics) + 1.4))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#555"/>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left+plot_w}" y2="{zero_y:.2f}" stroke="#888" stroke-dasharray="4 4"/>',
        f'<text x="{left-10}" y="{top+5}" text-anchor="end" font-family="monospace" font-size="11">{hi:.2f}</text>',
        f'<text x="{left-10}" y="{zero_y-4}" text-anchor="end" font-family="monospace" font-size="11">0</text>',
        f'<text x="{left-10}" y="{top+plot_h}" text-anchor="end" font-family="monospace" font-size="11">{lo:.2f}</text>',
    ]
    for row_idx, row in enumerate(rows):
        center = left + group_w * row_idx + group_w / 2
        for metric_idx, metric in enumerate(metrics):
            value = float(row[metric])
            y = scale([value], lo, hi, plot_h, top)[0]
            x = center - (len(metrics) * bar_w) / 2 + metric_idx * bar_w
            bar_top = min(y, zero_y)
            bar_h = abs(zero_y - y)
            parts.append(
                f'<rect x="{x:.2f}" y="{bar_top:.2f}" width="{bar_w*0.82:.2f}" height="{bar_h:.2f}" '
                f'fill="{COLORS.get(metric, "#444")}"><title>{html.escape(label_params(row["params"]))}: {metric}={value:.4f}</title></rect>'
            )
        parts.append(
            f'<text x="{center:.2f}" y="{height-42}" transform="rotate(-35 {center:.2f} {height-42})" '
            f'text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(labels[row_idx])}</text>'
        )
    legend_x = left
    for metric in metrics:
        parts.append(f'<rect x="{legend_x}" y="{height-22}" width="12" height="12" fill="{COLORS.get(metric, "#444")}"/>')
        parts.append(f'<text x="{legend_x+17}" y="{height-12}" font-family="sans-serif" font-size="12">{html.escape(metric)}</text>')
        legend_x += 17 + len(metric) * 7 + 18
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.results.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for stage, rows in results["stages"].items():
        ordered = sorted(rows, key=lambda row: (row["net_gain"], row["mean_delta_margin"], -row["correct_to_wrong"], row["valid_rate"]), reverse=True)
        path = output_dir / f"{stage}_transitions.svg"
        svg_grouped_bars(path, stage, f"{stage}: transition counts", ordered, ["net_gain", "wrong_to_correct", "correct_to_wrong"], "count")
        artifacts[f"{stage}_transitions_svg"] = str(path.relative_to(args.results.parent))
        path = output_dir / f"{stage}_margin.svg"
        svg_grouped_bars(path, stage, f"{stage}: mean delta margin", ordered, ["mean_delta_margin"], "logit margin")
        artifacts[f"{stage}_margin_svg"] = str(path.relative_to(args.results.parent))
        path = output_dir / f"{stage}_flip_rate.svg"
        svg_grouped_bars(path, stage, f"{stage}: baseline vs ARA flip rate", ordered, ["baseline_flip_rate", "ara_flip_rate"], "flip rate")
        artifacts[f"{stage}_flip_rate_svg"] = str(path.relative_to(args.results.parent))
    artifact_path = output_dir / "sweep_visualization_artifacts.json"
    artifact_path.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "artifacts": artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
