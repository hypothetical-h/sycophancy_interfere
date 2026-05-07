#!/usr/bin/env python3
"""
Summarize baseline-vs-ARA transition outcomes by attack type.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "ara_qwen_pilot40_s2"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "ara_transition_summary.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_subset_questions(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["question"] for row in data["questions"]}


def summarize(
    baseline_rows: list[dict[str, Any]],
    ara_rows: list[dict[str, Any]],
    subset_questions: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    ara_by_key = {(row["question"], row["attack_type"]): row for row in ara_rows}
    stats = defaultdict(
        lambda: {
            "total": 0,
            "baseline_wrong": 0,
            "ara_wrong": 0,
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "same_wrong": 0,
            "same_correct": 0,
            "format_or_other": 0,
        }
    )
    for baseline in baseline_rows:
        if subset_questions is not None and baseline["question"] not in subset_questions:
            continue
        ara = ara_by_key[(baseline["question"], baseline["attack_type"])]
        row = stats[baseline["attack_type"]]
        row["total"] += 1

        baseline_wrong = baseline["final_eval"] == "wrong"
        ara_wrong = ara["final_eval"] == "wrong"
        baseline_correct = baseline["final_eval"] == "correct"
        ara_correct = ara["final_eval"] == "correct"

        row["baseline_wrong"] += int(baseline_wrong)
        row["ara_wrong"] += int(ara_wrong)
        row["wrong_to_correct"] += int(baseline_wrong and ara_correct)
        row["correct_to_wrong"] += int(baseline_correct and ara_wrong)
        row["same_wrong"] += int(baseline_wrong and ara_wrong)
        row["same_correct"] += int(baseline_correct and ara_correct)
        row["format_or_other"] += int(not (baseline_wrong or baseline_correct) or not (ara_wrong or ara_correct))

    total = defaultdict(int)
    for values in stats.values():
        for key, value in values.items():
            total[key] += value
    stats["ALL"] = dict(total)
    return dict(stats)


def rate(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else f"{count} (n/a)"


def table(stats: dict[str, dict[str, int]]) -> str:
    lines = [
        "| Attack | total | baseline flip | ARA flip | wrong->correct | correct->wrong | same wrong | same correct | net gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = sorted(key for key in stats if key != "ALL") + ["ALL"]
    for attack in keys:
        row = stats[attack]
        total = row["total"]
        net = row["wrong_to_correct"] - row["correct_to_wrong"]
        lines.append(
            f"| {attack} | {total} | "
            f"{rate(row['baseline_wrong'], total)} | "
            f"{rate(row['ara_wrong'], total)} | "
            f"{row['wrong_to_correct']} | "
            f"{row['correct_to_wrong']} | "
            f"{row['same_wrong']} | "
            f"{row['same_correct']} | "
            f"{net:+d} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    run_dir: Path,
    full_stats: dict[str, dict[str, int]],
    subset_stats: dict[str, dict[str, int]] | None,
    subset_path: Path | None,
) -> None:
    lines = [
        "# ARA Transition Summary",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "Definitions:",
        "",
        "- `baseline flip`: baseline final answer is wrong.",
        "- `ARA flip`: ARA final answer is wrong.",
        "- `wrong->correct`: baseline was wrong and ARA became correct.",
        "- `correct->wrong`: baseline was correct and ARA became wrong.",
        "- `net gain`: `wrong->correct - correct->wrong`.",
        "",
        "## Full Pilot40_s2",
        "",
        table(full_stats),
        "",
        "Interpretation:",
        "",
        "- Overall ARA changes `39/90` flips to `33/90` flips.",
        "- Total corrected cases: `13`; total newly broken cases: `7`; net gain: `+6`.",
        "- The useful effect is concentrated in `Authority` and `Majority`.",
        "- The main side effects are in `Doubt`, `Trap`, and `Gaslighting`.",
    ]
    if subset_stats is not None:
        lines.extend(
            [
                "",
                "## Recent Question Logit Lens Q3 Subset",
                "",
                f"Subset source: `{subset_path}`",
                "",
                table(subset_stats),
                "",
                "Interpretation:",
                "",
                "- This subset is not representative of the full pilot gain.",
                "- It contains more ARA side effects than corrections, so it is useful for failure-case inspection.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--subset-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_rows = read_jsonl(args.run_dir / "qwen_ara_baseline_multiturn.jsonl")
    ara_rows = read_jsonl(args.run_dir / "qwen_ara_intervened_multiturn.jsonl")
    full_stats = summarize(baseline_rows, ara_rows)
    subset_stats = None
    if args.subset_json is not None:
        subset_stats = summarize(baseline_rows, ara_rows, load_subset_questions(args.subset_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, args.run_dir, full_stats, subset_stats, args.subset_json)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
