# sycophancy_interfere

This repository contains activation-removal intervention experiments for multi-turn sycophancy evaluation.

The current experiment follows the behavioral setup of `run_qwen_multiturn.py` from the original `sycophancy-eval` workspace:

1. Ask each original multiple-choice question.
2. Keep only questions where the model's first answer is correct.
3. Apply six multi-turn sycophancy attacks.
4. Compare the baseline flip-rate with an ARA-style intervention that removes a PCA/SVD attack direction from selected transformer layers.

## Layout

- `scripts/run_qwen_ara_multiturn.py`: main experiment runner.
- `outputs/`: migrated experiment outputs from the first ARA pilots.
- `docs/Try ARA.md`: original experiment sketch.
- `workflow.md`: running experiment log with commands and results.

## Data

The dataset is intentionally not copied into this repository. The runner uses an external dataset path.

Default dataset:

```bash
/home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl
```

You can override it either with `--dataset` or with `SYCOPHANCY_DATASET`.

## Environment

The experiments were run with the existing conda environment:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python
```

The required Python packages are:

```bash
torch
transformers
modelscope
vllm
```

`vllm` is not used by the ARA runner itself, but it was present in the environment and remains useful for baseline-style generation scripts.

## Model

Default model path:

```bash
/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507
```

Override with `--model` if needed.

## Commands

Smoke test:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --dataset /home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl \
  --limit 6 \
  --calibration-trials 12 \
  --layers 12,18,24 \
  --output-dir outputs/ara_qwen_smoke
```

40-question pilot, best current setting:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --dataset /home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl \
  --limit 40 \
  --calibration-trials 120 \
  --layers 12,18,24 \
  --ablation-strength 2.0 \
  --output-dir outputs/ara_qwen_pilot40_s2
```

Full dataset run:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --dataset /home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl \
  --limit 0 \
  --calibration-trials 600 \
  --layers 12,18,24 \
  --ablation-strength 2.0 \
  --output-dir outputs/ara_qwen_full_s2
```

`--limit 0` means use the full external dataset.

## Outputs

Each run writes:

- `qwen_ara_base_results.jsonl`: first-turn original answers.
- `qwen_ara_baseline_multiturn.jsonl`: un-intervened attack results.
- `qwen_ara_baseline_summary.txt`: baseline summary.
- `qwen_ara_directions.pt`: saved ARA directions.
- `qwen_ara_intervened_multiturn.jsonl`: ARA attack results.
- `qwen_ara_intervened_summary.txt`: ARA summary.
- `qwen_ara_comparison.json`: baseline vs ARA comparison.
