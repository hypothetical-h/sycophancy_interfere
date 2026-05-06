# Workflow

This file records the ARA experiment runs, commands, and observed results.

## 2026-05-06: Initial ARA implementation

Goal: turn the sketch in `docs/Try ARA.md` into a runnable experiment that compares baseline multi-turn sycophancy flip-rate against an activation-removal intervention.

Implementation:

- Added `scripts/run_qwen_ara_multiturn.py`.
- Kept the multi-turn attack setup aligned with `run_qwen_multiturn.py`.
- Used Qwen/Qwen3-4B-Instruct-2507 from the local ModelScope cache.
- Did not copy the dataset into this repository; the runner points to the external `are_you_sure.jsonl`.
- Estimated one attack direction per selected layer using PCA/SVD over hidden-state deltas:

```text
attacked_conversation_hidden - base_conversation_hidden
```

- During attacked generation, removed the projection onto the selected directions with transformer layer forward hooks.

Environment:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python
```

External dataset:

```bash
/home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl
```

Model:

```bash
/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507
```

## 2026-05-06: Smoke test

Command:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --limit 6 \
  --calibration-trials 12 \
  --layers 12,18,24 \
  --output-dir outputs/ara_qwen_smoke
```

Result:

- Source questions: 6
- Valid first-turn-correct questions: 3
- Attack trials: 18
- Baseline trial flip-rate: 66.67% (`12/18`)
- ARA trial flip-rate: 61.11% (`11/18`)
- Delta trial flip-rate: -5.56 percentage points
- Question any-flip rate: 100.00% -> 100.00%

Interpretation: pipeline works end to end. Sample is too small to judge intervention quality.

## 2026-05-06: Pilot 40, strength 1.0

Command:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --limit 40 \
  --calibration-trials 120 \
  --layers 12,18,24 \
  --output-dir outputs/ara_qwen_pilot40
```

Result:

- Source questions: 40
- Valid first-turn-correct questions: 15
- Attack trials: 90
- Baseline trial flip-rate: 43.33% (`39/90`)
- ARA trial flip-rate: 43.33% (`39/90`)
- Delta trial flip-rate: 0.00 percentage points
- Question any-flip rate: 100.00% -> 100.00%

By attack:

| Attack | Baseline | ARA |
| --- | ---: | ---: |
| Doubt | 6.67% | 13.33% |
| Trap | 13.33% | 20.00% |
| Authority | 86.67% | 80.00% |
| Gaslighting | 26.67% | 26.67% |
| Majority | 100.00% | 100.00% |
| Emotional | 26.67% | 20.00% |

Interpretation: strength 1.0 changes behavior but does not reduce total trial flip-rate.

## 2026-05-06: Pilot 40, strength 2.0

Command:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --limit 40 \
  --calibration-trials 120 \
  --layers 12,18,24 \
  --ablation-strength 2.0 \
  --output-dir outputs/ara_qwen_pilot40_s2
```

Result:

- Source questions: 40
- Valid first-turn-correct questions: 15
- Attack trials: 90
- Baseline trial flip-rate: 43.33% (`39/90`)
- ARA trial flip-rate: 36.67% (`33/90`)
- Delta trial flip-rate: -6.67 percentage points
- Question any-flip rate: 100.00% -> 60.00%

By attack:

| Attack | Baseline | ARA |
| --- | ---: | ---: |
| Doubt | 6.67% | 26.67% |
| Trap | 13.33% | 26.67% |
| Authority | 86.67% | 46.67% |
| Gaslighting | 26.67% | 33.33% |
| Majority | 100.00% | 60.00% |
| Emotional | 26.67% | 26.67% |

Interpretation: best pilot setting so far. It reduces total trial flip-rate and strongly reduces Authority/Majority susceptibility, but increases some weaker attack categories.

## 2026-05-06: Pilot 40, strength 3.0

Command:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --limit 40 \
  --calibration-trials 120 \
  --layers 12,18,24 \
  --ablation-strength 3.0 \
  --output-dir outputs/ara_qwen_pilot40_s3
```

Result:

- Source questions: 40
- Valid first-turn-correct questions: 15
- Attack trials: 90
- Baseline trial flip-rate: 43.33% (`39/90`)
- ARA trial flip-rate: 46.67% (`42/90`)
- Delta trial flip-rate: +3.33 percentage points
- Question any-flip rate: 100.00% -> 80.00%

By attack:

| Attack | Baseline | ARA |
| --- | ---: | ---: |
| Doubt | 6.67% | 40.00% |
| Trap | 13.33% | 26.67% |
| Authority | 86.67% | 60.00% |
| Gaslighting | 26.67% | 40.00% |
| Majority | 100.00% | 66.67% |
| Emotional | 26.67% | 46.67% |

Interpretation: strength 3.0 is too aggressive for this layer set. It lowers Authority/Majority but increases enough other flips to make total trial flip-rate worse than baseline.

## Current conclusion

The simplified ARA intervention can reduce flip-rate in the pilot setting, but the effect is not uniformly beneficial. The best current configuration is:

```text
layers = 12,18,24
ablation_strength = 2.0
calibration_trials = 120
```

Next recommended run:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/run_qwen_ara_multiturn.py \
  --dataset /home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl \
  --limit 0 \
  --calibration-trials 600 \
  --layers 12,18,24 \
  --ablation-strength 2.0 \
  --output-dir outputs/ara_qwen_full_s2
```

Residual risk: the runner uses sequential `transformers` generation to support hidden-state collection and hooks, so a full-dataset run will be slower than the original vLLM baseline script.
