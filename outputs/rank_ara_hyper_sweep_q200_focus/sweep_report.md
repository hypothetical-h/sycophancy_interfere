# Rank-k ARA Hyperparameter Sweep

Selection rule: sort by `net_gain`, then `mean_delta_margin`, then fewer `correct_to_wrong`, then `valid_rate`.

Configuration:

- model: `/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507`
- dataset: `/home/hanruoshui/sycophancy-eval/datasets/are_you_sure.jsonl`
- limit: `650`
- target_correct: `200`
- source_questions_scanned: `550`
- seed: `42`
- layer_groups: `[[33], [35], [31, 32, 33], [33, 34, 35]]`
- alphas: `[0.25, 0.5, 1.0]`
- ranks: `[1, 2, 4]`
- token_windows: `[4, 16, 32]`
- default_alpha: `1.0`
- default_rank: `4`
- default_token_window: `16`
- calibration_trials: `120`
- max_new_tokens: `32`

- valid_base_questions: `200`
- attack_trials: `1200`

## layers

| rank | layers | alpha | subspace rank | token_window | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | baseline flip | ARA flip |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[35]` | 1.00 | 4 | 16 | -10 | 1 | 11 | 99.58% | 0.0099 | 40.83% | 41.67% |
| 2 | `[33]` | 1.00 | 4 | 16 | -41 | 11 | 52 | 99.58% | -3.1358 | 40.83% | 44.33% |
| 3 | `[31, 32, 33]` | 1.00 | 4 | 16 | -61 | 9 | 70 | 99.33% | -4.3593 | 40.83% | 45.75% |
| 4 | `[33, 34, 35]` | 1.00 | 4 | 16 | -121 | 22 | 143 | 99.75% | -4.2361 | 40.83% | 51.08% |

Best `layers` params: `{'layers': [35], 'alpha': 1.0, 'rank': 4, 'token_window': 16}`

## alpha

| rank | layers | alpha | subspace rank | token_window | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | baseline flip | ARA flip |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[35]` | 0.25 | 4 | 16 | -2 | 0 | 2 | 99.58% | 0.0705 | 40.83% | 41.00% |
| 2 | `[35]` | 0.50 | 4 | 16 | -6 | 0 | 6 | 99.58% | 0.0967 | 40.83% | 41.33% |
| 3 | `[35]` | 1.00 | 4 | 16 | -10 | 1 | 11 | 99.58% | 0.0099 | 40.83% | 41.67% |

Best `alpha` params: `{'layers': [35], 'alpha': 0.25, 'rank': 4, 'token_window': 16}`

## rank

| rank | layers | alpha | subspace rank | token_window | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | baseline flip | ARA flip |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[35]` | 0.25 | 2 | 16 | +2 | 2 | 0 | 99.50% | 0.0150 | 40.83% | 40.58% |
| 2 | `[35]` | 0.25 | 1 | 16 | +0 | 0 | 0 | 99.58% | 0.0043 | 40.83% | 40.83% |
| 3 | `[35]` | 0.25 | 4 | 16 | -2 | 0 | 2 | 99.58% | 0.0705 | 40.83% | 41.00% |

Best `rank` params: `{'layers': [35], 'alpha': 0.25, 'rank': 2, 'token_window': 16}`

## token_window

| rank | layers | alpha | subspace rank | token_window | net gain | wrong->correct | correct->wrong | valid rate | mean Δmargin | baseline flip | ARA flip |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[35]` | 0.25 | 2 | 16 | +2 | 2 | 0 | 99.50% | 0.0150 | 40.83% | 40.58% |
| 2 | `[35]` | 0.25 | 2 | 32 | -1 | 1 | 2 | 99.58% | 0.0637 | 40.83% | 40.92% |
| 3 | `[35]` | 0.25 | 2 | 4 | -1 | 0 | 1 | 99.58% | 0.0080 | 40.83% | 40.92% |

Best `token_window` params: `{'layers': [35], 'alpha': 0.25, 'rank': 2, 'token_window': 16}`

## Final Choice

`{'layers': [35], 'alpha': 0.25, 'rank': 2, 'token_window': 16}`
