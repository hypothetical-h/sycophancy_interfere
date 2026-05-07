# ARA Transition Summary

Run dir: `outputs/ara_qwen_pilot40_s2`

Definitions:

- `baseline flip`: baseline final answer is wrong.
- `ARA flip`: ARA final answer is wrong.
- `wrong->correct`: baseline was wrong and ARA became correct.
- `correct->wrong`: baseline was correct and ARA became wrong.
- `net gain`: `wrong->correct - correct->wrong`.

## Full Pilot40_s2

| Attack | total | baseline flip | ARA flip | wrong->correct | correct->wrong | same wrong | same correct | net gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT_Attack_1_Doubt | 15 | 1 (6.7%) | 4 (26.7%) | 0 | 3 | 1 | 11 | -3 |
| MT_Attack_2_Trap | 15 | 2 (13.3%) | 4 (26.7%) | 0 | 2 | 2 | 11 | -2 |
| MT_Attack_3_Authority | 15 | 13 (86.7%) | 7 (46.7%) | 6 | 0 | 7 | 2 | +6 |
| MT_Attack_4_Gaslighting | 15 | 4 (26.7%) | 5 (33.3%) | 0 | 1 | 4 | 10 | -1 |
| MT_Attack_5_Majority | 15 | 15 (100.0%) | 9 (60.0%) | 6 | 0 | 9 | 0 | +6 |
| MT_Attack_6_Emotional | 15 | 4 (26.7%) | 4 (26.7%) | 1 | 1 | 3 | 10 | +0 |
| ALL | 90 | 39 (43.3%) | 33 (36.7%) | 13 | 7 | 26 | 44 | +6 |

Interpretation:

- Overall ARA changes `39/90` flips to `33/90` flips.
- Total corrected cases: `13`; total newly broken cases: `7`; net gain: `+6`.
- The useful effect is concentrated in `Authority` and `Majority`.
- The main side effects are in `Doubt`, `Trap`, and `Gaslighting`.

## Recent Question Logit Lens Q3 Subset

Subset source: `outputs/ara_qwen_pilot40_s2/question_logit_lens_q3/question_logit_lens_multitoken.json`

| Attack | total | baseline flip | ARA flip | wrong->correct | correct->wrong | same wrong | same correct | net gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT_Attack_1_Doubt | 3 | 1 (33.3%) | 2 (66.7%) | 0 | 1 | 1 | 1 | -1 |
| MT_Attack_2_Trap | 3 | 1 (33.3%) | 2 (66.7%) | 0 | 1 | 1 | 1 | -1 |
| MT_Attack_3_Authority | 3 | 3 (100.0%) | 2 (66.7%) | 1 | 0 | 2 | 0 | +1 |
| MT_Attack_4_Gaslighting | 3 | 2 (66.7%) | 2 (66.7%) | 0 | 0 | 2 | 1 | +0 |
| MT_Attack_5_Majority | 3 | 3 (100.0%) | 3 (100.0%) | 0 | 0 | 3 | 0 | +0 |
| MT_Attack_6_Emotional | 3 | 2 (66.7%) | 2 (66.7%) | 0 | 0 | 2 | 1 | +0 |
| ALL | 18 | 12 (66.7%) | 13 (72.2%) | 1 | 2 | 11 | 4 | -1 |

Interpretation:

- This subset is not representative of the full pilot gain.
- It contains more ARA side effects than corrections, so it is useful for failure-case inspection.
