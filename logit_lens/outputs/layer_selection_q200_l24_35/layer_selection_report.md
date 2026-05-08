# Logit Lens Layer Selection Report

Configuration:

- model: `/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507`
- target_correct: `200`
- source_seen: `552`
- baseline_correct_collected: `200`
- attack_trials: `1200`
- rank: `2`
- token_window: `16`
- selected_layers: `[24, 35]`

Artifacts:

- module1_margin_svg: `figures/module1_layer_margin.svg`
- module2_intervention_margin_svg: `figures/module2_intervention_margin.svg`
- module3_delta_energy_svg: `figures/module3_delta_energy.svg`
- module4_similarity_layer_24_svg: `figures/module4_attack_subspace_similarity_layer_24.svg`
- module4_similarity_layer_35_svg: `figures/module4_attack_subspace_similarity_layer_35.svg`
- module4_layer_attack_summary_svg: `figures/module4_layer_attack_cross_layer_similarity.svg`
- module4_layer_similarity_doubt_svg: `figures/module4_layer_similarity_attack_doubt.svg`
- module4_layer_similarity_trap_svg: `figures/module4_layer_similarity_attack_trap.svg`
- module4_layer_similarity_authority_svg: `figures/module4_layer_similarity_attack_authority.svg`
- module4_layer_similarity_gaslighting_svg: `figures/module4_layer_similarity_attack_gaslighting.svg`
- module4_layer_similarity_majority_svg: `figures/module4_layer_similarity_attack_majority.svg`
- module4_layer_similarity_emotional_svg: `figures/module4_layer_similarity_attack_emotional.svg`

## Module 1: Layer-wise Logit Margin

Margin is `logit(correct) - logit(user_wrong)`. Positive means the layer favors the correct answer over the user's wrong answer.

| Layer | base margin | attack margin | attack-base |
| ---: | ---: | ---: | ---: |
| 0 | 0.0829 | 0.0934 | 0.0105 |
| 1 | 0.5735 | 0.5515 | -0.0220 |
| 2 | 0.5213 | 0.5101 | -0.0112 |
| 3 | 0.6417 | 0.5928 | -0.0489 |
| 4 | 0.1433 | 0.1672 | 0.0239 |
| 5 | 0.0602 | 0.0945 | 0.0343 |
| 6 | -0.0209 | -0.0218 | -0.0009 |
| 7 | -0.0451 | 0.0823 | 0.1274 |
| 8 | 0.2777 | 0.1606 | -0.1171 |
| 9 | 0.0351 | -0.0375 | -0.0726 |
| 10 | -0.0032 | 0.1387 | 0.1419 |
| 11 | -0.3069 | -0.3093 | -0.0024 |
| 12 | -0.3835 | -0.4663 | -0.0828 |
| 13 | -0.1689 | -0.5562 | -0.3873 |
| 14 | -0.4434 | -0.7406 | -0.2972 |
| 15 | -0.7069 | -1.0470 | -0.3401 |
| 16 | -0.2372 | -0.1588 | 0.0784 |
| 17 | -0.0816 | 0.1148 | 0.1964 |
| 18 | -0.4410 | -0.2770 | 0.1640 |
| 19 | -0.1488 | -0.1554 | -0.0066 |
| 20 | -0.1211 | -0.3440 | -0.2229 |
| 21 | -0.2554 | -0.3032 | -0.0478 |
| 22 | -0.0050 | -0.5578 | -0.5528 |
| 23 | 0.6719 | -0.1341 | -0.8059 |
| 24 | 5.6751 | 0.5800 | -5.0952 |
| 25 | 4.1216 | 0.2377 | -3.8839 |
| 26 | 4.0479 | 0.0103 | -4.0376 |
| 27 | 3.8117 | 0.0879 | -3.7238 |
| 28 | 3.1659 | -0.3275 | -3.4933 |
| 29 | 9.9719 | -0.8682 | -10.8401 |
| 30 | 9.1091 | -0.2732 | -9.3822 |
| 31 | 12.7394 | -0.6363 | -13.3757 |
| 32 | 12.8342 | 0.3895 | -12.4447 |
| 33 | 13.3520 | 1.5367 | -11.8153 |
| 34 | 10.0477 | 1.3291 | -8.7186 |
| 35 | 20.0656 | 5.2209 | -14.8447 |

## Module 2: Intervention Margin

| Layer | baseline attack margin | intervened attack margin | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.0934 | 0.0934 | 0.0000 |
| 1 | 0.5515 | 0.5515 | 0.0000 |
| 2 | 0.5101 | 0.5101 | 0.0000 |
| 3 | 0.5928 | 0.5928 | 0.0000 |
| 4 | 0.1672 | 0.1672 | 0.0000 |
| 5 | 0.0945 | 0.0945 | 0.0000 |
| 6 | -0.0218 | -0.0218 | 0.0000 |
| 7 | 0.0823 | 0.0823 | 0.0000 |
| 8 | 0.1606 | 0.1606 | 0.0000 |
| 9 | -0.0375 | -0.0375 | 0.0000 |
| 10 | 0.1387 | 0.1387 | 0.0000 |
| 11 | -0.3093 | -0.3093 | 0.0000 |
| 12 | -0.4663 | -0.4663 | 0.0000 |
| 13 | -0.5562 | -0.5562 | 0.0000 |
| 14 | -0.7406 | -0.7406 | 0.0000 |
| 15 | -1.0470 | -1.0470 | 0.0000 |
| 16 | -0.1588 | -0.1588 | 0.0000 |
| 17 | 0.1148 | 0.1148 | 0.0000 |
| 18 | -0.2770 | -0.2770 | 0.0000 |
| 19 | -0.1554 | -0.1554 | 0.0000 |
| 20 | -0.3440 | -0.3440 | 0.0000 |
| 21 | -0.3032 | -0.3032 | 0.0000 |
| 22 | -0.5578 | -0.5578 | 0.0000 |
| 23 | -0.1341 | -0.1341 | 0.0000 |
| 24 | 0.5800 | 0.5800 | 0.0000 |
| 25 | 0.2377 | 0.2923 | 0.0546 |
| 26 | 0.0103 | 0.0518 | 0.0415 |
| 27 | 0.0879 | 0.1283 | 0.0404 |
| 28 | -0.3275 | -0.3221 | 0.0054 |
| 29 | -0.8682 | -0.9721 | -0.1039 |
| 30 | -0.2732 | -0.3932 | -0.1200 |
| 31 | -0.6363 | -0.7751 | -0.1388 |
| 32 | 0.3895 | 0.3786 | -0.0108 |
| 33 | 1.5367 | 1.5193 | -0.0174 |
| 34 | 1.3291 | 1.3418 | 0.0127 |
| 35 | 5.2209 | 5.4724 | 0.2516 |

## Module 3: Layer-wise Delta Energy

| Layer | mean ||delta h|| | mean cosine(base, attack) |
| ---: | ---: | ---: |
| 0 | 0.3655 | 0.9979 |
| 1 | 0.8179 | 0.9955 |
| 2 | 1.1316 | 0.9933 |
| 3 | 1.8039 | 0.9893 |
| 4 | 2.5507 | 0.9891 |
| 5 | 3.1231 | 0.9906 |
| 6 | 4.0271 | 0.9872 |
| 7 | 5.7011 | 0.9787 |
| 8 | 7.7749 | 0.9649 |
| 9 | 7.8871 | 0.9700 |
| 10 | 7.8541 | 0.9679 |
| 11 | 8.8704 | 0.9617 |
| 12 | 9.4800 | 0.9510 |
| 13 | 10.5389 | 0.9429 |
| 14 | 11.7794 | 0.9311 |
| 15 | 15.7400 | 0.8714 |
| 16 | 16.3159 | 0.8727 |
| 17 | 17.2203 | 0.8924 |
| 18 | 19.4622 | 0.8973 |
| 19 | 24.6100 | 0.8516 |
| 20 | 28.3950 | 0.8463 |
| 21 | 33.7486 | 0.8203 |
| 22 | 35.4942 | 0.8179 |
| 23 | 42.7834 | 0.8298 |
| 24 | 50.0933 | 0.8576 |
| 25 | 55.7996 | 0.8788 |
| 26 | 61.7439 | 0.8892 |
| 27 | 68.2335 | 0.8911 |
| 28 | 74.9972 | 0.8958 |
| 29 | 84.0539 | 0.8961 |
| 30 | 97.1243 | 0.9028 |
| 31 | 108.6549 | 0.9072 |
| 32 | 121.7102 | 0.9163 |
| 33 | 139.1978 | 0.9291 |
| 34 | 166.4872 | 0.9449 |
| 35 | 94.5299 | 0.8172 |

## Module 4: Subspace Quality

| Layer | explained energy | projection norm ratio |
| ---: | ---: | ---: |
| 24 | 0.3103 | 0.4363 |
| 35 | 0.3328 | 0.4581 |

Pairwise per-attack subspace similarity is reported as mean singular value of `U_attack_a @ U_attack_b.T`. Higher means more aligned subspaces.

### Layer 24 Attack Subspace Similarity

| attack | Doubt | Trap | Authority | Gaslighting | Majority | Emotional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubt | 1.000 | 0.987 | 0.980 | 0.972 | 0.976 | 0.973 |
| Trap | 0.987 | 1.000 | 0.994 | 0.967 | 0.987 | 0.987 |
| Authority | 0.980 | 0.994 | 1.000 | 0.964 | 0.990 | 0.989 |
| Gaslighting | 0.972 | 0.967 | 0.964 | 1.000 | 0.970 | 0.964 |
| Majority | 0.976 | 0.987 | 0.990 | 0.970 | 1.000 | 0.987 |
| Emotional | 0.973 | 0.987 | 0.989 | 0.964 | 0.987 | 1.000 |

Principal angle spectrum by attack pair:

| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |
| --- | --- | --- | ---: | ---: |
| Doubt vs Trap | `[0.996, 0.977]` | `[5.1, 12.3]` | 0.987 | 12.3 |
| Doubt vs Authority | `[0.992, 0.968]` | `[7.1, 14.4]` | 0.980 | 14.4 |
| Doubt vs Gaslighting | `[0.981, 0.964]` | `[11.3, 15.4]` | 0.972 | 15.4 |
| Doubt vs Majority | `[0.986, 0.965]` | `[9.5, 15.3]` | 0.976 | 15.3 |
| Doubt vs Emotional | `[0.987, 0.959]` | `[9.1, 16.6]` | 0.973 | 16.6 |
| Trap vs Authority | `[0.995, 0.994]` | `[5.9, 6.1]` | 0.994 | 6.1 |
| Trap vs Gaslighting | `[0.982, 0.952]` | `[10.9, 17.9]` | 0.967 | 17.9 |
| Trap vs Majority | `[0.988, 0.986]` | `[9.0, 9.4]` | 0.987 | 9.4 |
| Trap vs Emotional | `[0.989, 0.985]` | `[8.5, 10.1]` | 0.987 | 10.1 |
| Authority vs Gaslighting | `[0.982, 0.945]` | `[10.8, 19.1]` | 0.964 | 19.1 |
| Authority vs Majority | `[0.992, 0.988]` | `[7.5, 9.0]` | 0.990 | 9.0 |
| Authority vs Emotional | `[0.991, 0.988]` | `[7.9, 8.8]` | 0.989 | 8.8 |
| Gaslighting vs Majority | `[0.983, 0.956]` | `[10.5, 17.1]` | 0.970 | 17.1 |
| Gaslighting vs Emotional | `[0.985, 0.944]` | `[9.9, 19.3]` | 0.964 | 19.3 |
| Majority vs Emotional | `[0.990, 0.984]` | `[8.2, 10.2]` | 0.987 | 10.2 |

### Layer 35 Attack Subspace Similarity

| attack | Doubt | Trap | Authority | Gaslighting | Majority | Emotional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubt | 1.000 | 0.985 | 0.990 | 0.982 | 0.978 | 0.987 |
| Trap | 0.985 | 1.000 | 0.973 | 0.975 | 0.987 | 0.969 |
| Authority | 0.990 | 0.973 | 1.000 | 0.981 | 0.972 | 0.990 |
| Gaslighting | 0.982 | 0.975 | 0.981 | 1.000 | 0.977 | 0.985 |
| Majority | 0.978 | 0.987 | 0.972 | 0.977 | 1.000 | 0.971 |
| Emotional | 0.987 | 0.969 | 0.990 | 0.985 | 0.971 | 1.000 |

Principal angle spectrum by attack pair:

| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |
| --- | --- | --- | ---: | ---: |
| Doubt vs Trap | `[0.994, 0.975]` | `[6.1, 12.8]` | 0.985 | 12.8 |
| Doubt vs Authority | `[0.994, 0.985]` | `[6.2, 10.0]` | 0.990 | 10.0 |
| Doubt vs Gaslighting | `[0.986, 0.978]` | `[9.5, 11.9]` | 0.982 | 11.9 |
| Doubt vs Majority | `[0.983, 0.972]` | `[10.6, 13.5]` | 0.978 | 13.5 |
| Doubt vs Emotional | `[0.991, 0.984]` | `[7.7, 10.4]` | 0.987 | 10.4 |
| Trap vs Authority | `[0.994, 0.952]` | `[6.3, 17.9]` | 0.973 | 17.9 |
| Trap vs Gaslighting | `[0.983, 0.967]` | `[10.5, 14.7]` | 0.975 | 14.7 |
| Trap vs Majority | `[0.990, 0.985]` | `[8.1, 10.0]` | 0.987 | 10.0 |
| Trap vs Emotional | `[0.988, 0.950]` | `[8.9, 18.3]` | 0.969 | 18.3 |
| Authority vs Gaslighting | `[0.987, 0.975]` | `[9.2, 12.9]` | 0.981 | 12.9 |
| Authority vs Majority | `[0.984, 0.960]` | `[10.2, 16.2]` | 0.972 | 16.2 |
| Authority vs Emotional | `[0.991, 0.988]` | `[7.5, 8.7]` | 0.990 | 8.7 |
| Gaslighting vs Majority | `[0.983, 0.971]` | `[10.6, 13.8]` | 0.977 | 13.8 |
| Gaslighting vs Emotional | `[0.987, 0.983]` | `[9.1, 10.7]` | 0.985 | 10.7 |
| Majority vs Emotional | `[0.982, 0.961]` | `[11.0, 16.1]` | 0.971 | 16.1 |
