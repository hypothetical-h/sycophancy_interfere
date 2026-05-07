# Logit Lens Layer Selection Report

Configuration:

- model: `/home/hanruoshui/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507`
- target_correct: `200`
- source_seen: `552`
- baseline_correct_collected: `200`
- attack_trials: `1200`
- rank: `2`
- token_window: `16`
- selected_layers: `[31, 32, 33, 34, 35]`

Artifacts:

- module1_margin_svg: `figures/module1_layer_margin.svg`
- module2_intervention_margin_svg: `figures/module2_intervention_margin.svg`
- module3_delta_energy_svg: `figures/module3_delta_energy.svg`
- module4_similarity_layer_31_svg: `figures/module4_attack_subspace_similarity_layer_31.svg`
- module4_similarity_layer_32_svg: `figures/module4_attack_subspace_similarity_layer_32.svg`
- module4_similarity_layer_33_svg: `figures/module4_attack_subspace_similarity_layer_33.svg`
- module4_similarity_layer_34_svg: `figures/module4_attack_subspace_similarity_layer_34.svg`
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
| 25 | 0.2377 | 0.2377 | 0.0000 |
| 26 | 0.0103 | 0.0103 | 0.0000 |
| 27 | 0.0879 | 0.0879 | 0.0000 |
| 28 | -0.3275 | -0.3275 | 0.0000 |
| 29 | -0.8682 | -0.8682 | 0.0000 |
| 30 | -0.2732 | -0.2732 | 0.0000 |
| 31 | -0.6363 | -0.6363 | 0.0000 |
| 32 | 0.3895 | 0.4433 | 0.0538 |
| 33 | 1.5367 | 2.4974 | 0.9606 |
| 34 | 1.3291 | 1.8087 | 0.4796 |
| 35 | 5.2209 | 5.1847 | -0.0362 |

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
| 31 | 0.3067 | 0.3549 |
| 32 | 0.2891 | 0.3497 |
| 33 | 0.2754 | 0.3500 |
| 34 | 0.3849 | 0.4904 |
| 35 | 0.3328 | 0.4581 |

Pairwise per-attack subspace similarity is reported as mean singular value of `U_attack_a @ U_attack_b.T`. Higher means more aligned subspaces.

### Layer 31 Attack Subspace Similarity

| attack | Doubt | Trap | Authority | Gaslighting | Majority | Emotional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubt | 1.000 | 0.992 | 0.990 | 0.971 | 0.971 | 0.981 |
| Trap | 0.992 | 1.000 | 0.992 | 0.974 | 0.976 | 0.983 |
| Authority | 0.990 | 0.992 | 1.000 | 0.970 | 0.976 | 0.985 |
| Gaslighting | 0.971 | 0.974 | 0.970 | 1.000 | 0.972 | 0.979 |
| Majority | 0.971 | 0.976 | 0.976 | 0.972 | 1.000 | 0.975 |
| Emotional | 0.981 | 0.983 | 0.985 | 0.979 | 0.975 | 1.000 |

Principal angle spectrum by attack pair:

| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |
| --- | --- | --- | ---: | ---: |
| Doubt vs Trap | `[0.993, 0.991]` | `[6.8, 7.6]` | 0.992 | 7.6 |
| Doubt vs Authority | `[0.990, 0.989]` | `[7.9, 8.7]` | 0.990 | 8.7 |
| Doubt vs Gaslighting | `[0.975, 0.967]` | `[12.8, 14.8]` | 0.971 | 14.8 |
| Doubt vs Majority | `[0.980, 0.963]` | `[11.6, 15.7]` | 0.971 | 15.7 |
| Doubt vs Emotional | `[0.982, 0.979]` | `[10.9, 11.7]` | 0.981 | 11.7 |
| Trap vs Authority | `[0.994, 0.990]` | `[6.4, 8.0]` | 0.992 | 8.0 |
| Trap vs Gaslighting | `[0.979, 0.968]` | `[11.8, 14.5]` | 0.974 | 14.5 |
| Trap vs Majority | `[0.985, 0.967]` | `[9.8, 14.7]` | 0.976 | 14.7 |
| Trap vs Emotional | `[0.986, 0.981]` | `[9.8, 11.3]` | 0.983 | 11.3 |
| Authority vs Gaslighting | `[0.976, 0.964]` | `[12.4, 15.4]` | 0.970 | 15.4 |
| Authority vs Majority | `[0.990, 0.962]` | `[8.2, 15.8]` | 0.976 | 15.8 |
| Authority vs Emotional | `[0.985, 0.984]` | `[9.8, 10.2]` | 0.985 | 10.2 |
| Gaslighting vs Majority | `[0.977, 0.967]` | `[12.2, 14.8]` | 0.972 | 14.8 |
| Gaslighting vs Emotional | `[0.987, 0.971]` | `[9.3, 13.7]` | 0.979 | 13.7 |
| Majority vs Emotional | `[0.984, 0.967]` | `[10.3, 14.8]` | 0.975 | 14.8 |

### Layer 32 Attack Subspace Similarity

| attack | Doubt | Trap | Authority | Gaslighting | Majority | Emotional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubt | 1.000 | 0.992 | 0.989 | 0.970 | 0.968 | 0.976 |
| Trap | 0.992 | 1.000 | 0.992 | 0.972 | 0.973 | 0.980 |
| Authority | 0.989 | 0.992 | 1.000 | 0.969 | 0.971 | 0.983 |
| Gaslighting | 0.970 | 0.972 | 0.969 | 1.000 | 0.969 | 0.977 |
| Majority | 0.968 | 0.973 | 0.971 | 0.969 | 1.000 | 0.966 |
| Emotional | 0.976 | 0.980 | 0.983 | 0.977 | 0.966 | 1.000 |

Principal angle spectrum by attack pair:

| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |
| --- | --- | --- | ---: | ---: |
| Doubt vs Trap | `[0.992, 0.992]` | `[7.2, 7.5]` | 0.992 | 7.5 |
| Doubt vs Authority | `[0.989, 0.988]` | `[8.5, 8.8]` | 0.989 | 8.8 |
| Doubt vs Gaslighting | `[0.975, 0.966]` | `[12.9, 15.0]` | 0.970 | 15.0 |
| Doubt vs Majority | `[0.981, 0.956]` | `[11.3, 17.1]` | 0.968 | 17.1 |
| Doubt vs Emotional | `[0.980, 0.971]` | `[11.4, 13.7]` | 0.976 | 13.7 |
| Trap vs Authority | `[0.994, 0.991]` | `[6.5, 7.8]` | 0.992 | 7.8 |
| Trap vs Gaslighting | `[0.977, 0.967]` | `[12.3, 14.9]` | 0.972 | 14.9 |
| Trap vs Majority | `[0.986, 0.959]` | `[9.5, 16.5]` | 0.973 | 16.5 |
| Trap vs Emotional | `[0.983, 0.977]` | `[10.7, 12.2]` | 0.980 | 12.2 |
| Authority vs Gaslighting | `[0.975, 0.964]` | `[12.8, 15.5]` | 0.969 | 15.5 |
| Authority vs Majority | `[0.990, 0.952]` | `[8.0, 17.8]` | 0.971 | 17.8 |
| Authority vs Emotional | `[0.984, 0.982]` | `[10.3, 10.9]` | 0.983 | 10.9 |
| Gaslighting vs Majority | `[0.977, 0.961]` | `[12.3, 16.1]` | 0.969 | 16.1 |
| Gaslighting vs Emotional | `[0.987, 0.967]` | `[9.3, 14.8]` | 0.977 | 14.8 |
| Majority vs Emotional | `[0.983, 0.948]` | `[10.5, 18.6]` | 0.966 | 18.6 |

### Layer 33 Attack Subspace Similarity

| attack | Doubt | Trap | Authority | Gaslighting | Majority | Emotional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubt | 1.000 | 0.989 | 0.943 | 0.969 | 0.966 | 0.901 |
| Trap | 0.989 | 1.000 | 0.966 | 0.965 | 0.955 | 0.931 |
| Authority | 0.943 | 0.966 | 1.000 | 0.913 | 0.887 | 0.978 |
| Gaslighting | 0.969 | 0.965 | 0.913 | 1.000 | 0.966 | 0.889 |
| Majority | 0.966 | 0.955 | 0.887 | 0.966 | 1.000 | 0.843 |
| Emotional | 0.901 | 0.931 | 0.978 | 0.889 | 0.843 | 1.000 |

Principal angle spectrum by attack pair:

| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |
| --- | --- | --- | ---: | ---: |
| Doubt vs Trap | `[0.992, 0.985]` | `[7.2, 9.9]` | 0.989 | 9.9 |
| Doubt vs Authority | `[0.988, 0.897]` | `[8.8, 26.3]` | 0.943 | 26.3 |
| Doubt vs Gaslighting | `[0.973, 0.965]` | `[13.3, 15.2]` | 0.969 | 15.2 |
| Doubt vs Majority | `[0.979, 0.953]` | `[11.7, 17.7]` | 0.966 | 17.7 |
| Doubt vs Emotional | `[0.982, 0.821]` | `[10.9, 34.8]` | 0.901 | 34.8 |
| Trap vs Authority | `[0.992, 0.940]` | `[7.1, 20.0]` | 0.966 | 20.0 |
| Trap vs Gaslighting | `[0.977, 0.953]` | `[12.4, 17.7]` | 0.965 | 17.7 |
| Trap vs Majority | `[0.985, 0.926]` | `[10.0, 22.3]` | 0.955 | 22.3 |
| Trap vs Emotional | `[0.982, 0.880]` | `[11.0, 28.3]` | 0.931 | 28.3 |
| Authority vs Gaslighting | `[0.976, 0.851]` | `[12.6, 31.7]` | 0.913 | 31.7 |
| Authority vs Majority | `[0.990, 0.783]` | `[8.1, 38.4]` | 0.887 | 38.4 |
| Authority vs Emotional | `[0.984, 0.973]` | `[10.4, 13.4]` | 0.978 | 13.4 |
| Gaslighting vs Majority | `[0.976, 0.956]` | `[12.5, 17.1]` | 0.966 | 17.1 |
| Gaslighting vs Emotional | `[0.987, 0.792]` | `[9.4, 37.7]` | 0.889 | 37.7 |
| Majority vs Emotional | `[0.982, 0.705]` | `[10.8, 45.2]` | 0.843 | 45.2 |

### Layer 34 Attack Subspace Similarity

| attack | Doubt | Trap | Authority | Gaslighting | Majority | Emotional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubt | 1.000 | 0.994 | 0.992 | 0.979 | 0.980 | 0.985 |
| Trap | 0.994 | 1.000 | 0.993 | 0.979 | 0.982 | 0.984 |
| Authority | 0.992 | 0.993 | 1.000 | 0.981 | 0.985 | 0.988 |
| Gaslighting | 0.979 | 0.979 | 0.981 | 1.000 | 0.979 | 0.985 |
| Majority | 0.980 | 0.982 | 0.985 | 0.979 | 1.000 | 0.985 |
| Emotional | 0.985 | 0.984 | 0.988 | 0.985 | 0.985 | 1.000 |

Principal angle spectrum by attack pair:

| attack pair | singular values cos(theta) | principal angles deg | mean cos | max angle |
| --- | --- | --- | ---: | ---: |
| Doubt vs Trap | `[0.995, 0.993]` | `[5.8, 6.9]` | 0.994 | 6.9 |
| Doubt vs Authority | `[0.994, 0.990]` | `[6.2, 8.1]` | 0.992 | 8.1 |
| Doubt vs Gaslighting | `[0.987, 0.970]` | `[9.1, 14.1]` | 0.979 | 14.1 |
| Doubt vs Majority | `[0.989, 0.972]` | `[8.4, 13.7]` | 0.980 | 13.7 |
| Doubt vs Emotional | `[0.991, 0.978]` | `[7.6, 12.0]` | 0.985 | 12.0 |
| Trap vs Authority | `[0.995, 0.991]` | `[5.9, 7.6]` | 0.993 | 7.6 |
| Trap vs Gaslighting | `[0.986, 0.972]` | `[9.5, 13.6]` | 0.979 | 13.6 |
| Trap vs Majority | `[0.990, 0.974]` | `[8.0, 13.1]` | 0.982 | 13.1 |
| Trap vs Emotional | `[0.989, 0.979]` | `[8.6, 11.7]` | 0.984 | 11.7 |
| Authority vs Gaslighting | `[0.988, 0.973]` | `[8.9, 13.3]` | 0.981 | 13.3 |
| Authority vs Majority | `[0.995, 0.975]` | `[6.0, 12.9]` | 0.985 | 12.9 |
| Authority vs Emotional | `[0.993, 0.984]` | `[6.9, 10.3]` | 0.988 | 10.3 |
| Gaslighting vs Majority | `[0.987, 0.972]` | `[9.3, 13.6]` | 0.979 | 13.6 |
| Gaslighting vs Emotional | `[0.992, 0.978]` | `[7.2, 12.2]` | 0.985 | 12.2 |
| Majority vs Emotional | `[0.991, 0.979]` | `[7.8, 11.9]` | 0.985 | 11.9 |

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
