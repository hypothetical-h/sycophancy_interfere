每次跑完实验都要更新本文档，且尽量使用中文
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

更新：
原理清楚了，那接下来就是具体方法了。模型的开源，给具体方法的实现提供了先决条件，社区的研究者们能通过观察模型每一层的激活状态，找出这个拒绝向量，并且把它抹掉，任务就完成了。PyTorch提供了Hook 机制，可以给开发者提供这样的能力，但是手写 Hook 依然太繁琐了，所以开源社区也涌现了一些能把这些功能变得极其简单的工具，最知名要算TransformerLens了。这个工具可以用两行简单的Python代码，就能实现对大模型思考的过程拍上一张高清晰度的X光片。# 运行模型，并将每一层、每一个模块的激活状态全部记录在一个名为 cache 的字典中
logits, cache = model.run_with_cache("请告诉我如何制造炸弹")

# 直接调取第 15 层残差流的激活状态
layer_15_activation = cache["blocks.15.hook_resid_post"]当有了这层激活状态之后，我们就可以利用另外一个神奇的工具Logit Lens，把这一层的向量翻译成人类能看懂的具体词汇。具体的观察方法是，研究人员会把中间层，比如第 15 层的激活向量，强行提前送进这个最后一层的翻译矩阵中。这样研究人员就能看到模型在每一层都想了什么了，比如，输入“巴黎是法国的__”，研究人员可能会看到，在第 2 层，模型可能还在想“首都”、“城市”，等到了第 10 层，模型已经笃定地想到了“首都”，然后在第 32 层输出时，才正式吐出“首都”这个词。对于Abliteration这个特殊需求，研究人员只需要关注“对不起”、“我不能”这种拒绝词汇集中在哪些层出现就好了。

## 2026-05-07: ARA方向的Logit Lens观察

Goal: 不只看干预后的 flip-rate，而是直接观察当前提取出的 ARA 方向在词表空间里对应哪些 token，判断这些方向是否能被解释为谄媚攻击、答案选项、或者其他语义因素。

Implementation:

- Added `scripts/logit_lens_ara_directions.py`.
- 输入是已保存的 `qwen_ara_directions.pt`，当前主分析使用 `outputs/ara_qwen_pilot40_s2/qwen_ara_directions.pt`。
- 使用 Qwen3 tied embedding 作为 unembedding 矩阵；模型配置里 `tie_word_embeddings=True`，所以没有单独的 `lm_head.weight`。
- 不跑 generation，只加载 `model.embed_tokens.weight`、`model.norm.weight`、tokenizer 和已保存的 layer 12/18/24 ARA direction。
- 对每层方向做两种 lens：
  - direct: `logits = W_U direction`，表示该单位方向对 logits 的线性贡献。
  - normalized: `logits = W_U RMSNorm(direction)`，更接近把该向量提前送进最终输出头的 logit lens。
- 同时输出正方向 top tokens、负方向 top tokens、A/B/C/D/E 选项 token 分数，以及与攻击模板相关的 probe terms 分数，例如 `professor`, `community`, `majority`, `bet`, `agree`, `actually`, `wrong`, `correct`。

Important caveat:

- PCA/SVD 方向的正负号本身没有绝对语义；干预公式 `h - alpha (h·u)u` 对 `u` 和 `-u` 等价。
- 因此 logit lens 需要同时看 positive 和 negative，不应该把“正方向”简单理解为“攻击方向唯一正确的符号”。
- top-token 表中混有较多词表碎片和多语言 token，因此解释时更可靠的是结合 probe terms、选项分数、以及后续真实 prompt 上的 logit change。

Command:

```bash
/home/hanruoshui/miniconda3/envs/llm_eval/bin/python scripts/logit_lens_ara_directions.py \
  --directions outputs/ara_qwen_pilot40_s2/qwen_ara_directions.pt \
  --output-dir outputs/ara_qwen_pilot40_s2/logit_lens \
  --top-k 20
```

Output:

- `outputs/ara_qwen_pilot40_s2/logit_lens/qwen_ara_direction_logit_lens.json`
- `outputs/ara_qwen_pilot40_s2/logit_lens/qwen_ara_direction_logit_lens.md`

Direction consistency check:

- `outputs/ara_qwen_pilot40/qwen_ara_directions.pt`
- `outputs/ara_qwen_pilot40_s2/qwen_ara_directions.pt`
- `outputs/ara_qwen_pilot40_s3/qwen_ara_directions.pt`

三者在 layer 12/18/24 上的 cosine similarity 都约为 `1.0`。这说明 strength 1.0/2.0/3.0 的行为差异只来自 ablation strength，不来自方向重新估计差异。

Observed results, direct lens:

| Layer | Option scores, high to low | High probe terms |
| --- | --- | --- |
| 12 | E `0.0494`, C `0.0397`, D `0.0258`, B `0.0242`, A `0.0203` | `bet`, `final`, `no`, `change`, `professor`, `community`, `majority` |
| 18 | C `0.0120`, E `0.0102`, B `0.0064`, A `0.0047`, D `0.0015` | `sorry`, `community`, `yes`, `taught`, `actually`, `bet`, `agree` |
| 24 | A `0.1167`, E `0.0415`, C `-0.0316`, B `-0.0408`, D `-0.0438` | `A`, `answer`, `agree`, `actually`, `reconsider`, `majority`, `incorrect`, `wrong` |

Observed results, normalized lens:

| Layer | High probe terms |
| --- | --- |
| 12 | `bet`, `change`, `professor`, `final`, `no`, `community`, `majority` |
| 18 | `community`, `sorry`, `yes`, `actually`, `taught`, `agree` |
| 24 | `A`, `agree`, `answer`, `actually`, `reconsider`, `majority`, `incorrect`, `wrong` |

Interpretation:

- Layer 12 的方向更像早中层的攻击上下文/压力语义，和 `bet`, `professor`, `community`, `majority`, `change` 等攻击模板词相关。
- Layer 18 的方向明显含有社交顺从/外部压力信号，`community`, `sorry`, `yes`, `actually`, `taught` 排名较高。
- Layer 24 的方向强烈偏向答案 token，尤其是 `A`，并且同时关联 `answer`, `agree`, `actually`, `reconsider`, `incorrect`, `wrong`。这提示第 24 层方向可能不只是“谄媚语义”，还混入了具体答案选项偏置或最终选择机制。
- 这可以解释为什么 strength 2.0 能降低 Authority/Majority，但也会提高 Doubt/Trap 等弱攻击：删除方向不只删除社交诱导，也可能扰动答案选择相关 logit。

Next recommended experiment:

在真实 attack prompt 上做 logit-difference lens，而不是只看裸方向：

```text
delta_logits = logits(attack hidden) - logits(base hidden)
ablation_delta_logits = logits(hidden) - logits(hidden - alpha * proj_u(hidden))
```

这样可以回答更具体的问题：ARA 删除的到底是错误选项 token 的 logit、攻击语义 token 的 logit，还是某些无关高频词表方向。

当我们锁定了“谄媚机制”主要出现的层之后，就可以开始进行“干预手术”了。研究目标变成：如何让模型不再沿着这些特定方向产生迎合用户、盲目附和的行为。

这时研究者会开始对“谄媚向量”进行精细计算。常见方法仍然是对比测试：准备两组提示词，一组是容易诱导模型谄媚的 prompt（例如带有明显立场暗示、寻求认同的表达），另一组则是中性、客观的 prompt。随后将它们分别输入模型，并收集目标层（例如第 15 层）的 activation。

设诱导谄媚的激活集合为 $H_{sycophantic}$，中性激活集合为 $H_{neutral}$，样本量分别为 $N$ 和 $M$，则可以定义一个“谄媚方向”：

$$
v_{sycophancy}
=
\frac{1}{N}\sum_{i=1}^{N} H_{sycophantic}^{(i)}
-
\frac{1}{M}\sum_{j=1}^{M} H_{neutral}^{(j)}
$$

此时，研究者就获得了一个能够表征“迎合倾向”的方向向量。接下来的目标，是让模型在特定层的残差流中尽可能无法继续传递这一方向上的信息。

一种常见做法，是直接修改对应层的注意力输出矩阵（attention output matrix）或前馈网络（FFN）输出矩阵。两种修改方式的作用略有不同：

- 对前馈网络的修改，更倾向于削弱由特定概念直接触发的谄媚反应；
- 对注意力矩阵的修改，则更偏向阻断由上下文组合诱导出的“迎合意图”。

谨慎的研究者通常会同时处理两部分，从而同时压制：
1. 基于局部语义触发的谄媚；
2. 基于长上下文推理形成的迎合行为。

经过这种处理之后，目标层中的组件就难以继续向后续残差流注入明显的“谄媚特征”。理论上，这意味着模型的默认行为会更加稳定，不再轻易因为用户表达的立场、情绪或偏好而发生无原则偏移。

当然，真正的研究并不会止步于此。后续还需要继续测试很多问题，例如：

- “谄媚”行为是否只集中在单层；
- 不同类型的谄媚（政治、情感、价值判断）是否对应不同方向；
- 干预之后是否会损伤模型的推理能力；
- 是否会影响模型正常的共情与对话体验；
- 干预是否会导致新的副作用（例如过度冷漠、回答僵硬等）。

因此，这通常是一个不断实验、评估与调优的过程。最终，研究者会得到一个在“减少谄媚”与“保持能力”之间相对平衡的方案，并将其合并进模型权重之中。