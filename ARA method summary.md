知乎原文链接：https://www.zhihu.com/question/2023204569811828949/answer/2024575866349393237
理论支持：“线性表征假说”（认为大模型会将高级的抽象概念——以sycophancy为例，编码为高维空间中的单一线性方向）
如果我们能找到这个方向，并在权重中把它抹掉，就不会触发相应行为
具体方法：
1. Hook：运行模型的过程中，可以将每一层，每一个模块的激活状态记录在一个名为cache的字典里，可以直接调取任意层残差流的激活状态。
2. Logit Lens：把中间层的激活向量强行送入最后一层的翻译矩阵；本实验中具体：看翻译结果和A，B，C，D的相似度。用来锁定谄媚机制主要出现的层。
3. Arbitrary-Rank Ablation 尤其对复杂逻辑推理/思考的模型有效（CoT？）
具体方法：
    (1) 不再是只取最后一个token对应的hidden state，而是取所有的（trajectory trace）。把它们拼成一个大矩阵，再做奇异值分解，V就是谄媚的subspace。这个实验现在把token window放成了一个参数，只取最后若干token。
    (2) 构造一个rank-k惩罚矩阵$\delta_W$，用这个矩阵来做多维层面的减法（实际上就是取SVD分解出的V的前 k 个右奇异向量，用它们作为惩罚矩阵的标准正交基）
    从数学上，应该看到哪个向量值突然衰减来确定k的值

# 现阶段实验结论与下一步路线

## 当前最可靠的实验事实

q200 复验中，最稳的 rank-k ARA 配置是：

```text
layers = [35]
alpha = 0.25
rank = 2
token_window = 16
```

行为结果：

```text
baseline flip = 490/1200 = 40.83%
ARA flip      = 40.58%
wrong->correct = 2
correct->wrong = 0
net_gain = +2
```

这说明现在的方法有一个很弱但方向正确的信号：它不是强干预成功，而是低副作用 proof-of-signal。

同时，强干预配置副作用明显：

```text
[33], alpha=1.0, rank=4, token_window=16:
net_gain = -41, correct->wrong = 52

[31,32,33], alpha=1.0, rank=4, token_window=16:
net_gain = -61, correct->wrong = 70

[33,34,35], alpha=1.0, rank=4, token_window=16:
net_gain = -121, correct->wrong = 143
```

因此，不能简单地认为“readout alignment 强的层越多越好”。多层高强度 removal 很容易伤害原本正确的答案机制。

## 机理上已经比较明确的部分

### 1. attack 确实显著改变 late-layer answer margin

Module 1 中：

```text
margin = logit(correct) - logit(user_wrong)
```

关键 late layers：

```text
layer 31: attack-base = -13.3757
layer 32: attack-base = -12.4447
layer 33: attack-base = -11.8153
layer 34: attack-base = -8.7186
layer 35: attack-base = -14.8447
```

这说明攻击不是只改变表面文本，而是在 late layers 明显压低正确答案相对用户错误答案的 readout margin。

### 2. 最优 ARA 配置确实把 layer 35 margin 往正确方向推了一点

Module 2 中，固定 hook layer 35：

```text
baseline attack margin   = 5.2209
intervened attack margin = 5.3490
delta margin = +0.1281
```

这和 q200 behavioral net gain `+2` 一致：方向对，但幅度很小。

### 3. layer 35 的 attack subspace 在不同 attack type 间高度共享

Layer 35 rank-2 shared subspace:

```text
explained_energy = 0.3328
projection_norm_ratio = 0.4581
```

不同 attack type 的 principal angle 很小，例如：

```text
Doubt vs Authority: singular values [0.994, 0.985], angles [6.2, 10.0]
Authority vs Emotional: singular values [0.991, 0.988], angles [7.5, 8.7]
Majority vs Emotional: singular values [0.982, 0.961], angles [11.0, 16.1]
```

所以 shared ARA subspace 有几何依据。

### 4. late layers 不是一个连续同质区间

layer x attack subspace similarity 显示：

```text
31-32: about 0.92
32-33: about 0.90+
33-34: drops to about 0.47-0.71
34-35: drops to about 0.41-0.45
```

解释：

```text
31-33 更像一组中间 late-layer attack mechanism
35 更像最终 readout 前的另一种几何阶段
```

这解释了为什么 `[33,34,35]` 多层 hook 副作用很大。即使每层用的是各自的惩罚矩阵，多个几何上不同的 subspace 串联 removal，也可能同时伤到不同功能方向。

# 下一步：如何获得更明显的干预效果

## 方向 A：做低强度非连续多层 hook，而不是连续 late-layer 强 hook

目前还没有系统尝试 rank-k 非连续多层配置。旧的 `[12,18,24]` 是 rank-1 启发式 pilot，不是现在这套 q200 rank-k sweep。

建议下一轮 q200 focus sweep：

```text
layers:
  [35]
  [33,35]
  [32,35]
  [31,33,35]
  [24,35]

alpha:
  0.05, 0.10, 0.20, 0.25, 0.35

rank:
  1, 2

token_window:
  16
```

理由：

1. `[35]` 已经是最稳单层。
2. 31-33 与 35 几何不连续，所以如果要多层，应跳层、低强度，而不是 `[33,34,35]` 连续强 removal。
3. `alpha=1.0` 已经证明太强；下一步应围绕 `0.05-0.35` 找低副作用区间。
4. `rank=4` 容易引入 correct->wrong；rank 1/2 更合理。

选择标准仍然应优先行为指标：

```text
primary: net_gain = wrong->correct - correct->wrong
secondary: correct->wrong as low as possible
third: valid_rate
diagnostic: mean_delta_margin
```

## 方向 B：把 projection removal 改成 gated / selective removal

现在 hook 是无条件移除：

```text
h' = h - alpha * P h
```

问题是：即使样本本来没有被攻击翻错，hook 也照样移除 subspace，于是容易造成 correct->wrong。

更合理的做法是只在“攻击确实把 hidden state 推向错误答案方向”时干预：

```text
d_logit = W_lm[wrong] - W_lm[correct]
d_syco  = h_attack - h_base
gate = 1[cos(d_syco, d_logit) > tau]
h' = h - alpha * gate * P h
```

实际推理时没有 `h_base`，可以用近似 gate：

```text
gate = 1[logit(wrong) - logit(correct) > threshold]
```

或者连续 gate：

```text
g = sigmoid(beta * (logit(wrong) - logit(correct) - tau))
h' = h - alpha * g * P h
```

预期收益：

```text
减少 correct->wrong
让干预集中在真正危险的样本上
```

这是目前最可能把弱正收益变成更明显收益的方向。

## 方向 C：从“移除 sycophancy subspace”改成“沿答案 readout 修正”

当前 ARA 是 remove attack delta subspace：

```text
h' = h - alpha * P_syco h
```

但 q200 中出现了一个问题：`mean_delta_margin` 有时变好，behavior 却不一定变好；说明单纯 remove subspace 不一定精准推动答案。

可以尝试 readout-aware correction：

```text
d_logit = W_lm[wrong] - W_lm[correct]
remove only component of P_syco that aligns with d_logit
```

例如构造：

```text
u = normalize(P_syco d_logit)
h' = h - alpha * (h · u) u
```

含义：

```text
只移除 sycophancy subspace 中最直接提高 wrong-vs-correct logit 的方向
```

这比完整 rank-k removal 更窄，理论上更少伤害其它能力。

## 方向 D：按 attack type 或 attack family 做 mixture-of-subspaces

虽然 layer 35 的不同 attack type subspace 很一致，但不是完全相同。可以比较：

```text
shared subspace
per-attack subspace
clustered subspace:
  authority-like: Authority, Majority
  pressure-like: Doubt, Trap, Gaslighting, Emotional
```

推理时不一定知道 attack type，但可以用 prompt classifier 或简单关键词先做 routing。

这个方向主要用于解释机制：

```text
如果 shared subspace 和 per-attack subspace 效果差不多，说明 sycophancy 机制更统一。
如果 per-attack 明显更好，说明不同攻击通过不同心理/语义通道进入模型。
```

## 方向 E：把 calibration set 从“所有攻击样本”改成“真正 flip 的攻击样本”

现在估计 subspace 时用的是 first-turn correct 后的所有 attack trials，包括：

```text
attack 后仍回答正确的样本
attack 后翻错的样本
```

但真正的 sycophancy direction 可能主要存在于 baseline attack 会 flip 的样本中。

下一步可以比较三种 calibration：

```text
all attacks
only baseline flipped attacks
only high wrong-logit-margin attacks
```

其中 high wrong-logit-margin 定义为：

```text
logit(user_wrong) - logit(correct) > tau
```

预期：

```text
如果 only-flipped subspace 更有效，说明当前 all-attacks calibration 被大量非翻转样本稀释。
```

# 下一步：如何得到更明确的机理结论

## 1. 做 causal mediation: hidden shift -> wrong readout -> final flip

现在我们有三个量：

```text
d_syco = h_attack - h_base
d_logit = W_wrong - W_correct
final flip
```

下一步应该统计：

```text
alignment = cosine(d_syco, d_logit)
projection = dot(d_syco, normalize(d_logit))
```

然后按样本分组：

```text
baseline attack flips
baseline attack does not flip
ARA wrong->correct
ARA correct->wrong
```

关键问题：

```text
真正 flip 的样本是否有更大的 readout projection？
ARA 修正成功的样本是否显著降低这个 projection？
correct->wrong 的样本是否也被错误降低了某些 correct-supporting direction？
```

如果成立，机理链条会更完整：

```text
attack text
-> late-layer hidden shift
-> alignment with wrong-answer readout
-> final answer flip
-> ARA reduces this component
```

## 2. 做 singular spectrum 和 rank cutoff 的稳定性分析

现在 rank=2 在 q200 最稳，但还需要解释为什么是 2。

建议每层画：

```text
singular value spectrum
cumulative explained energy
behavioral result vs rank
```

重点看 layer 35：

```text
rank 1: net_gain 0, correct->wrong 0
rank 2: net_gain +2, correct->wrong 0
rank 4: net_gain -2, correct->wrong 2
```

这可以形成一个比较清楚的结论：

```text
前两个 singular directions 捕捉可干预信号；
更高 rank 开始包含非谄媚或任务相关成分，导致副作用。
```

## 3. 区分“answer readout 层”和“attack representation 层”

cross-layer similarity 显示：

```text
31-33 是一组
35 是另一组
```

下一步可以把层功能分成：

```text
31-33: attack representation / social pressure encoding
35: answer readout / final decision steering
```

实验设计：

```text
只在 31-33 干预
只在 35 干预
31-33 弱干预 + 35 弱干预
31-33 检测，35 执行干预
```

其中最后一个最有机制意义：

```text
用 31-33 的 hidden shift/readout alignment 作为 detector
如果 detector 判断 attack pressure 强，再在 35 做低强度 correction
```

这比所有层都 hook 更符合现在的层间几何观察。

## 4. 做 answer-token vs free-generation 的分离评估

现在 final answer 是从生成文本里 extract A-E。可能存在两种不同现象：

```text
next-token logits 改善，但最终生成仍受格式/后续 token 影响
最终答案改善，但 next-token margin 没有明显改善
```

下一步应同时记录：

```text
next-token argmax among A-E
generated final answer
margin(correct - wrong)
full option distribution entropy
```

这样可以判断 ARA 影响的是：

```text
答案 readout 本身
还是生成路径/格式/后续 token dynamics
```

# 推荐的下一轮实验优先级

最推荐按这个顺序做：

## Priority 1: q200 low-alpha non-contiguous multilayer sweep

```text
layers = [35], [33,35], [32,35], [31,33,35], [24,35]
alpha = 0.05, 0.10, 0.20, 0.25, 0.35
rank = 1, 2
token_window = 16
```

目标：找比 `[35], 0.25, rank=2` 更强但仍低副作用的组合。

## Priority 2: gated ARA

先做 logit-margin gate：

```text
g = sigmoid(beta * (logit_wrong - logit_correct - tau))
h' = h - alpha * g * P h
```

目标：降低 correct->wrong。

## Priority 3: flipped-only calibration

比较：

```text
all attacks
baseline flipped attacks only
high wrong-margin attacks only
```

目标：避免 subspace 被非翻转样本稀释。

## Priority 4: causal diagnostic report

按样本统计：

```text
alignment/projection vs final flip
alignment/projection vs ARA transition type
```

目标：把“hidden shift 朝错误答案 readout 方向移动”这件事和最终行为建立更明确的因果证据链。
    
