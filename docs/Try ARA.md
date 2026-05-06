如果你想尝试 multi-dimensional / subspace ablation 实验，下面是一个简化的流程框架：

1️⃣ 步骤 1：选择模型

首先，选择你要实验的模型。可以从以下两个模型中选择一个：

Qwen/Qwen3-4B-Instruct-2507


2️⃣ 步骤 2：收集初始数据

通过一个简单的实验进行数据收集：

输入：

are_you_sure.jsonl以及生成的multiturn攻击

输出：

模型的初始回答。

你需要记录模型输出的 每一层激活（可以使用 transformers 的 run_with_cache 方法）：

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

model_name = "Qwen/Qwen3-4B-Thinking-2507"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample input
input_text = "What is the capital of France?"

# Get model output and cache activations
output = model.generate(tokenizer.encode(input_text, return_tensors="pt"))
数据记录：
原始输出 和 激活数据 都要保存。
3️⃣ 步骤 3：进行 CoT 干预（不是CoT模型没有这一步）

通过 CoT 干预对模型做推理过程解释：

Question: What is the capital of France?
Reasoning: The capital of France is Paris because Paris is the largest city in France, located in the north-central part of the country...
Final Answer: Paris

这会帮助你记录模型的思维链，并观察 思维链是否能够减少模型的 flip 率，即是否有助于“坚持原始答案”。

4️⃣ 步骤 4：激活分析与子空间移除
目标：

分析模型内部的激活，判断是否有“谄媚”特征的方向或维度（例如，是否模型会根据用户的输入进行调整）。可以使用 PCA 或 SVD 之类的方式来提取这些子空间。

实验示范：
import torch
import numpy as np
from sklearn.decomposition import PCA

# 获取模型某一层的激活
layer_activations = model.layer[0].output

# 进行PCA分析（降维）
pca = PCA(n_components=5)
reduced_activations = pca.fit_transform(layer_activations)

# 对应子空间移除
ablation_result = remove_subspace(reduced_activations)
5️⃣ 步骤 5：子空间移除与反向推理

通过 subspace projection 删除模型的子空间（例如：根据 SVD 或 PCA 得到的特征）。再根据你删除的子空间去分析输出行为。

def remove_subspace(activations):
    # 执行子空间移除操作
    # 这可以是直接删除主成分，或者投影到子空间外
    pass
6️⃣ 步骤 6：结果分析

最后，比较模型的性能和行为：

模型是否在经过子空间干预后更加坚持正确答案？
flip rate 是否下降？
模型是否在推理过程中受到不必要的用户影响？