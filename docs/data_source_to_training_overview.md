# DiagPRM 数据来源与训练流程说明

本文档说明当前 DiagPRM 项目如何从原始医疗数据集构造 KG，再由 KG 构造 RL 数据、SFT 数据，并最终进行 SFT/RL 训练。当前论文叙事固定为：

**KG-supervised turn-level credit assignment for diagnostic dialogue**

核心思想是：医生模型不直接看到 KG，而是在自然语言问诊中和病人交互；KG 作为隐藏监督信号，约束病人模拟器和 reward manager。医生每问出一个新的关键诊断事实，就可以通过 `Delta_KG` 获得 turn-level credit。

## 1. 原始数据来源

我们目前使用两类医疗资源。

**第一类：疾病-症状 KG 来源**

这部分用于构造主 KG，即 disease-to-symptom / disease-to-fact 结构。

- Columbia：从临床病历中抽取的疾病-症状关联。
- ChronoMedKG：覆盖面较广的疾病和症状知识图谱。
- Hetionet：较规范的 biomedical graph，其中包含 disease-symptom 关系。
- PrimeKG：补充 disease-phenotype / disease-symptom 证据，主要用于常见病检查和增强。



**第二类：医疗 QA / 考试 / 对话数据**

这部分不直接作为主疾病 KG 使用。原因是 MedQA、MedMCQA、MedicalExam 等很多样本是考试题或 case-level QA，不是稳定的 disease-level symptom profile。我们主要用它们来补充常见病、寻找更自然的 patient-observable facts。

- MedQA
- MedMCQA
- MedicalExam
- mediQ conversation data


## 2. 从原始数据到 Clean KG

原始疾病-症状数据会先被统一成 master KG。基本格式是：

```json
{
  "disease": {
    "symptom_or_fact": weight
  }
}
```

合并时会统一疾病名和症状/事实文本。对于同一个 disease-symptom pair，如果多个来源都有证据，则保留较强的权重；不同来源提供的不同症状会合并保留。之后过滤掉症状太少、噪声太重、非病人可观察的条目。

当前清洗后的主数据目录是：

- `diagprm_dataset/clean_v2`



每个 KG case 的核心字段类似：

```json
{
  "disease": "keratoconus",
  "chief_complaint": "A patient reports ...",
  "initial_symptoms": ["vision loss", "visual deterioration"],
  "symptoms_pool": {
    "acute hydrops": 0.39,
    "poor spectacle correction": 0.34
  },
  "symptom_facts": [
    {"fact_id": "F000", "text": "vision loss", "weight": 0.40},
    {"fact_id": "F001", "text": "acute hydrops", "weight": 0.39}
  ]
}
```

其中 `symptoms_pool` 和 `symptom_facts` 不暴露给医生模型，只作为病人模拟器和 reward manager 的隐藏监督。

## 3. 常见病 KG Augmentation

原始 KG 覆盖很广，但有大量罕见病和专科事实。这个覆盖对论文有价值，但训练早期会带来一个问题：医生模型问的问题可能在自然语言上合理，却很难精确命中隐藏 KG 中的专业事实，导致 reward 过稀疏。

因此我们额外做了常见病增强。增强的目标不是把 QA 数据一条条变成 KG，而是从 QA/考试/对话数据中找常见病证据，再整理成 disease-level、patient-observable 的症状和病史事实。



当前增强统计：

- 定义了 31 个常见病。
- 其中 28 个能匹配到原 KG 中的疾病名或别名。
- 新增 84 条 synthetic common-disease train cases。
- train 中共有 110 条样本被增强。
- train split 总计 4805 条。
- val/test 各 500 条。


## 4. 从 KG 到 RL 数据

RL 数据直接来自 KG case。每条 case 包含：

- 目标疾病；
- 初始主诉，由少量 initial symptoms 构造；
- 隐藏的诊断事实池；
- 每个事实稳定对应的 `fact_id`；
- train/val/test split 信息。

在 RL rollout 中，医生模型只看到自然语言对话。完整流程是：

1. Doctor actor 收到 system prompt 和病人的自然语言主诉。
2. Doctor 输出 JSON：

```json
{"action": "ask", "question": "..."}
```

或者：

```json
{"action": "diagnose", "diagnosis": "..."}
```

3. Patient simulator 根据隐藏 KG facts 回答自然语言；内部同时返回一个隐藏 `fact_id`，表示这个回答对应哪个 KG fact。
4. 下一轮 doctor 只能看到自然语言 patient answer，看不到 `fact_id`。
5. Reward manager 根据隐藏 `fact_id` 轨迹计算 KG 覆盖率增量。

turn-level reward 的核心是：

```text
r_turn(k) = Delta_KG(k)
```

其中 `Delta_KG(k)` 表示第 `k` 轮之后 KG 覆盖率相对于上一轮的增量。最终诊断 reward 只在 episode 末尾给出。因此我们的方法不是只奖励最终 diagnosis，而是奖励“问出有诊断价值的新事实”的过程。



## 5. 从 KG/Rollouts 到 SFT 数据

SFT 的目的不是覆盖所有 RL case，而是作为 warm start，让模型先学会基本问诊格式：

- 一次只问一个 focused question；
- 输出合法 JSON；
- 先收集证据，再给 diagnosis；
- 形成多轮问诊的基本行为模式。

当前 SFT 数据由 KG-grounded teacher/self-play 管线生成：

1. 从 KG case 中选择样本。
2. 为每个 case 选择若干值得询问的隐藏 facts。
3. 生成多条 candidate trajectories。
4. 用较强 teacher LLM / vLLM 服务把 opening、question、answer 改写成更自然的对话，但保持 fact id 和疾病标签不变。
5. 对候选轨迹打分和过滤。
6. 保留质量较高的轨迹用于 SFT。


当前 SFT 数据统计：

- raw rows：2153。
- dedup rows：2153。
- filtered rows：1849。
- filtered unique disease ids：1131。
- filtered 平均 KG coverage：约 0.787。
- filtered 平均新增事实数：约 3.324。
- filtered 平均 turns：约 4.324。

每条 SFT 数据是一个 `messages` 对话序列：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "patient opening"},
  {"role": "assistant", "content": "{\"action\":\"ask\",...}"},
  {"role": "user", "content": "patient answer"},
  {"role": "assistant", "content": "{\"action\":\"diagnose\",...}"}
]
```



## 6. 完整流程图

```text
原始 disease-symptom 数据源
  -> 合并 master KG
  -> clean_v2 清洗过滤
  -> QA-informed 常见病增强
  -> clean_v2_common_aug_qa_plus_seed_v2
  -> RL parquet 数据
  -> KG-supervised RL 训练 doctor policy

KG cases
  -> teacher/self-play 生成候选轨迹
  -> candidate scoring/filtering
  -> merged SFT parquet
  -> Qwen3 SFT warm-start checkpoint
  -> 作为 RL actor 初始化
```

