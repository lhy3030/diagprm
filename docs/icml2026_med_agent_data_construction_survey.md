# ICML 2026 / Recent Medical Agent Data Construction Survey

本文档回答一个和 DiagPRM 论文定位直接相关的问题：

> 我们的方法要求 KG 覆盖 RL 训练数据，因此 RL 数据必须从 KG case 构造。这会不会太局限？近期 medical AI / agent 论文是如何构造交互数据、环境和 reward/evaluation 的？

结论先行：**这种限制是合理的，但需要在论文里写成“verifiable KG-grounded diagnostic dialogue environment”，而不是声称覆盖所有真实临床对话。** 近期 agent benchmark 的共同做法也是先构造一个可验证的 hidden state / environment，再让 agent 在其中交互；区别是它们的 hidden state 可能来自 EHR、MedQA case、OSCE template、FHIR database 或人工设计的 grader。我们的 hidden state 是 disease-symptom KG，因此 RL 数据从 KG 采样不是缺陷，而是 credit assignment 成立的前提。

## 0. 调研边界

用户给出的目标页面是 ICML 2026 OpenReview accepted spotlight/regular 列表。当前环境下，OpenReview 动态页面和 API 无法完整抓取；浏览器访问 OpenReview 也被本地安全策略拦截。因此本文没有声称完成 ICML 2026 全量 accepted paper 枚举。

本文采用的证据来源是：

- 可公开核验的 ICML 2026 agent 论文：Reward Hacking Benchmark, arXiv 页面标注 ICML，新闻报道也说明其 ICML 2026 accepted。
- 近期最相关的 medical agent / clinical decision benchmark：AgentClinic、MedAgentBench、MedicalAgentsBench、MedAgentBoard。
- 重点关注“数据如何从原始来源变成交互环境 / agent trajectory / reward 或 evaluator”，而不是普通医疗图像或纯 QA 论文。

## 1. 可核验的 ICML 2026 Agent 论文：Reward Hacking Benchmark

**Paper:** Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use  
**Status:** arXiv 页面主题标注含 `Machine Learning, ICML`；新闻报道说明 accepted to ICML 2026。  
**Medical?** 不是医疗，但它是 ICML 2026 agent/evaluation 方向中对我们最有方法论启发的一篇。

### 数据 / 环境如何构造

RHB 不是从自然数据集中直接抽样，而是**人工设计一组可执行、多步、带隐藏校验的 tool-use tasks**。任务覆盖 data pipeline、log forensics、performance optimization、multi-file reconstruction 等。每个任务都有：

- agent 可见的 workspace 和工具；
- 正常完成路径；
- 可被利用的 shortcut / exploit opportunity；
- 隐藏或重算的 grader；
- integrity instrumentation，用于区分 honest success 和 reward hacking。

它还构造了 chained task regime，用 chain length 模拟 longer-horizon agent behavior。

### 对我们的启发

RHB 的关键不是“数据天然真实”，而是**环境和 reward 可验证**。它为了研究 reward hacking，必须构造一个 grader 能检测的环境；我们为了研究 KG-supervised turn-level credit assignment，也必须构造一个 KG 能覆盖的环境。

对应到 DiagPRM：

- RHB 的 hidden grader / integrity checks 类似我们的 hidden KG facts。
- RHB 的 multi-step tool workflow 类似我们的 multi-turn diagnostic dialogue。
- RHB 的 reward hacking 风险提醒我们：KG reward 需要防止模型为了 `Delta_KG` 过度追问、不诊断。这正是我们现在实验里观察到的 ask-collapse，需要 final-turn diagnose constraint 和 timeout penalty。

## 2. AgentClinic：把 MedQA / MIMIC-IV / NEJM 转成交互式诊断环境

**Paper:** AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments  
**Medical agent relevance:** 非常高。它把静态医学 QA / EHR case 改造成 patient-doctor interactive diagnosis。

### 数据来源

AgentClinic 使用多种来源：

- MedQA：医学考试题，原本是静态选择题。
- MIMIC-IV：真实 EHR 数据。
- NEJM case challenges：临床 case challenge。
- MedMCQA：用于 specialist cases。

### 数据如何构造

它不是直接把原始题目丢给 agent，而是先转成 OSCE-style structured case。论文描述：对 AgentClinic-MedQA 和 AgentClinic-MIMIC-IV，先从 MedQA/MIMIC-IV 抽样，再填充 structured JSON case，其中包含患者信息、症状、病史、体检、检查结果、正确诊断等。部分 structured JSON 由 GPT-4 初步填充，再人工验证。

AgentClinic 的交互环境包含四类 agent：

- patient agent：只知道自己的病情资料；
- doctor agent：被评估模型，通过对话和工具收集信息；
- measurement agent：模拟体温、血压、EKG、实验室检查等；
- moderator：最终比较诊断和 ground truth。

MIMIC-IV 部分为了适配“单一最终诊断”，从 MIMIC-IV 里选择只有一个诊断的病人；并抽取 lab events、microbiology events 和 medical records。论文还定义了 OSCE case schema，例如 demographics、history、primary/secondary symptoms、past medical history、social history、review of systems、vital signs、physical examination、test results、correct diagnosis。

### 对我们的启发

AgentClinic 证明了一个重要点：**将静态医疗 QA 转成多轮诊断环境是合理的，但必须先结构化 hidden patient state。**

这和我们很像：

- AgentClinic 的 OSCE JSON 是 hidden patient state。
- DiagPRM 的 KG facts 是 hidden patient state。
- doctor 都只能通过对话逐步收集信息。
- 最终诊断由 hidden ground truth 评价。

区别在于 AgentClinic 主要做 evaluation benchmark；我们的目标是 RL training with turn-level credit，因此我们需要每一轮都能判断“问出了哪个新事实”。这就要求 RL 数据必须来自 KG 或结构化 fact table，否则 `Delta_KG` 没法稳定计算。

## 3. MedAgentBench：从真实 EHR 构造 FHIR 交互环境

**Paper:** MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents  
**Medical agent relevance:** 高。它不是诊断对话，而是 EHR tool-use agent benchmark。

### 数据来源

MedAgentBench 使用 Stanford STARR deidentified clinical data warehouse。论文说明其 patient profiles 来自去标识化、时间 jittered 的真实临床数据。

### 数据如何构造

它构造了：

- 100 个 patient profiles；
- 超过 700,000 条数据元素；
- 300 个 patient-specific tasks；
- 10 类任务，包括 patient information retrieval、lab result retrieval、patient data aggregation、recording patient data、test ordering、referral ordering、medication ordering 等。

患者数据被上传到 FHIR-compliant environment。agent 可调用九类 FHIR function，例如：

- `condition.search`
- `lab.search`
- `vital.search`
- `vital.create`
- `medicationrequest.search`
- `medicationrequest.create`
- `procedure.search`
- `procedure.create`
- `patient.search`

每轮 agent 选择 GET、POST 或 finish，最多 8 轮。评价是 pass@1，因为医疗场景容错低。

### 对我们的启发

MedAgentBench 的构造逻辑是：

```text
真实 EHR source
  -> 去标识化 / jitter
  -> 抽取结构化 records
  -> 上传到 FHIR database
  -> 设计 task + tool schema
  -> agent 交互
  -> evaluator 判断任务是否完成
```

这和我们可以类比为：

```text
medical KG source
  -> disease-symptom/fact cleaning
  -> hidden fact table
  -> patient simulator + reward manager
  -> doctor dialogue policy
  -> KG coverage / diagnosis evaluator
```

也就是说，**agent 训练/评估通常不是直接用“自然数据”，而是先把原始数据变成一个可交互、可验证的 environment**。我们选择 KG environment 是合理的。

## 4. MedicalAgentsBench：从多个医学 QA 数据集中筛复杂推理题

**Paper:** MedicalAgentsBench for Complex Medical Reasoning  
**Medical agent relevance:** 中高。它更偏 complex medical reasoning / agent frameworks，不是自然诊断对话训练。

### 数据来源与构造

该工作从八个医学数据集的 union 中抽取问题，通过 difficulty-aware curation 和 contamination screening 构造 862 个 complex clinical questions。它评估 internalized reasoning models 和 externalized agent-based frameworks。

### 对我们的启发

它说明 QA 数据可以用于 medical agent / reasoning 评估，但它更适合：

- 复杂医学推理；
- 多 agent debate；
- treatment planning / diagnosis formulation；
- inference-time agent scaffold 比较。

它不直接解决我们的 turn-level reward 问题，因为 QA 数据通常没有逐轮 hidden fact id。若要用于 DiagPRM，需要先做结构化转换：

```text
QA case
  -> disease label
  -> patient-observable facts
  -> initial symptoms
  -> hidden fact ids
  -> patient simulator
  -> KG-compatible reward
```

否则只能做 final-answer evaluation，不能做 `Delta_KG` credit assignment。

## 5. MedAgentBoard：多任务医疗 agent benchmark，但不主打诊断对话

**Paper:** MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks  
**Venue:** NeurIPS 2025 Datasets & Benchmarks, not ICML 2026.  
**Medical agent relevance:** 中等。它覆盖多种医疗任务，但更像 broad benchmark。

### 数据来源与构造

MedAgentBoard 覆盖四类任务：

- medical / visual question answering；
- lay summary generation；
- structured EHR predictive modeling；
- clinical workflow automation。

它比较 multi-agent collaboration、single LLM、conventional methods。结论很重要：multi-agent 不总是优于 single LLM 或传统模型，只有在某些 workflow automation 场景中有优势。

### 对我们的启发

它提醒我们论文不应该把 contribution 写成“self-play SFT”或“multi-agent 框架很强”。医疗场景里，审稿人更关心：

- task-specific evidence；
- benchmark 是否贴合任务；
- 是否有可靠 evaluator；
- 是否和传统或简单 baseline 比较。

对 DiagPRM 来说，核心应该继续放在：

```text
KG building + KG as turn-level reward
```

SFT/self-play 只是 warm-start，不是核心 contribution。

## 6. 和 DiagPRM 的关系：为什么 RL 数据必须从 KG 来

我们的论文实现要求：

```text
KG must cover RL training data.
```

这确实带来局限：RL 训练的 disease/fact 空间被 KG 限制，不能直接覆盖任意真实对话。但这是方法成立的必要条件，因为我们不是训练普通 chatbot，而是在训练：

```text
KG-supervised turn-level credit assignment for diagnostic dialogue
```

要给每一轮问诊分配 credit，就必须知道：

- 这一轮 patient answer 对应哪个 hidden fact；
- 这个 fact 是否属于目标 disease 的 KG；
- 它是否是新事实；
- 它的 diagnostic weight；
- 当前 coverage 是否提升。

如果 RL 数据不是从 KG 或等价结构化 fact table 生成，就无法稳定计算 `Delta_KG`。这和 AgentClinic/MedAgentBench 的做法一致：它们也先构造 hidden patient state 或 FHIR environment，再让 agent 交互。

## 7. 推荐论文写法

### 7.1 不要这样写

不建议写成：

> We train a general medical dialogue agent from arbitrary clinical conversations.

这会被挑战：真实对话没有逐轮 fact id，KG 覆盖也不完整。

### 7.2 建议这样写

建议写成：

> We study a verifiable diagnostic-dialogue setting where each training case is grounded in a disease-specific KG of patient-observable diagnostic facts. This design enables turn-level credit assignment: when the doctor elicits a new KG-supported fact through natural-language dialogue, the policy receives process reward proportional to the increase in KG coverage.

中文理解：

> 我们不是宣称训练一个无边界的医疗聊天机器人，而是研究一个可验证的诊断对话 RL 设置。在这个设置里，每个 case 都有 KG 支持的隐藏事实，因此可以精确判断医生哪一轮问出了有价值的新诊断信息。

### 7.3 局限性可以主动承认

可以写：

> The KG-grounded design restricts RL training to diseases and facts covered by the KG. We view this as a deliberate trade-off: coverage is narrower than unconstrained clinical dialogue, but it provides reliable process-level supervision that ordinary outcome-only medical QA or raw dialogues do not provide.

中文：

> KG 约束让训练域变窄，但换来了普通医疗 QA / 原始对话没有的逐轮过程监督。

## 8. 对实验构造的建议

### 8.1 主实验继续用 KG-derived RL

主实验必须保持：

```text
KG case -> hidden fact ids -> patient simulator -> Delta_KG reward
```

否则论文核心就会散。

### 8.2 常见病 KG 是合理增强

学长说的 common disease augmentation 很重要。因为 full KG 里有大量罕见病和专业术语，doctor 很难自然命中。常见病增强可以写成：

> To avoid evaluating only rare or highly specialized disease names, we construct a common-disease subset by augmenting KG entries with patient-observable facts derived from common diseases appearing in medical QA sources.

注意：QA 不直接变成 RL 轨迹，而是用于补全 common disease KG。

### 8.3 SFT 不必覆盖全部 RL 数据

SFT 的作用是 warm-start：

- 学 JSON 格式；
- 学一次问一个问题；
- 学先问诊后诊断；
- 学自然 patient dialogue surface。

它不需要把所有 RL case 都变成 SFT。更合理的是选 high-quality KG-grounded trajectories，覆盖疾病多样性和常见问诊模式即可。

### 8.4 需要单独报告 KG reward 是否真的工作

建议实验表里至少报告：

- `known_fact_rate`
- `final_kg_coverage`
- `delta_kg_sum`
- `diagnose_rate`
- `last_ask_rate`
- `premature_diag_rate`
- `accuracy / correct diagnosis`

这直接对应学长说的“KG as turn-level reward 这件事要重点测”。

## 9. 论文贡献建议

最终建议把 contribution 写成三点：

1. **KG-grounded diagnostic dialogue environment**  
   从 disease-symptom KG 构造 hidden patient facts，使诊断对话变成可验证的 sequential decision-making problem。

2. **KG-supervised turn-level credit assignment**  
   用 `Delta_KG` 给医生每一轮问诊分配过程奖励，缓解 outcome-only diagnosis reward 的稀疏性。

3. **Empirical analysis of process reward in diagnostic dialogue**  
   通过 Outcome-only GRPO、DiagPRM、w/o `Delta_KG`、`alpha=0`、KG tool only、hypothesis tracking ablation 等实验，证明 KG reward 是否提高 evidence collection、diagnosis timing 和 final accuracy。

SFT/self-play 数据可以作为：

- warm-start procedure；
- released KG-grounded dialogue data；
- implementation detail；

但不建议作为核心 contribution。

## 10. 参考链接

- Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use. arXiv: https://arxiv.org/abs/2605.02964
- Times of India report on ICML 2026 acceptance of Reward Hacking Benchmark: https://timesofindia.indiatimes.com/world/us/meet-kunvar-thaman-solo-indian-researcher-whose-paper-was-accepted-at-an-elite-ai-conference-dominated-by-openai-and-deepmind/articleshow/130853557.cms
- AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments. arXiv: https://arxiv.org/abs/2405.07960
- MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents. arXiv: https://arxiv.org/abs/2501.14654
- MedicalAgentsBench for Complex Medical Reasoning. arXiv: https://arxiv.org/abs/2503.07459
- MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks. arXiv: https://arxiv.org/abs/2505.12371

