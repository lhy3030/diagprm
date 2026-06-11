"""
Verifier Prompts

Verifier 是一个轻量规则检查器（非独立 LLM agent），用于检测 Doctor 的问题是否
重复或包含多个问题。当前实现为纯规则（Jaccard 相似度 + 问号计数），此处
保留 LLM 版本的 prompt 供需要时切换。

变量：
  VERIFIER_SYSTEM_PROMPT : {previous_questions}, {current_question}
"""

VERIFIER_SYSTEM_PROMPT = """Analyze the doctor's question and respond with EXACTLY ONE tag:
- <Repeated>: if this question is very similar to a previous question already asked
- <Multiple>: if this question contains multiple distinct questions
- <Normal>: otherwise

Previous questions asked:
{previous_questions}

Current question:
{current_question}

Response (one tag only):"""
