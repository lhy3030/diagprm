"""
DiagPRM SFT Data Generator
===========================
使用两个 GPT 角色扮演（Doctor-Agent + Patient-Simulator）对话，
生成符合 DiagPRM 训练格式的 SFT 数据集。

对话格式（每轮 Doctor 输出）：
  <think>...</think>
  <hypothesis_state>
    <hypothesis name="Disease A"><confirmed>[...]</confirmed><pending>[...]</pending></hypothesis>
  </hypothesis_state>
  <action>continue</action>
  <question>...</question>

最终诊断格式：
  <action>diagnose</action>
  <diagnosis>Disease Name</diagnosis>

输出 JSONL 格式（兼容 verl / diagprm RL 训练）：
  {
    "prompt": [{"role": "system", ...}, {"role": "user", ...}],
    "response": "<think>...</think>...<action>diagnose</action>...",
    "ground_truth": {"disease": "...", "atomic_facts": [...]},
    "extra_info": {"kg_disease": "...", "num_turns": N, "final_correct": true}
  }

用法：
  python generate_sft_data.py \\
    --input_jsonl /path/to/medqa_diag_test.jsonl \\
    --kg_path /path/to/master_kg.json \\
    --output_jsonl /path/to/sft_output.jsonl \\
    --api_key sk-... \\
    --model gpt-4o \\
    --max_turns 8 \\
    --n_samples 200 \\
    --concurrency 5
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Doctor Agent System Prompt（与 RL 阶段保持完全一致的格式）
# ─────────────────────────────────────────────────────────────────────────────

DOCTOR_SYSTEM_PROMPT = """You are an expert diagnostic physician conducting a structured medical interview.

Your goal is to diagnose the patient's condition by asking targeted questions, maintaining hypotheses, and deciding when to make a final diagnosis.

## Output Format (STRICTLY required every turn):

<think>
[Analyze current evidence. List confirmed symptoms. Evaluate each hypothesis against current evidence. Decide your next action.]
</think>
<hypothesis_state>
  <hypothesis name="[Disease Name 1]">
    <confirmed>[symptom1, symptom2, ...]</confirmed>
    <pending>[symptom3, symptom4, ...]</pending>
  </hypothesis>
  <hypothesis name="[Disease Name 2]">
    <confirmed>[...]</confirmed>
    <pending>[...]</pending>
  </hypothesis>
</hypothesis_state>
<action>continue</action>
<question>[Your focused single-symptom question here]</question>

## Action Types (choose EXACTLY ONE):
- `continue` : Ask ONE focused question to gather more evidence. You may update `<hypothesis_state>` (add/remove/reorder hypotheses) at any time to reflect new evidence. Output `<question>`.
- `diagnose` : Sufficient evidence gathered to confirm diagnosis. Output `<diagnosis>[Disease Name]</diagnosis>` instead of `<question>`.

## Rules:
1. Ask only ONE question per turn.
2. Never repeat a question already asked.
3. The FIRST hypothesis in `<hypothesis_state>` is your primary hypothesis.
4. You can freely reorder or update hypotheses under `continue` to reflect your changing belief.
5. When diagnosing, use `<action>diagnose</action>` and `<diagnosis>Disease Name</diagnosis>` (no `<question>`).
6. Maximum {max_turns} turns allowed.
7. Be strategic: use the KG knowledge to ask about the most discriminating symptoms first.

## KG Disease Symptoms Reference:
The target disease is in the KG. Key associated symptoms:
{kg_hints}

## Current turn: Turn {current_turn} / {max_turns}
"""

DOCTOR_INITIAL_PROMPT = """A patient presents with the following chief complaint:

{chief_complaint}

Please start your diagnostic interview. Begin by forming initial hypotheses based on the chief complaint and KG symptom knowledge, then ask your first targeted question."""


# ─────────────────────────────────────────────────────────────────────────────
# Patient Simulator System Prompt
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_SYSTEM_PROMPT = """You are a patient in a medical consultation. Answer the doctor's questions based ONLY on the following information about yourself.

Patient Information:
{atomic_facts}

Rules:
1. Answer ONLY based on the provided information above.
2. If the question asks about something not mentioned in your information, say: "I don't have that symptom" or "I'm not sure about that."
3. Keep answers brief and natural (1-2 sentences max).
4. Do NOT volunteer information the doctor hasn't asked about.
5. Speak in first person as a patient.
6. Be realistic: patients don't use medical jargon."""


# ─────────────────────────────────────────────────────────────────────────────
# KG 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_kg(kg_path: str) -> Dict:
    """加载 master_kg.json，返回 {disease_norm: {symptom_norm: weight}} 格式。"""
    logger.info(f"Loading KG from {kg_path} ...")
    with open(kg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    kg: Dict = {}
    for disease, symptoms in raw.items():
        d_norm = _normalize(disease)
        if isinstance(symptoms, dict):
            kg[d_norm] = {_normalize(s): float(w) for s, w in symptoms.items() if s and w is not None}
        elif isinstance(symptoms, list):
            kg[d_norm] = {_normalize(s): 1.0 for s in symptoms if s}
    logger.info(f"KG loaded: {len(kg):,} diseases")
    return kg


def _normalize(text: str) -> str:
    """小写 + 去特殊字符。"""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[^a-z0-9 \-']", " ", text.lower()).strip()


def get_kg_hints(disease: str, kg: Dict, top_k: int = 15) -> str:
    """
    从 KG 中提取 disease 的 top-k 症状作为 Doctor 的 hint，
    帮助 GPT 生成更有针对性的诊断问题。
    
    Returns:
        格式化的症状列表字符串。
    """
    d_norm = _normalize(disease)
    if d_norm not in kg:
        # 模糊匹配
        best_match = _fuzzy_match_disease(d_norm, kg)
        if best_match:
            d_norm = best_match
        else:
            return "(No KG entry found for this disease)"

    sym_dict = kg[d_norm]
    # 按权重降序取 top_k
    top_syms = sorted(sym_dict.items(), key=lambda x: -x[1])[:top_k]
    lines = [f"  - {sym} (weight: {w:.3f})" for sym, w in top_syms]
    return "\n".join(lines)


def _fuzzy_match_disease(disease_str: str, kg: Dict) -> Optional[str]:
    """Jaccard 模糊匹配 KG 中最近的疾病名。"""
    tokens = set(_normalize(disease_str).split())
    best_score = 0.0
    best_match = None
    for d in kg:
        d_tokens = set(d.split())
        if not d_tokens:
            continue
        intersection = len(tokens & d_tokens)
        union = len(tokens | d_tokens)
        score = intersection / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_match = d
    return best_match if best_score > 0.3 else None


def compute_kg_coverage(collected: Set[str], disease: str, kg: Dict) -> float:
    """计算收集症状对疾病的 KG 覆盖率（带权重）。"""
    d_norm = _normalize(disease)
    if d_norm not in kg or not kg[d_norm]:
        return 0.0
    sym_dict = kg[d_norm]
    total_w = sum(sym_dict.values())
    if total_w == 0:
        return 0.0
    covered_w = sum(w for s, w in sym_dict.items() if s in collected)
    return covered_w / total_w


def extract_symptoms_from_text(text: str, kg: Dict) -> Set[str]:
    """从文本中提取 KG 中存在的症状 n-gram。"""
    norm = _normalize(text)
    tokens = norm.split()
    ngrams: Set[str] = set()
    for n in range(1, 5):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i:i + n]))
    # 收集 KG 所有症状
    all_syms: Set[str] = set()
    for sym_dict in kg.values():
        all_syms.update(sym_dict.keys())
    return ngrams & all_syms


# ─────────────────────────────────────────────────────────────────────────────
# GPT API 调用
# ─────────────────────────────────────────────────────────────────────────────

async def call_gpt(
    client: AsyncOpenAI,
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
) -> Optional[str]:
    """异步调用 GPT API，带重试逻辑。"""
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"GPT API error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 对话解析工具
# ─────────────────────────────────────────────────────────────────────────────

def parse_action(doctor_response: str) -> Optional[str]:
    """解析 <action> 标签，switch 归并为 continue。"""
    match = re.search(
        r'<action>\s*(continue|switch|diagnose)\s*</action>',
        doctor_response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        raw = match.group(1).strip().lower()
        return "continue" if raw == "switch" else raw
    return None


def parse_question(doctor_response: str) -> Optional[str]:
    """解析 <question> 标签。"""
    match = re.search(
        r'<question>\s*(.*?)\s*</question>',
        doctor_response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    match2 = re.search(r'Question:\s*(.+)', doctor_response, re.IGNORECASE)
    if match2:
        return match2.group(1).strip()
    return None


def parse_final_diagnosis(doctor_response: str) -> Optional[str]:
    """解析 <diagnosis> 标签。"""
    match = re.search(
        r'<diagnosis>\s*(.*?)\s*</diagnosis>',
        doctor_response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


def check_format_valid(doctor_response: str) -> bool:
    """检查 Doctor 输出是否包含必要的格式标签。"""
    has_think = bool(re.search(r'<think>.*?</think>', doctor_response, re.DOTALL))
    has_hyp = bool(re.search(r'<hypothesis_state>', doctor_response, re.IGNORECASE))
    has_action = parse_action(doctor_response) is not None
    return has_think and has_hyp and has_action


# ─────────────────────────────────────────────────────────────────────────────
# 单条样本的 SFT 对话生成
# ─────────────────────────────────────────────────────────────────────────────

async def generate_single_dialogue(
    client: AsyncOpenAI,
    sample: Dict,
    kg: Dict,
    model: str,
    max_turns: int,
    patient_temperature: float = 0.3,
    doctor_temperature: float = 0.7,
) -> Optional[Dict]:
    """
    为单条 MedQA 样本生成完整的 DiagPRM 格式 SFT 对话。
    
    Returns:
        SFT 样本字典，或 None（对话失败时）。
    """
    # 提取信息
    if "extra_info" in sample and "atomic_facts" in sample["extra_info"]:
        atomic_facts = sample["extra_info"]["atomic_facts"]
    else:
        atomic_facts = []
    
    # ground truth 疾病名
    gt_disease = ""
    if "ground_truth" in sample:
        gt = sample["ground_truth"]
        if isinstance(gt, dict):
            gt_disease = gt.get("disease", gt.get("answer_info", ""))
        else:
            gt_disease = str(gt)
    
    # 从 prompt 中提取主诉
    chief_complaint = ""
    if "prompt" in sample and isinstance(sample["prompt"], list):
        for msg in sample["prompt"]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                chief_complaint = msg.get("content", "")
                break
    elif "prompt" in sample and isinstance(sample["prompt"], str):
        chief_complaint = sample["prompt"]
    
    if not chief_complaint or not gt_disease:
        logger.warning(f"Skipping sample with missing chief_complaint or ground_truth: {sample.get('index', '?')}")
        return None

    # 从 KG 获取疾病相关症状提示
    kg_hints = get_kg_hints(gt_disease, kg, top_k=15)
    
    # ── 患者消息初始化 ───────────────────────────────────────────────────────
    facts_text = "\n".join(f"- {f}" for f in atomic_facts) if atomic_facts else "No additional information available."
    patient_system = PATIENT_SYSTEM_PROMPT.format(atomic_facts=facts_text)
    
    # ── Doctor 对话历史 ──────────────────────────────────────────────────────
    doctor_messages: List[Dict] = []
    dialogue_turns: List[Dict] = []  # 记录完整对话用于 SFT 样本构建
    
    collected_symptoms: Set[str] = set()
    previous_questions: List[str] = []
    final_action = None
    final_diagnosis = None
    
    for turn_idx in range(max_turns):
        current_turn = turn_idx + 1
        is_last_turn = (turn_idx == max_turns - 1)
        
        # 构建 Doctor 系统消息（包含 KG hints 和 turn 计数）
        doctor_system_content = DOCTOR_SYSTEM_PROMPT.format(
            max_turns=max_turns,
            current_turn=current_turn,
            kg_hints=kg_hints,
        )
        
        # 首轮：添加初始 user 消息
        if turn_idx == 0:
            doctor_messages = [
                {"role": "system", "content": doctor_system_content},
                {"role": "user", "content": DOCTOR_INITIAL_PROMPT.format(
                    chief_complaint=chief_complaint,
                )},
            ]
        else:
            # 更新 system 消息中的 turn 计数
            doctor_messages[0] = {"role": "system", "content": doctor_system_content}
        
        # 调用 Doctor GPT
        doctor_response = await call_gpt(
            client, doctor_messages,
            model=model,
            temperature=doctor_temperature,
            max_tokens=1200,
        )
        if doctor_response is None:
            logger.warning(f"Doctor API failed at turn {current_turn}")
            break
        
        # 检查格式
        if not check_format_valid(doctor_response):
            logger.debug(f"Invalid doctor format at turn {current_turn}, skipping sample")
            # 给一次重试机会
            retry_msg = doctor_messages.copy()
            retry_msg.append({
                "role": "user",
                "content": (
                    "Your response is missing required XML tags. "
                    "Please reformat strictly following: "
                    "<think>...</think><hypothesis_state>...</hypothesis_state>"
                    "<action>continue|diagnose</action><question>...</question>"
                )
            })
            doctor_response = await call_gpt(
                client, retry_msg,
                model=model,
                temperature=doctor_temperature * 0.5,
                max_tokens=1200,
            )
            if doctor_response is None or not check_format_valid(doctor_response):
                break
        
        # 更新 Doctor 消息历史
        doctor_messages.append({"role": "assistant", "content": doctor_response})
        
        # 解析 action
        action = parse_action(doctor_response)
        final_action = action
        
        # 记录本轮
        dialogue_turns.append({
            "turn_id": turn_idx,
            "doctor_response": doctor_response,
            "action": action,
        })
        
        # 如果是确诊动作或最后一轮，停止
        if action == "diagnose" or is_last_turn:
            final_diagnosis = parse_final_diagnosis(doctor_response)
            break
        
        # ── 患者回答 ─────────────────────────────────────────────────────────
        question = parse_question(doctor_response)
        if question is None:
            # 没有问题，跳过本轮
            logger.debug(f"No question found at turn {current_turn}")
            break
        
        # 检查重复问题（简单 Jaccard 规则）
        q_tokens = set(question.lower().split())
        is_repeated = False
        for prev_q in previous_questions:
            p_tokens = set(prev_q.lower().split())
            if not p_tokens:
                continue
            overlap = len(q_tokens & p_tokens) / max(len(q_tokens | p_tokens), 1)
            if overlap > 0.7:
                is_repeated = True
                break
        
        if is_repeated:
            patient_answer = "You already asked me that."
        else:
            # 调用患者 GPT
            patient_messages = [
                {"role": "system", "content": patient_system},
                {"role": "user", "content": f"Doctor's question: {question}"},
            ]
            patient_answer = await call_gpt(
                client, patient_messages,
                model=model,
                temperature=patient_temperature,
                max_tokens=150,
            )
            if patient_answer is None:
                patient_answer = "I'm not sure about that."
        
        # 更新症状集合
        combined = (question or "") + " " + (patient_answer or "")
        new_syms = extract_symptoms_from_text(combined, kg)
        # 简单否定过滤：如果患者说 "no" / "don't"，不加入
        deny_re = re.compile(r"\b(no|not|don't|doesn't|never|absent|deny|without)\b", re.IGNORECASE)
        if not deny_re.search(patient_answer or ""):
            collected_symptoms.update(new_syms)
        
        # 记录患者回答
        dialogue_turns[-1]["patient_answer"] = patient_answer
        dialogue_turns[-1]["question"] = question
        previous_questions.append(question)
        
        # 将患者回答加入 Doctor 的对话历史
        doctor_messages.append({"role": "user", "content": patient_answer})
    
    if not dialogue_turns:
        return None
    
    # ── 构建 SFT 样本 ────────────────────────────────────────────────────────
    # SFT 格式：将整个多轮对话展开为 prompt + response
    # prompt: system + initial user
    # response: 所有 doctor 轮次的完整输出拼接（用于 SFT 的监督信号）
    
    # 方案：每一轮独立作为一条 SFT 样本（更细粒度的格式学习）
    sft_samples = []
    
    # 重建对话历史，逐轮生成样本
    rolling_messages: List[Dict] = []
    
    for t_idx, turn in enumerate(dialogue_turns):
        current_turn = t_idx + 1
        doctor_sys = DOCTOR_SYSTEM_PROMPT.format(
            max_turns=max_turns,
            current_turn=current_turn,
            kg_hints=kg_hints,
        )
        
        if t_idx == 0:
            user_content = DOCTOR_INITIAL_PROMPT.format(chief_complaint=chief_complaint)
            prompt_messages = [
                {"role": "system", "content": doctor_sys},
                {"role": "user", "content": user_content},
            ]
        else:
            # 继承历史，更新 system turn
            prompt_messages = [{"role": "system", "content": doctor_sys}] + rolling_messages[1:]
        
        doctor_resp = turn["doctor_response"]
        
        # 这一轮是 SFT 样本
        sft_sample = {
            "prompt": prompt_messages,
            "response": doctor_resp,
            "ground_truth": {
                "disease": gt_disease,
                "atomic_facts": atomic_facts,
            },
            "extra_info": {
                "source_index": sample.get("index", -1),
                "data_source": sample.get("data_source", "unknown"),
                "turn_id": t_idx,
                "total_turns": len(dialogue_turns),
                "action": turn["action"],
                "is_final_turn": (t_idx == len(dialogue_turns) - 1),
                "final_correct": (
                    final_diagnosis is not None and
                    _normalize(final_diagnosis) in _normalize(gt_disease) or
                    _normalize(gt_disease) in _normalize(final_diagnosis or "")
                ),
                "kg_coverage": compute_kg_coverage(collected_symptoms, gt_disease, kg),
            },
        }
        sft_samples.append(sft_sample)
        
        # 更新滚动历史（下一轮的 context）
        if t_idx == 0:
            rolling_messages = [
                {"role": "system", "content": doctor_sys},
                {"role": "user", "content": DOCTOR_INITIAL_PROMPT.format(chief_complaint=chief_complaint)},
                {"role": "assistant", "content": doctor_resp},
            ]
        else:
            rolling_messages.append({"role": "assistant", "content": doctor_resp})
        
        # 如果有患者回答，加入历史
        if "patient_answer" in turn:
            rolling_messages.append({"role": "user", "content": turn["patient_answer"]})
    
    return sft_samples


# ─────────────────────────────────────────────────────────────────────────────
# 批量生成主流程
# ─────────────────────────────────────────────────────────────────────────────

async def generate_sft_dataset(
    input_jsonl: str,
    kg_path: str,
    output_jsonl: str,
    api_key: str,
    model: str,
    max_turns: int,
    n_samples: int,
    concurrency: int,
    seed: int,
    base_url: Optional[str] = None,
) -> None:
    """主生成函数：读取 MedQA 数据，并发生成 SFT 对话，写入 JSONL。"""
    
    # 初始化 OpenAI 客户端
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    
    # 加载 KG
    kg = load_kg(kg_path)
    
    # 加载输入数据
    logger.info(f"Loading input data from {input_jsonl} ...")
    samples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line: {e}")
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # 随机采样
    random.seed(seed)
    if n_samples > 0 and n_samples < len(samples):
        samples = random.sample(samples, n_samples)
        logger.info(f"Sampled {n_samples} samples (seed={seed})")
    
    # 过滤：只保留有 atomic_facts 和 ground_truth disease 的样本
    valid_samples = []
    for s in samples:
        atomic_facts = s.get("extra_info", {}).get("atomic_facts", [])
        gt = s.get("ground_truth", {})
        gt_disease = ""
        if isinstance(gt, dict):
            gt_disease = gt.get("disease", gt.get("answer_info", ""))
        elif isinstance(gt, str):
            gt_disease = gt
        
        if atomic_facts and gt_disease:
            valid_samples.append(s)
        else:
            logger.debug(f"Skipped sample (missing atomic_facts or disease): index={s.get('index', '?')}")
    
    logger.info(f"Valid samples after filtering: {len(valid_samples)}")
    
    # 并发生成
    semaphore = asyncio.Semaphore(concurrency)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_written = 0
    failed = 0
    
    async def process_one(sample: Dict, idx: int) -> List[Dict]:
        async with semaphore:
            try:
                result = await generate_single_dialogue(
                    client=client,
                    sample=sample,
                    kg=kg,
                    model=model,
                    max_turns=max_turns,
                )
                if result:
                    logger.info(f"[{idx+1}/{len(valid_samples)}] Generated {len(result)} turns for sample index={sample.get('index', '?')}")
                    return result
                else:
                    logger.warning(f"[{idx+1}/{len(valid_samples)}] Failed to generate for sample index={sample.get('index', '?')}")
                    return []
            except Exception as e:
                logger.error(f"[{idx+1}/{len(valid_samples)}] Error: {e}", exc_info=True)
                return []
    
    # 分批并发执行
    with open(output_path, "w", encoding="utf-8") as out_f:
        tasks = [process_one(s, i) for i, s in enumerate(valid_samples)]
        
        # 使用 gather 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        for sft_samples in results:
            if sft_samples:
                for sft_sample in sft_samples:
                    out_f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                    total_written += 1
            else:
                failed += 1
        
        out_f.flush()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SFT data generation complete!")
    logger.info(f"  Total SFT samples written: {total_written}")
    logger.info(f"  Failed cases:              {failed}")
    logger.info(f"  Output file:               {output_path}")
    logger.info(f"{'='*60}")
    
    # 统计摘要
    _print_statistics(output_jsonl)


def _print_statistics(output_jsonl: str) -> None:
    """打印生成数据集的统计信息。"""
    try:
        samples = []
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        
        if not samples:
            return
        
        total = len(samples)
        final_turns = [s for s in samples if s.get("extra_info", {}).get("is_final_turn", False)]
        correct = [s for s in final_turns if s.get("extra_info", {}).get("final_correct", False)]
        diagnose_turns = [s for s in samples if s.get("extra_info", {}).get("action") == "diagnose"]
        
        turns_per_dialogue: Dict[Any, int] = {}
        for s in samples:
            idx = s.get("extra_info", {}).get("source_index", -1)
            turns_per_dialogue[idx] = turns_per_dialogue.get(idx, 0) + 1
        
        avg_turns = sum(turns_per_dialogue.values()) / max(len(turns_per_dialogue), 1)
        
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Total SFT turns:          {total}")
        logger.info(f"Unique dialogues:          {len(turns_per_dialogue)}")
        logger.info(f"Avg turns per dialogue:    {avg_turns:.1f}")
        logger.info(f"Correct final diagnosis:   {len(correct)}/{len(final_turns)} ({100*len(correct)/max(len(final_turns),1):.1f}%)")
        logger.info(f"Diagnose action turns:     {len(diagnose_turns)}")
        logger.info("=" * 30)
    except Exception as e:
        logger.warning(f"Failed to print statistics: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DiagPRM SFT training data using GPT Doctor + Patient simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file (e.g., medqa_diag_test.jsonl)",
    )
    parser.add_argument(
        "--kg_path",
        type=str,
        required=True,
        help="Path to master_kg.json",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output path for SFT data JSONL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Custom API base URL (for Azure, proxy, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=8,
        help="Maximum turns per dialogue (default: 8)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of input samples to process (default: 200, -1 = all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # API key 优先读取命令行参数，其次读取环境变量
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API key not provided. Use --api_key or set OPENAI_API_KEY env var.")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("DiagPRM SFT Data Generator")
    logger.info(f"  Model:       {args.model}")
    logger.info(f"  Max turns:   {args.max_turns}")
    logger.info(f"  N samples:   {args.n_samples}")
    logger.info(f"  Concurrency: {args.concurrency}")
    logger.info(f"  Input:       {args.input_jsonl}")
    logger.info(f"  KG:          {args.kg_path}")
    logger.info(f"  Output:      {args.output_jsonl}")
    if args.base_url:
        logger.info(f"  Base URL:    {args.base_url}")
    logger.info("=" * 60)
    
    asyncio.run(
        generate_sft_dataset(
            input_jsonl=args.input_jsonl,
            kg_path=args.kg_path,
            output_jsonl=args.output_jsonl,
            api_key=api_key,
            model=args.model,
            max_turns=args.max_turns,
            n_samples=args.n_samples,
            concurrency=args.concurrency,
            seed=args.seed,
            base_url=args.base_url,
        )
    )


if __name__ == "__main__":
    main()
