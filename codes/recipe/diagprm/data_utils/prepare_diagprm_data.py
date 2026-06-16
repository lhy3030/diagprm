"""
DiagPRM 数据预处理脚本

将 mediQ / MedQA 等数据集转换为 DiagPRM 训练格式，
输出 parquet 文件（verl RLHFDataset 标准格式）。

支持两种输入格式：

格式 1（merged_train_dataset.jsonl，ATPO MCQA 数据）：
{
  "prompt": [{"role": "system", ...}, {"role": "user", "content": "<chief_complaint>\nProblem: ...\nOptions: {...}"}],
  "ground_truth": {"answer": "A", "answer_info": "Disease Name"},
  "extra_info": {"atomic_facts": ["1. ...", "2. ...", ...]},
  "data_source": "mcqa"
}

格式 2（kg_train_dataset.jsonl，KG 构建的诊断对话数据，推荐）：
{
  "prompt": [{"role": "system", ...}, {"role": "user", "content": "<initial_info>\nProblem: ...\nOptions: {...}"}],
  "ground_truth": {"answer": "A", "answer_info": "Disease Name", "disease": "disease name",
                   "symptoms_pool": {"symptom": weight, ...}},
  "extra_info": {"atomic_facts": ["1. ...", ...]},
  "data_source": "kg"
}

输出格式（diagprm_train.parquet）：
每行：
{
  "prompt": [{"role": "system", "content": "<DOCTOR_SYSTEM_PROMPT>"}, 
             {"role": "user", "content": "<chief_complaint>"}],
  "reward_model": {
    "ground_truth": {
      "disease": "Disease Name",         # GT 疾病名（规范化）
      "disease_raw": "Disease Name",     # 原始字符串（for logging）
      "answer": "A",                     # MCQA 的选项（兼容 ATPO eval）
      "atomic_facts": ["1. ...", ...]    # 患者 simulator 使用
    }
  },
  "data_source": "diagprm",
  "agent_name": "diagprm_interaction",
  "extra_info": {...}  
}

使用方法：
  python recipe/diagprm/data_utils/prepare_diagprm_data.py \
    --input /path/to/merged_train_dataset.jsonl \
    --output_dir ./data \
    --val_ratio 0.05 \
    --kg_path /path/to/master_kg.json
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ── 系统提示（与 diagprm_agent_loop 中保持一致）──────────────────────────────

DOCTOR_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert diagnostic physician conducting a structured medical interview.

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
- `continue` : Evidence insufficient. Ask ONE focused question. Output `<question>`.
- `switch`   : Primary hypothesis ruled out. Switch to new hypothesis. Output `<question>`.
- `diagnose` : Sufficient evidence gathered. Output `<diagnosis>[Disease Name]</diagnosis>`.

## Rules:
1. Ask only ONE question per turn. Never repeat a question.
2. First hypothesis in `<hypothesis_state>` is your primary hypothesis.
3. When diagnosing: `<action>diagnose</action>` + `<diagnosis>Disease Name</diagnosis>`.
4. Maximum {max_turns} turns allowed.
"""

DOCTOR_INITIAL_PROMPT_TEMPLATE = """\
A patient presents with the following chief complaint:

{chief_complaint}

Please start your diagnostic interview."""


def normalize_disease(text: str) -> str:
    """规范化疾病名称（小写、去标点）。"""
    return re.sub(r"[^a-z0-9 \-']", " ", text.lower()).strip()


def extract_chief_complaint(user_content: str) -> str:
    """
    从 ATPO 格式的 user content 中提取主诉（去掉 Problem 和 Options 部分）。
    
    原始格式：
    "A patient presented with ...\nProblem: ...\nOptions: {...}"
    
    DiagPRM 格式（只保留主诉）：
    "A patient presented with ..."
    """
    # 截断到 "Problem:" 之前
    problem_match = re.search(r'\nProblem:', user_content)
    if problem_match:
        return user_content[:problem_match.start()].strip()
    return user_content.strip()


def extract_disease_from_answer_info(answer_info: str) -> str:
    """
    从 answer_info 字符串中提取疾病名称。
    
    answer_info 格式：可能是 "Disease Name" 或 "Description of Disease Name"
    """
    if not answer_info:
        return ""
    # 简单地直接使用 answer_info 作为疾病名
    # 如果需要，可以在这里加入 KG 匹配来验证
    return answer_info.strip()


def build_diagprm_prompt(
    chief_complaint: str,
    max_turns: int = 10,
) -> List[Dict]:
    """构建 DiagPRM 格式的 prompt（系统提示 + 主诉）。"""
    system_content = DOCTOR_SYSTEM_PROMPT_TEMPLATE.format(max_turns=max_turns)
    user_content = DOCTOR_INITIAL_PROMPT_TEMPLATE.format(chief_complaint=chief_complaint)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def process_record(
    record: Dict,
    max_turns: int = 10,
    kg: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    将单条数据转换为 DiagPRM 格式。
    支持两种输入格式：
      - ATPO MCQA 格式 (data_source=mcqa)：ground_truth.answer_info 是选项文本
      - KG 对话格式 (data_source=kg)：ground_truth.disease 是规范化疾病名，
        ground_truth.symptoms_pool 是症状权重字典

    Returns:
        转换后的记录，或 None（如果数据无效）
    """
    # 提取字段
    prompt = record.get("prompt", [])
    ground_truth = record.get("ground_truth", {})
    extra_info = record.get("extra_info", {})
    data_source = record.get("data_source", "unknown")

    if not prompt:
        return None

    # 获取 user 消息中的患者主诉
    user_content = ""
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    if not user_content:
        return None

    chief_complaint = extract_chief_complaint(user_content)
    if len(chief_complaint) < 20:  # 过短的主诉可能是无效数据
        return None

    # 提取 ground truth 疾病名
    # KG 格式：ground_truth.disease 已经是规范化疾病名
    # MCQA 格式：ground_truth.answer_info 是选项文本（可能不是疾病名）
    if isinstance(ground_truth, dict):
        answer = ground_truth.get("answer", "")
        # 优先使用 KG 格式的 disease 字段（已规范化）
        if ground_truth.get("disease"):
            disease_norm = normalize_disease(ground_truth["disease"])
            disease_raw = ground_truth.get("answer_info", ground_truth["disease"])
        else:
            # MCQA 格式：用 answer_info 作为疾病名
            answer_info = ground_truth.get("answer_info", "")
            disease_raw = answer_info if answer_info else answer
            disease_norm = normalize_disease(disease_raw)
    else:
        answer = str(ground_truth)
        disease_raw = answer
        disease_norm = normalize_disease(disease_raw)

    # 如果提供了 KG，验证疾病是否在 KG 中（可选过滤）
    has_kg_coverage = kg is not None and disease_norm in kg
    if kg is not None and not has_kg_coverage:
        # 疾病不在 KG 中：跳过 KG-driven reward 但仍可训练
        # 这些数据仍然有 r_hyp 和 r_diag，只是没有 delta_kg
        pass

    # KG 格式：symptoms_pool 包含完整的症状权重字典
    symptoms_pool = ground_truth.get("symptoms_pool", {})

    # 构建 reward_model 字段（先获取 atomic_facts，再决定 chief_complaint 内容）
    atomic_facts = extra_info.get("atomic_facts", [])

    # KG 格式：symptoms_pool 里 top-N 症状补充到 atomic_facts（如果 atomic_facts 为空）
    if not atomic_facts and symptoms_pool:
        sorted_syms = sorted(symptoms_pool.items(), key=lambda x: -x[1])
        atomic_facts = [
            f"{i+1}. The patient has {s}."
            for i, (s, _) in enumerate(sorted_syms[:3])
        ]

    # 增强 chief complaint：把初始揭示的症状加入主诉，让模型有真实起始信息
    # KG 格式 chief_complaint 是通用模板，加入 atomic_facts 内容让它有实质信息
    if atomic_facts and len(chief_complaint) < 100:
        # chief_complaint 是通用模板（"A patient presents to the clinic with..."）
        # 在后面加上已知的初始症状
        facts_text = " ".join(
            f.split(". ", 1)[1] if ". " in f else f
            for f in atomic_facts[:2]
        )
        chief_complaint = chief_complaint + " The patient reports: " + facts_text

    # 构建 DiagPRM prompt
    diagprm_prompt = build_diagprm_prompt(chief_complaint, max_turns=max_turns)

    reward_model_info = {
        "ground_truth": {
            "disease": disease_norm,         # 规范化疾病名（给 KG 查询用）
            "disease_raw": disease_raw,      # 原始字符串（logging）
            "answer": answer,                # MCQA 选项（兼容 ATPO 评估）
            "atomic_facts": atomic_facts,    # 患者 simulator 用
            # KG 格式额外字段：完整症状池（供 patient simulator 按需查询）
            "symptoms_pool": symptoms_pool,
        }
    }

    return {
        "prompt": diagprm_prompt,
        "reward_model": reward_model_info,
        "data_source": f"diagprm_{data_source}",
        "agent_name": "diagprm_interaction",
        "extra_info": {
            "original_data_source": data_source,
            "has_kg_coverage": has_kg_coverage,
            "n_atomic_facts": len(atomic_facts),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DiagPRM training data")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/liuhaoyu/iclr_2027/diagprm/diagprm_dataset/kg_train_dataset.jsonl",
        help="Input JSONL file path. Supports KG format (kg_train_dataset.jsonl, recommended) "
             "or ATPO MCQA format (merged_train_dataset.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/liuhaoyu/iclr_2027/diagprm/diagprm_dataset",
        help="Output directory for parquet files (diagprm_train.parquet / diagprm_val.parquet)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use as validation set",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum number of dialogue turns",
    )
    parser.add_argument(
        "--kg_path",
        type=str,
        default="",
        help="Path to master_kg.json (optional, for validation only)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)",
    )
    parser.add_argument(
        "--filter_no_kg",
        action="store_true",
        default=False,
        help="Filter out records where the GT disease is not in KG",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 可选：加载 KG 用于验证
    kg = None
    if args.kg_path:
        print(f"Loading KG from {args.kg_path}...")
        with open(args.kg_path) as f:
            raw_kg = json.load(f)
        kg = {}
        for d, syms in raw_kg.items():
            d_norm = normalize_disease(d)
            if isinstance(syms, dict):
                kg[d_norm] = syms
            elif isinstance(syms, list):
                kg[d_norm] = {s: 1.0 for s in syms}
        print(f"KG loaded: {len(kg):,} diseases")

    # 加载并处理数据
    print(f"Reading input: {args.input}")
    records = []
    skipped = 0
    kg_covered = 0

    with open(args.input) as f:
        for i, line in enumerate(tqdm(f, desc="Processing")):
            if args.max_samples and i >= args.max_samples:
                break
            try:
                raw = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            processed = process_record(raw, max_turns=args.max_turns, kg=kg)
            if processed is None:
                skipped += 1
                continue

            # 可选：过滤掉 KG 中没有的疾病
            if args.filter_no_kg and kg is not None:
                if not processed["extra_info"]["has_kg_coverage"]:
                    skipped += 1
                    continue

            if processed["extra_info"]["has_kg_coverage"]:
                kg_covered += 1

            records.append(processed)

    print(f"\nProcessed: {len(records):,} records")
    print(f"Skipped: {skipped:,} records")
    if kg:
        print(f"KG coverage: {kg_covered:,} / {len(records):,} ({100*kg_covered/max(len(records),1):.1f}%)")

    # 分割训练/验证集
    import random
    random.seed(42)
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"\nTrain: {len(train_records):,}, Val: {len(val_records):,}")

    # 保存为 parquet
    def save_parquet(data: List[Dict], path: str):
        """将数据保存为 parquet 格式（verl 兼容）。"""
        # 将 nested dict/list 字段序列化为 JSON 字符串
        rows = []
        for rec in data:
            row = {
                # verl 需要 prompt 以 list[dict] 形式存储
                "prompt": json.dumps(rec["prompt"], ensure_ascii=False),
                "reward_model": json.dumps(rec["reward_model"], ensure_ascii=False),
                "data_source": rec["data_source"],
                "agent_name": rec["agent_name"],
                "extra_info": json.dumps(rec["extra_info"], ensure_ascii=False),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        print(f"Saved {len(df):,} records to {path}")

    train_path = os.path.join(args.output_dir, "diagprm_train.parquet")
    val_path = os.path.join(args.output_dir, "diagprm_val.parquet")

    save_parquet(train_records, train_path)
    save_parquet(val_records, val_path)

    # 打印样例
    print("\n--- Sample Record ---")
    sample = train_records[0]
    print(f"Chief complaint (truncated): {sample['prompt'][1]['content'][:200]}...")
    gt = sample['reward_model']['ground_truth']
    print(f"GT disease: {gt['disease_raw']} -> normalized: {gt['disease']}")
    print(f"n_atomic_facts: {len(gt['atomic_facts'])}")
    print("Done!")


if __name__ == "__main__":
    main()
