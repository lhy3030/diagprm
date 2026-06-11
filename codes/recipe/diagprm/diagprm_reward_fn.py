"""
DiagPRM - Turn-level Reward Function

每一轮的 reward 由以下分量组成：
  r(k) = Δ_k^kg  +  r_hyp(k)  +  r_switch(k)  +  r_diag（仅确诊轮）

各分量含义：
  Δ_k^kg      : KG 覆盖率差分（dense，每轮都有）
  r_hyp(k)    : 假设正确性奖励（每轮，感知主假设是否与 GT 一致）
  r_switch(k) : 假设切换修正（稀疏，仅 <action>switch</action> 时触发）
  r_diag      : 确诊奖励（仅 <action>diagnose</action> 时触发）

全部奖励来源于 KG 静态查询 + ground truth label，无需任何模型前向传播。
"""

import re
from typing import Dict, List, Optional, Set, Tuple

import torch

from recipe.diagprm.kg_utils import (
    _normalize,
    compute_kg_coverage_delta,
    extract_symptoms_from_text,
    is_diagnosis_match,
    parse_final_diagnosis,
    parse_hypothesis_state,
    parse_question,
)

# ──────────────────────────────────────────────────────────────────────────────
# 格式解析工具（复用 ATPO 的 response_mask 解析逻辑）
# ──────────────────────────────────────────────────────────────────────────────

def parse_turns_from_response_mask(
    response_mask: torch.Tensor,
    response_ids: torch.Tensor,
    tokenizer,
) -> List[Dict]:
    """
    从 response_mask 中解析每一轮的 token 区间。
    response_mask：1 = 模型输出，0 = 患者输入（human turn）。
    
    Returns:
        List of {turn_id, start_position, end_position, response, is_final_turn, length}
    """
    turns = []
    mask_list = response_mask.tolist()
    i = 0
    turn_id = 0

    while i < len(mask_list):
        if mask_list[i] == 1:
            start_pos = i
            while i < len(mask_list) and mask_list[i] == 1:
                i += 1
            end_pos = i - 1

            turn_ids = response_ids[start_pos:i]
            response_text = tokenizer.decode(turn_ids, skip_special_tokens=True)

            is_final = True
            for j in range(i, len(mask_list)):
                if mask_list[j] == 1:
                    is_final = False
                    break

            turns.append({
                "turn_id": turn_id,
                "start_position": start_pos,
                "end_position": end_pos,
                "response": response_text,
                "is_final_turn": is_final,
                "length": end_pos - start_pos + 1,
            })
            turn_id += 1
        else:
            i += 1

    return turns


def extract_human_responses(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
) -> List[str]:
    """提取对话中患者（human）的回复文本列表。"""
    human_responses = []
    mask_list = response_mask.tolist()
    i = 0

    while i < len(mask_list):
        if mask_list[i] == 0:
            start_pos = i
            while i < len(mask_list) and mask_list[i] == 0:
                i += 1
            # 最后的 padding 段（后面没有更多 model token）不算 human response
            if i < len(mask_list):
                human_ids = response_ids[start_pos:i]
                human_text = tokenizer.decode(human_ids, skip_special_tokens=True)
                human_text = human_text.replace("user\n", "").replace("\nassistant", "").strip()
                human_responses.append(human_text)
        else:
            i += 1

    return human_responses


# ──────────────────────────────────────────────────────────────────────────────
# 单轮 reward 计算
# ──────────────────────────────────────────────────────────────────────────────

def calculate_turn_reward(
    model_response: str,
    human_response: str,              # 患者对本轮问题的回答（用于更新症状集合）
    prev_collected_symptoms: Set[str],
    curr_collected_symptoms: Set[str],
    prev_hypothesis: Optional[str],   # 上一轮 primary hypothesis（规范化）
    curr_hypothesis: Optional[str],   # 本轮 primary hypothesis（规范化）
    action: Optional[str],            # continue / switch / diagnose
    ground_truth_disease: str,        # 规范化的 GT 疾病名
    kg: Dict,
    is_final_turn: bool,
    # ---------- 奖励系数 ----------
    beta: float = 1.0,     # KG 覆盖率差分系数
    gamma1: float = 0.3,   # 假设正确性奖励系数
    lam: float = 0.5,      # 切换修正系数
    r_max: float = 2.0,    # 最大确诊奖励
    tau: float = 0.5,      # 最低 KG 覆盖率阈值（过早确诊惩罚）
    format_score: float = 0.1,
    weighted: bool = True,
) -> Dict:
    """
    计算单轮的完整 DiagPRM reward。

    Returns:
        dict with keys:
          process_reward, outcome_reward,
          delta_kg, r_hyp, r_switch, r_diag, format_reward,
          details (各字段的细节)
    """
    gt_norm = _normalize(ground_truth_disease)
    details = {
        "action": action,
        "curr_hypothesis": curr_hypothesis,
        "prev_hypothesis": prev_hypothesis,
        "gt_disease": gt_norm,
        "has_valid_format": False,
        "delta_kg": 0.0,
        "r_hyp": 0.0,
        "r_switch": 0.0,
        "r_diag": 0.0,
        "format_reward": 0.0,
        "coverage_before": 0.0,
        "coverage_after": 0.0,
        "n_symptoms_collected": len(curr_collected_symptoms),
        "is_correct_diagnosis": False,
        "premature_diagnosis": False,
    }

    # ── 1. 格式检查 ──────────────────────────────────────────────────────────
    # 检查是否包含 <think>...</think> 且有合法 action
    think_match = re.search(r'<think>.*?</think>', model_response, re.DOTALL | re.IGNORECASE)
    action_match = action is not None
    if think_match and action_match:
        details["has_valid_format"] = True
        format_reward = format_score
    else:
        format_reward = 0.0
    details["format_reward"] = format_reward

    # ── 2. KG 覆盖率差分 Δ_k^kg ─────────────────────────────────────────────
    from recipe.diagprm.kg_utils import compute_kg_coverage

    cov_before = compute_kg_coverage(prev_collected_symptoms, gt_norm, kg, weighted=weighted)
    cov_after = compute_kg_coverage(curr_collected_symptoms, gt_norm, kg, weighted=weighted)
    delta_kg = beta * (cov_after - cov_before)
    details["coverage_before"] = cov_before
    details["coverage_after"] = cov_after
    details["delta_kg"] = delta_kg

    # ── 3. 假设正确性奖励 r_hyp ──────────────────────────────────────────────
    r_hyp = 0.0
    if curr_hypothesis is not None:
        # 比较主假设与 GT（允许模糊包含匹配）
        if gt_norm and (gt_norm in curr_hypothesis or curr_hypothesis in gt_norm):
            r_hyp = gamma1
        else:
            r_hyp = -gamma1
    details["r_hyp"] = r_hyp

    # ── 4. 假设切换修正 r_switch ─────────────────────────────────────────────
    r_switch = 0.0
    if action == "switch" and prev_hypothesis is not None and curr_hypothesis is not None:
        prev_correct = gt_norm and (gt_norm in prev_hypothesis or prev_hypothesis in gt_norm)
        curr_correct = gt_norm and (gt_norm in curr_hypothesis or curr_hypothesis in gt_norm)
        if not prev_correct and curr_correct:
            # 从错到对：好的切换
            r_switch = lam
        elif prev_correct and not curr_correct:
            # 从对到错：坏的切换
            r_switch = -lam
        # 错到错或调整细节：0
    details["r_switch"] = r_switch

    # ── 5. 确诊奖励 r_diag（仅 is_final_turn 时） ────────────────────────────
    r_diag = 0.0
    if is_final_turn:
        if action == "diagnose":
            predicted = parse_final_diagnosis(model_response)
            if predicted is not None and is_diagnosis_match(predicted, gt_norm, kg):
                # 检查覆盖率是否足够（避免过早确诊奖励）
                if cov_after >= tau:
                    r_diag = r_max
                    details["is_correct_diagnosis"] = True
                else:
                    # 正确诊断但证据不足 → 减半奖励
                    r_diag = r_max * 0.5
                    details["is_correct_diagnosis"] = True
                    details["premature_diagnosis"] = True
            else:
                # 错误诊断
                r_diag = 0.0
        else:
            # 达到最大轮次但未确诊 → 惩罚
            r_diag = -1.0
    details["r_diag"] = r_diag

    # ── 6. 合并 ──────────────────────────────────────────────────────────────
    process_reward = format_reward + delta_kg + r_hyp + r_switch
    outcome_reward = r_diag  # 仅最终轮有效

    return {
        "process_reward": float(process_reward),
        "outcome_reward": float(outcome_reward),
        "format_reward": float(format_reward),
        "delta_kg": float(delta_kg),
        "r_hyp": float(r_hyp),
        "r_switch": float(r_switch),
        "r_diag": float(r_diag),
        "details": details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Episode 级别症状累积（从对话历史中提取已确认症状集合）
# ──────────────────────────────────────────────────────────────────────────────

def update_collected_symptoms(
    prev_symptoms: Set[str],
    human_response: str,
    model_question: str,
    kg: Dict,
) -> Set[str]:
    """
    根据患者的本轮回答 + 医生的问题，更新已收集症状集合。

    策略：
    1. 从 human_response 中提取被患者确认的症状（"yes" / "no" 判断）
    2. 从 model_question + human_response 联合文本中提取症状 n-gram

    Returns:
        updated symptom set（copy，不修改 prev_symptoms）
    """
    new_symptoms = set(prev_symptoms)

    # 拒绝模式：患者否认症状
    deny_patterns = re.compile(
        r'\b(no|not|deny|denies|denied|absence|absent|never|without)\b',
        re.IGNORECASE,
    )

    # 组合文本（问题 + 回答）
    combined = (model_question or "") + " " + (human_response or "")
    # 从文本中提取所有匹配 KG 的症状 n-gram
    candidates = extract_symptoms_from_text(combined, kg)

    for sym in candidates:
        # 简单的肯定/否定判断：如果 human_response 中包含 deny 词且症状出现，跳过
        # 更精细的方案：NLI；这里用简化规则
        sym_in_response = sym in _normalize(human_response or "")
        if sym_in_response:
            # 检查否认
            context_before = _get_context_before(sym, human_response)
            if not deny_patterns.search(context_before):
                new_symptoms.add(sym)

    return new_symptoms


def _get_context_before(target: str, text: str, window: int = 5) -> str:
    """获取 target 词之前 window 个词的文本（用于否认检测）。"""
    norm = _normalize(text)
    idx = norm.find(target)
    if idx == -1:
        return ""
    words = norm[:idx].split()
    return " ".join(words[-window:])


# ──────────────────────────────────────────────────────────────────────────────
# 完整轨迹 reward 计算（供 reward manager 调用）
# ──────────────────────────────────────────────────────────────────────────────

def compute_episode_rewards(
    turns_info: List[Dict],           # parse_turns_from_response_mask 的输出
    human_responses: List[str],       # extract_human_responses 的输出
    ground_truth: str,                # ground truth 疾病名
    kg: Dict,                         # master_kg
    reward_params: Dict,              # 奖励系数字典
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    对整条轨迹计算每轮的 process_reward 和 outcome_reward。

    Returns:
        process_rewards: List[float]（每轮在 end_position 处的 process reward）
        outcome_rewards: List[float]（非 0 值仅出现在 final turn）
        details_list: List[Dict]
    """
    gt_norm = _normalize(ground_truth)
    collected_symptoms: Set[str] = set()

    process_rewards = []
    outcome_rewards = []
    details_list = []

    prev_hypothesis: Optional[str] = None

    for turn_idx, turn_info in enumerate(turns_info):
        model_response = turn_info["response"]
        is_final = turn_info["is_final_turn"]

        # 获取患者回答（最后一轮通常没有患者回答）
        human_resp = human_responses[turn_idx] if turn_idx < len(human_responses) else ""

        # 解析本轮的 action 和 hypothesis
        curr_hypothesis, action = parse_hypothesis_state(model_response)
        question = parse_question(model_response)

        # 更新症状集合：把患者回答里确认的症状加入 collected
        prev_symptoms = set(collected_symptoms)
        if human_resp:
            collected_symptoms = update_collected_symptoms(
                collected_symptoms, human_resp, question or "", kg
            )

        # 计算本轮 reward
        turn_result = calculate_turn_reward(
            model_response=model_response,
            human_response=human_resp,
            prev_collected_symptoms=prev_symptoms,
            curr_collected_symptoms=collected_symptoms,
            prev_hypothesis=prev_hypothesis,
            curr_hypothesis=curr_hypothesis,
            action=action,
            ground_truth_disease=gt_norm,
            kg=kg,
            is_final_turn=is_final,
            **reward_params,
        )

        process_rewards.append(turn_result["process_reward"])
        outcome_rewards.append(turn_result["outcome_reward"])
        details_list.append(turn_result["details"])

        # 更新上一轮 hypothesis
        prev_hypothesis = curr_hypothesis

    return process_rewards, outcome_rewards, details_list
