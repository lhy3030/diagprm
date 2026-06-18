"""
DiagPRM - Turn-level Reward Function

每一轮的 reward 由以下分量组成：
  r(k) = Δ_k^kg  +  r_hyp(k)  +  r_diag（仅确诊轮）

各分量含义：
  Δ_k^kg   : KG 覆盖率差分（dense，每轮都有）
  r_hyp(k) : 假设正确性奖励（每轮，感知主假设是否与 GT 一致）
  r_diag   : 确诊奖励（仅 <action>diagnose</action> 时触发）

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
    action: Optional[str],            # continue / diagnose
    ground_truth_disease: str,        # 规范化的 GT 疾病名
    kg: Dict,
    is_final_turn: bool,
    # ---------- 奖励系数 ----------
    beta: float = 1.0,        # KG 覆盖率差分系数
    gamma1: float = 0.3,      # 假设正确性奖励系数
    turn_coef: float = 1.0,   # Turn 奖励总系数（r_turn 乘以该系数后加 r_diag）
    r_max: float = 2.0,       # 最大确诊奖励
    tau: float = 0.5,         # 最低 KG 覆盖率阈值（过早确诊惩罚）
    format_score: float = 0.1,
    weighted: bool = True,
    evidence_gated_hyp: bool = True,
    wrong_hyp_penalty_scale: float = 0.5,
    r_wrong_diag: float = -1.0,
    r_timeout: float = -1.0,
) -> Dict:
    """
    计算单轮的完整 DiagPRM reward。

    奖励结构：
        r(k) = turn_coef * r_turn(k) + r_diag(k)

    其中：
        r_turn(k) = format_reward + Δ_k^kg + r_hyp(k)
          - format_reward : 输出格式正确奖励（含 <think> 且有合法 action）
          - Δ_k^kg        : KG 覆盖率差分（本轮症状收集的增量贡献）
          - r_hyp(k)      : 主假设是否指向 GT（每轮都有，假设更新隐含在 continue 中）

        r_diag(k) : 确诊奖励（仅 is_final_turn 时非零）
          - 正确诊断且覆盖率充分 : +r_max
          - 正确诊断但覆盖率不足 : +r_max * 0.5（过早确诊）
          - 诊断错误              : r_wrong_diag
          - 超时未确诊            : r_timeout

    Returns:
        dict with keys:
          total_reward, r_turn, r_diag,
          format_reward, delta_kg, r_hyp,
          details
    """
    gt_norm = _normalize(ground_truth_disease)
    details = {
        "action": action,
        "curr_hypothesis": curr_hypothesis,
        "prev_hypothesis": prev_hypothesis,
        "gt_disease": gt_norm,
        "has_valid_format": False,
        "format_reward": 0.0,
        "delta_kg": 0.0,
        "r_hyp": 0.0,
        "r_turn": 0.0,
        "r_diag": 0.0,
        "coverage_before": 0.0,
        "coverage_after": 0.0,
        "n_symptoms_collected": len(curr_collected_symptoms),
        "is_correct_diagnosis": False,
        "premature_diagnosis": False,
    }

    # ── 1. 格式检查 ──────────────────────────────────────────────────────────
    # 检查是否包含合法 JSON（有 thought/hypothesis/action 字段）
    from recipe.diagprm.kg_utils import _parse_json_from_output
    parsed_json = _parse_json_from_output(model_response)
    json_valid = bool(
        parsed_json
        and parsed_json.get("action") in ("continue", "diagnose", "switch")
        and ("hypothesis" in parsed_json or "diagnosis" in parsed_json)
    )
    action_match = action is not None
    if json_valid and action_match:
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
    # 每轮均感知主假设是否为 GT，假设切换隐含在 continue 动作内的 hypothesis_state 更新中
    r_hyp = 0.0
    if curr_hypothesis is not None:
        # 比较主假设与 GT（允许模糊包含匹配）
        hyp_match = bool(gt_norm and (gt_norm in curr_hypothesis or curr_hypothesis in gt_norm))
        if evidence_gated_hyp:
            # 正确假设只有在本轮带来新增 GT evidence 时才奖励，避免早猜刷分。
            if hyp_match and delta_kg > 0:
                r_hyp = gamma1
            elif (not hyp_match) and action == "continue":
                r_hyp = -gamma1 * wrong_hyp_penalty_scale
        else:
            r_hyp = gamma1 if hyp_match else -gamma1
    details["r_hyp"] = r_hyp

    # ── 4. 合并 Turn 奖励 ────────────────────────────────────────────────────
    # r_turn 封装了本轮问诊质量的全部即时信号（去掉 r_switch）
    r_turn = format_reward + delta_kg + r_hyp
    details["r_turn"] = r_turn

    # ── 5. 确诊奖励 r_diag（仅 is_final_turn 时） ────────────────────────────
    # 注意：action 到达此处时已经过归一化，只有 "continue" / "diagnose" / None 三种值
    # （旧的 "switch" 在 parse_hypothesis_state 中已转换为 "continue"）
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
                r_diag = r_wrong_diag
        else:
            # 达到最大轮次仍是 continue / None（未触发 diagnose）→ 超时惩罚
            r_diag = r_timeout
    details["r_diag"] = r_diag

    # ── 6. 最终合并：r(k) = turn_coef * r_turn + r_diag ─────────────────────
    total_reward = turn_coef * r_turn + r_diag

    return {
        "total_reward": float(total_reward),      # r(k) 最终标量
        "r_turn": float(r_turn),                  # 本轮即时信号合计
        "r_diag": float(r_diag),                  # 确诊奖励（非最终轮为 0）
        # 细项（供 metrics 展示）
        "format_reward": float(format_reward),
        "delta_kg": float(delta_kg),
        "r_hyp": float(r_hyp),
        # 兼容旧接口（process_reward / outcome_reward 字段保留）
        "process_reward": float(turn_coef * r_turn),
        "outcome_reward": float(r_diag),
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


# ──────────────────────────────────────────────────────────────────────────────
# 方案A：从 dialogue_history 直接计算 episode reward（推荐路径）
# ──────────────────────────────────────────────────────────────────────────────

def compute_episode_rewards_from_history(
    dialogue_history: List[Dict],   # Agent Loop 传入的 [{turn_id, doctor_response, patient_answer, patient_fact_id, ...}]
    ground_truth: str,              # ground truth 疾病名
    kg: Dict,                       # master_kg
    reward_params: Dict,            # 奖励系数字典
    initial_symptoms: Optional[List[str]] = None,
    fact_id_to_text: Optional[Dict[str, str]] = None,
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    从 dialogue_history 直接计算每轮 reward，无需解析 token mask。

    hidden patient_fact_id 字段：
      - fact id（如 "F003"）→ 通过 fact_id_to_text 精确解析为 KG 症状
      - "unknown" → 无效提问，本轮 delta_kg = 0，症状集合不更新
      - 旧数据的 patient_fact 字符串仍作为 fallback 支持

    奖励结构（每轮）：
        r(k) = turn_coef * r_turn(k) + r_diag(k)
        r_turn(k) = format_reward + Δ_k^kg + r_hyp(k)
    """
    from recipe.diagprm.kg_utils import _normalize, extract_symptoms_from_text

    gt_norm = _normalize(ground_truth)
    collected_symptoms: Set[str] = set()
    disease_syms = kg.get(gt_norm, {})
    for sym in initial_symptoms or []:
        norm_sym = _normalize(sym)
        if norm_sym in disease_syms:
            collected_symptoms.add(norm_sym)

    total_rewards = []
    r_diag_list = []
    details_list = []
    fact_id_to_text = fact_id_to_text or {}

    prev_hypothesis: Optional[str] = None
    _params = {
        k: v for k, v in reward_params.items()
        if k not in {"lam", "unknown_penalty", "duplicate_penalty"}
    }
    unknown_penalty = float(reward_params.get("unknown_penalty", -0.05))
    duplicate_penalty = float(reward_params.get("duplicate_penalty", -0.05))

    for turn_idx, entry in enumerate(dialogue_history):
        doctor_response = entry.get("doctor_response", "")
        patient_fact_id = str(entry.get("patient_fact_id", "unknown") or "unknown").strip()
        patient_fact = ""
        if patient_fact_id and patient_fact_id.lower() != "unknown":
            patient_fact = fact_id_to_text.get(patient_fact_id, "")
        if not patient_fact:
            # Hidden debug field from new agent loop, then legacy verbatim field.
            patient_fact = (
                entry.get("patient_fact_text", "")
                or entry.get("patient_fact", "unknown")
                or "unknown"
            )
        is_final        = entry.get("is_final", turn_idx == len(dialogue_history) - 1)

        # 解析本轮 action 和 hypothesis
        curr_hypothesis, action = parse_hypothesis_state(doctor_response)

        # Update symptom set from patient_fact.
        # patient_fact is now a KG verbatim symptom key (e.g. "chest pain").
        # 1. Try direct normalised lookup in this disease's symptom dict (fastest).
        # 2. Fall back to full KG n-gram extraction (covers edge cases / legacy data).
        prev_symptoms = set(collected_symptoms)
        is_unknown_fact = patient_fact.lower() == "unknown" or not patient_fact.strip()
        is_duplicate_fact = False
        if patient_fact.lower() != "unknown" and patient_fact.strip():
            norm_fact = _normalize(patient_fact)
            is_duplicate_fact = norm_fact in collected_symptoms
            if norm_fact in disease_syms:
                # Direct hit: verbatim KG key for this disease
                collected_symptoms.add(norm_fact)
            else:
                # Fallback: n-gram match against full KG symptom pool
                matched = extract_symptoms_from_text(patient_fact, kg)
                collected_symptoms.update(matched)
                # Last resort: add the normalised string as-is
                if not matched and norm_fact:
                    collected_symptoms.add(norm_fact)

        # 计算本轮 reward
        turn_result = calculate_turn_reward(
            model_response=doctor_response,
            human_response=entry.get("patient_answer", ""),
            prev_collected_symptoms=prev_symptoms,
            curr_collected_symptoms=collected_symptoms,
            prev_hypothesis=prev_hypothesis,
            curr_hypothesis=curr_hypothesis,
            action=action,
            ground_truth_disease=gt_norm,
            kg=kg,
            is_final_turn=is_final,
            **_params,
        )

        if (not is_final) and is_unknown_fact:
            turn_result["total_reward"] += unknown_penalty
            turn_result["details"]["unknown_penalty"] = unknown_penalty
        if (not is_final) and is_duplicate_fact:
            turn_result["total_reward"] += duplicate_penalty
            turn_result["details"]["duplicate_penalty"] = duplicate_penalty
        turn_result["details"]["patient_fact_id"] = patient_fact_id
        turn_result["details"]["patient_fact"] = patient_fact if patient_fact != "unknown" else ""

        total_rewards.append(turn_result["total_reward"])
        r_diag_list.append(turn_result["r_diag"])
        details_list.append(turn_result["details"])
        prev_hypothesis = curr_hypothesis

    return total_rewards, r_diag_list, details_list


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
    initial_symptoms: Optional[List[str]] = None,
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    对整条轨迹计算每轮的 total_reward（含 turn_coef）和 r_diag。

    奖励结构（每轮）：
        r(k) = turn_coef * r_turn(k) + r_diag(k)
        r_turn(k) = format_reward + Δ_k^kg + r_hyp(k)

    Returns:
        total_rewards : List[float]  每轮 r(k) = turn_coef * r_turn + r_diag
        r_diag_list   : List[float]  每轮的确诊奖励（非最终轮为 0）
        details_list  : List[Dict]
    """
    gt_norm = _normalize(ground_truth)
    collected_symptoms: Set[str] = set()
    disease_syms = kg.get(gt_norm, {})
    for sym in initial_symptoms or []:
        norm_sym = _normalize(sym)
        if norm_sym in disease_syms:
            collected_symptoms.add(norm_sym)

    total_rewards = []
    r_diag_list = []
    details_list = []

    prev_hypothesis: Optional[str] = None
    unknown_penalty = float(reward_params.get("unknown_penalty", -0.05))
    duplicate_penalty = float(reward_params.get("duplicate_penalty", -0.05))

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
        matched_before = set(collected_symptoms)
        if human_resp:
            collected_symptoms = update_collected_symptoms(
                collected_symptoms, human_resp, question or "", kg
            )
        is_unknown_fact = bool(human_resp) and collected_symptoms == prev_symptoms
        is_duplicate_fact = bool(human_resp) and matched_before == collected_symptoms and bool(
            extract_symptoms_from_text((question or "") + " " + human_resp, kg) & matched_before
        )

        # 计算本轮 reward（过滤掉 lam 参数，因为已去掉 r_switch）
        _params = {
            k: v for k, v in reward_params.items()
            if k not in {"lam", "unknown_penalty", "duplicate_penalty"}
        }
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
            **_params,
        )

        if (not is_final) and is_unknown_fact:
            turn_result["total_reward"] += unknown_penalty
            turn_result["details"]["unknown_penalty"] = unknown_penalty
        if (not is_final) and is_duplicate_fact:
            turn_result["total_reward"] += duplicate_penalty
            turn_result["details"]["duplicate_penalty"] = duplicate_penalty

        total_rewards.append(turn_result["total_reward"])
        r_diag_list.append(turn_result["r_diag"])
        details_list.append(turn_result["details"])

        # 更新上一轮 hypothesis
        prev_hypothesis = curr_hypothesis

    return total_rewards, r_diag_list, details_list
