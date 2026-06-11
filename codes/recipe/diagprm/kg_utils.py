"""
DiagPRM - KG Utility Module
加载 master_kg.json，提供带权重的 KG 覆盖率计算功能。

KG 格式（master_kg.json）：
{
  "disease_name": {
    "symptom_name": weight_float,   # 带权重格式
    ...
  },
  ...
}
或旧格式：
{
  "disease_name": ["symptom1", "symptom2", ...]
}
"""

import json
import re
import os
from functools import lru_cache
from typing import Dict, Set, Optional, List, Tuple


# ── 全局 KG 单例 ──────────────────────────────────────────────────────────────
_KG: Optional[Dict] = None
_KG_PATH: str = ""


def load_kg(kg_path: str) -> Dict:
    """加载 master_kg.json，返回 {disease: {symptom: weight}} 格式。
    
    兼容两种存储格式：
    1. 带权重：{"disease": {"symptom": 0.8, ...}}
    2. 旧列表：{"disease": ["symptom1", "symptom2", ...]}
    """
    global _KG, _KG_PATH
    if _KG is not None and _KG_PATH == kg_path:
        return _KG

    with open(kg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    kg = {}
    for disease, symptoms in raw.items():
        disease_norm = _normalize(disease)
        if isinstance(symptoms, dict):
            # 带权重格式
            kg[disease_norm] = {_normalize(s): float(w) for s, w in symptoms.items() if s and w is not None}
        elif isinstance(symptoms, list):
            # 旧列表格式：权重统一为 1.0
            kg[disease_norm] = {_normalize(s): 1.0 for s in symptoms if s}
        else:
            # 跳过非法格式
            continue

    _KG = kg
    _KG_PATH = kg_path
    print(f"[KG] Loaded {len(kg):,} diseases from {kg_path}")
    return _KG


def _normalize(text: str) -> str:
    """统一小写 + 去特殊字符，用于模糊匹配。"""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[^a-z0-9 \-']", " ", text.lower()).strip()


# ── 症状提取 ──────────────────────────────────────────────────────────────────

def extract_symptoms_from_text(text: str, kg: Dict) -> Set[str]:
    """
    从文本中提取与 KG 已知症状匹配的词项。
    使用滑动窗口 n-gram（1-4 词），对 KG 内所有症状做 set intersection。
    
    Returns: 规范化的症状名称集合（KG 中的 key 格式）。
    """
    norm_text = _normalize(text)
    tokens = norm_text.split()

    # 收集文本所有 1-4 词 n-gram
    ngrams: Set[str] = set()
    for n in range(1, 5):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))

    # 提取所有疾病的症状 key 集合（全 KG 症状池）
    all_symptoms = _get_all_symptoms(kg)
    return ngrams & all_symptoms


@lru_cache(maxsize=1)
def _get_all_symptoms(kg_frozenset) -> frozenset:
    """返回 KG 中所有症状名（规范化）的集合，带缓存。"""
    symptoms = set()
    for sym_dict in kg_frozenset:
        symptoms.update(sym_dict.keys())
    return frozenset(symptoms)


def get_all_symptoms_from_kg(kg: Dict) -> frozenset:
    """获取 KG 所有症状集合（带缓存包装）。"""
    # 把 kg 转成可哈希的结构供 lru_cache 使用
    sym_items = tuple(frozenset(v.items()) for v in kg.values())
    return _get_all_symptoms(sym_items)


# ── 覆盖率计算 ─────────────────────────────────────────────────────────────────

def compute_kg_coverage(
    collected_symptoms: Set[str],
    disease: str,
    kg: Dict,
    weighted: bool = True,
) -> float:
    """
    计算已收集症状对某疾病的 KG 覆盖率。

    Args:
        collected_symptoms: 已规范化的症状名称集合（KG key 格式）
        disease: 规范化的疾病名称
        kg: 已加载的知识图谱 {disease: {symptom: weight}}
        weighted: True = 使用 IDF 权重；False = 简单 intersection/union

    Returns:
        float in [0, 1]
    """
    disease_norm = _normalize(disease)
    if disease_norm not in kg or not kg[disease_norm]:
        return 0.0

    sym_dict = kg[disease_norm]  # {symptom: weight}

    if weighted:
        total_weight = sum(sym_dict.values())
        if total_weight == 0:
            return 0.0
        covered_weight = sum(
            w for s, w in sym_dict.items() if s in collected_symptoms
        )
        return covered_weight / total_weight
    else:
        total = len(sym_dict)
        if total == 0:
            return 0.0
        covered = sum(1 for s in sym_dict if s in collected_symptoms)
        return covered / total


def compute_kg_coverage_delta(
    prev_symptoms: Set[str],
    curr_symptoms: Set[str],
    disease: str,
    kg: Dict,
    beta: float = 1.0,
    weighted: bool = True,
) -> float:
    """
    计算一轮前后的 KG 覆盖率差分 Δ_k^kg。

    Returns:
        float: beta * (coverage(curr) - coverage(prev))
    """
    prev_cov = compute_kg_coverage(prev_symptoms, disease, kg, weighted=weighted)
    curr_cov = compute_kg_coverage(curr_symptoms, disease, kg, weighted=weighted)
    return beta * (curr_cov - prev_cov)


# ── 假设解析工具 ──────────────────────────────────────────────────────────────

def parse_hypothesis_state(model_output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从模型输出中解析 <hypothesis_state> 标签，提取：
    - primary_hypothesis: 第一个 hypothesis name（即主假设）
    - action: <action> 标签内容（continue / switch / diagnose）

    Returns:
        (primary_hypothesis_str | None, action_str | None)
    """
    primary = None
    action = None

    # 解析 primary hypothesis（第一个 <hypothesis name="...">）
    hyp_match = re.search(
        r'<hypothesis[^>]+name\s*=\s*["\']([^"\']+)["\']',
        model_output,
        re.IGNORECASE,
    )
    if hyp_match:
        primary = _normalize(hyp_match.group(1))

    # 解析 action
    action_match = re.search(
        r'<action>\s*(continue|switch|diagnose)\s*</action>',
        model_output,
        re.IGNORECASE | re.DOTALL,
    )
    if action_match:
        action = action_match.group(1).strip().lower()

    return primary, action


def parse_final_diagnosis(model_output: str) -> Optional[str]:
    """
    解析 <diagnosis>...</diagnosis> 标签内容作为最终诊断。
    如果没有该标签，尝试解析 "Final Answer:" 格式（ATPO 兼容）。
    """
    # DiagPRM 格式
    diag_match = re.search(
        r'<diagnosis>\s*(.*?)\s*</diagnosis>',
        model_output,
        re.IGNORECASE | re.DOTALL,
    )
    if diag_match:
        return _normalize(diag_match.group(1).strip())

    # ATPO 兼容格式
    fa_match = re.search(
        r'Final Answer:\s*([A-Z])',
        model_output,
        re.IGNORECASE,
    )
    if fa_match:
        return fa_match.group(1).strip().upper()

    return None


def parse_question(model_output: str) -> Optional[str]:
    """解析 <question>...</question> 内容。"""
    q_match = re.search(
        r'<question>\s*(.*?)\s*</question>',
        model_output,
        re.IGNORECASE | re.DOTALL,
    )
    if q_match:
        return q_match.group(1).strip()
    # 兼容 "Question: xxx" 格式
    q_match2 = re.search(r'Question:\s*(.+)', model_output, re.IGNORECASE)
    if q_match2:
        return q_match2.group(1).strip()
    return None


# ── 诊断匹配 ─────────────────────────────────────────────────────────────────

def is_diagnosis_match(predicted: str, ground_truth: str, kg: Dict) -> bool:
    """
    判断预测诊断是否匹配 ground truth。
    支持：
    1. 精确规范化匹配
    2. KG 中疾病名模糊包含匹配
    """
    pred_norm = _normalize(predicted)
    gt_norm = _normalize(ground_truth)

    if pred_norm == gt_norm:
        return True

    # 允许疾病名互相包含（如 "type 2 diabetes mellitus" 匹配 "diabetes mellitus"）
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True

    # 检查 KG 中是否有别名（两者都在 KG 中且高度相似）
    gt_in_kg = gt_norm in kg
    pred_in_kg = pred_norm in kg
    if gt_in_kg and pred_in_kg:
        # 两者都是合法疾病但不同：返回 False
        return False

    return False


def fuzzy_match_disease(disease_str: str, kg: Dict, top_k: int = 1) -> List[Tuple[str, float]]:
    """
    从 KG 中找最相近的疾病名（用于 ground truth 对齐）。
    Returns: [(disease_name, score), ...]，按 score 降序。
    """
    norm = _normalize(disease_str)
    tokens = set(norm.split())

    scores = []
    for d in kg:
        d_tokens = set(d.split())
        if not d_tokens:
            continue
        # Jaccard similarity
        intersection = len(tokens & d_tokens)
        union = len(tokens | d_tokens)
        score = intersection / union if union > 0 else 0.0
        if score > 0:
            scores.append((d, score))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]
