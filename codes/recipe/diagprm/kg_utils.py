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

# KG 症状池缓存（避免每次调用都遍历全 KG）
# key: id(kg)，当 kg 对象未变时直接复用
_SYMPTOMS_CACHE: Dict[int, frozenset] = {}


def get_all_symptoms_from_kg(kg: Dict) -> frozenset:
    """获取 KG 所有症状集合（带 id 缓存，避免 lru_cache 的 unhashable 问题）。"""
    kg_id = id(kg)
    if kg_id not in _SYMPTOMS_CACHE:
        symptoms: Set[str] = set()
        for sym_dict in kg.values():
            symptoms.update(sym_dict.keys())
        _SYMPTOMS_CACHE[kg_id] = frozenset(symptoms)
    return _SYMPTOMS_CACHE[kg_id]


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
    all_symptoms = get_all_symptoms_from_kg(kg)
    return ngrams & all_symptoms


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

def _parse_json_from_output(model_output: str) -> dict:
    """
    从模型输出中提取 JSON 对象。支持三种格式：
      1. ```json ... ``` 代码块
      2. 去掉 <think>...</think> 后的裸 {...}
      3. 直接在整个输出中找 {...}（兜底）
    """
    import json as _json
    # 1. ```json ... ```
    code_block = re.search(r'```json\s*([\s\S]*?)```', model_output, re.IGNORECASE)
    if code_block:
        try:
            result = _json.loads(code_block.group(1).strip())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    # 2. 去掉 <think> 块后找 {...}
    stripped = re.sub(r'<think>[\s\S]*?</think>', '', model_output, flags=re.IGNORECASE)
    brace_match = re.search(r'(\{[\s\S]*\})', stripped)
    if brace_match:
        try:
            result = _json.loads(brace_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    # 3. 对整个输出找 {...}（兜底）
    brace_match2 = re.search(r'(\{[\s\S]*\})', model_output)
    if brace_match2:
        try:
            result = _json.loads(brace_match2.group(1).strip())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return {}


def parse_hypothesis_state(model_output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从模型输出中解析 JSON 格式，提取可选诊断名和动作。

    主方法不再要求 hypothesis 字段；旧 hypothesis/XML 格式只作为 fallback。

    Returns:
        (diagnosis_or_legacy_hypothesis | None, action_str | None)
    """
    # 优先尝试 JSON
    parsed = _parse_json_from_output(model_output)
    primary = None
    action = None

    if parsed:
        hyp = parsed.get("hypothesis") or parsed.get("diagnosis")
        if hyp:
            primary = _normalize(str(hyp))
        raw_action = parsed.get("action", "").strip().lower()
        if raw_action in ("ask", "continue", "switch"):
            action = "ask"
        elif raw_action == "diagnose":
            action = "diagnose"
        return primary, action

    # Fallback: 旧 XML 格式
    hyp_match = re.search(
        r'<hypothesis[^>]+name\s*=\s*["\']([^"\']+)["\']',
        model_output,
        re.IGNORECASE,
    )
    if hyp_match:
        primary = _normalize(hyp_match.group(1))

    action_match = re.search(
        r'<action>\s*(ask|continue|switch|diagnose)\s*</action>',
        model_output,
        re.IGNORECASE | re.DOTALL,
    )
    if action_match:
        raw_action = action_match.group(1).strip().lower()
        action = "ask" if raw_action in ("ask", "continue", "switch") else raw_action

    return primary, action


def parse_final_diagnosis(model_output: str) -> Optional[str]:
    """
    解析最终诊断。
    优先从 JSON 格式读取 "diagnosis" 字段；
    fallback 到旧的 <diagnosis>...</diagnosis> XML 标签；
    再 fallback 到 "Final Answer:" 格式（ATPO 兼容）。
    """
    _PLACEHOLDER_PATTERNS = re.compile(
        r'^\s*\[.*?\]\s*$|^disease\s*name$|^diagnosis\s*here$',
        re.IGNORECASE,
    )

    # 1. 优先 JSON 格式
    parsed = _parse_json_from_output(model_output)
    if parsed:
        diag = parsed.get("diagnosis") or (parsed.get("hypothesis") if parsed.get("action", "").lower() == "diagnose" else None)
        if diag:
            diag_str = str(diag).strip()
            if diag_str and not _PLACEHOLDER_PATTERNS.match(diag_str):
                return _normalize(diag_str)

    # 2. 旧 XML 格式
    all_diag = re.findall(
        r'<diagnosis>\s*(.*?)\s*</diagnosis>',
        model_output,
        re.IGNORECASE | re.DOTALL,
    )
    for raw in reversed(all_diag):
        raw = raw.strip()
        if raw and not _PLACEHOLDER_PATTERNS.match(raw):
            return _normalize(raw)

    # 3. ATPO 兼容格式
    fa_match = re.search(
        r'Final Answer:\s*([A-Z])',
        model_output,
        re.IGNORECASE,
    )
    if fa_match:
        return fa_match.group(1).strip().upper()

    return None


def parse_question(model_output: str) -> Optional[str]:
    """解析 question 字段（JSON 格式，fallback 到 XML）。"""
    # JSON 格式优先
    parsed = _parse_json_from_output(model_output)
    if parsed and parsed.get("question"):
        return str(parsed["question"]).strip()
    # Fallback: 旧 XML
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


def lookup_disease_name(query: str, kg: Dict) -> Optional[str]:
    """
    Doctor Agent 的 KG 工具接口：给定任意疾病名查询词，返回 KG 中
    最匹配的标准疾病名（KG 的 key，经 normalize 后的统一格式）。

    KG 的 key 本身已经过 _normalize()，因此返回值就是供 Doctor 填入
    "diagnosis" 字段的精确字符串，保证与 parse_final_diagnosis + is_diagnosis_match
    中的 normalize 匹配逻辑一致。

    若精确匹配（normalize 后相同）则直接返回；
    否则用 Jaccard 相似度找最近邻；
    若最高分 < 0.2 则返回 None（表示 KG 中无此病）。

    Returns:
        str  — KG 中的标准疾病名（normalized），供 Doctor 直接填入 "diagnosis"
        None — KG 中找不到匹配项
    """
    if not query:
        return None
    norm_query = _normalize(query)

    # 1. 精确匹配（normalize 后）
    if norm_query in kg:
        return norm_query

    # 2. Jaccard 模糊匹配
    matches = fuzzy_match_disease(query, kg, top_k=1)
    if matches and matches[0][1] >= 0.2:
        return matches[0][0]  # fuzzy_match_disease 已返回 normalized key
    return None


def query_diseases_by_symptom(
    symptom: str,
    kg: Dict,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Doctor Agent 的 KG 工具接口：给定一个症状描述，返回 KG 中含有该症状的
    疾病候选列表，按症状权重（该症状在疾病中的诊断重要性）降序排列。

    用法场景：
      - 第 1 轮：患者说了初始症状，Doctor 用该症状查哪些疾病包含它，
        从而形成初始假设候选列表。

    Args:
        symptom : 症状描述字符串（自然语言，会做 normalize + n-gram 匹配）
        kg      : 已加载的知识图谱 {disease_norm: {symptom_norm: weight}}
        top_k   : 返回最多 top_k 个候选疾病

    Returns:
        [(disease_name, weight), ...] 按 weight 降序，disease_name 为 KG normalized key
    """
    if not symptom:
        return []
    norm_sym = _normalize(symptom)
    tokens = set(norm_sym.split())

    # 收集所有症状的 n-gram（1-4 词），用于模糊匹配
    ngrams: Set[str] = set()
    sym_tokens = norm_sym.split()
    for n in range(1, min(5, len(sym_tokens) + 1)):
        for i in range(len(sym_tokens) - n + 1):
            ngrams.add(" ".join(sym_tokens[i: i + n]))

    candidates: List[Tuple[str, float]] = []
    for disease, sym_dict in kg.items():
        best_w = 0.0
        for kg_sym, w in sym_dict.items():
            # 精确匹配
            if kg_sym == norm_sym:
                best_w = max(best_w, w)
                break
            # n-gram 子集匹配（症状词在 ngrams 中）
            if kg_sym in ngrams or norm_sym in kg_sym or kg_sym in norm_sym:
                best_w = max(best_w, w)
            else:
                # Jaccard token overlap
                kg_tokens = set(kg_sym.split())
                if kg_tokens and tokens:
                    jac = len(tokens & kg_tokens) / len(tokens | kg_tokens)
                    if jac >= 0.5:
                        best_w = max(best_w, w * jac)
        if best_w > 0:
            candidates.append((disease, best_w))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def query_symptoms_by_disease(
    disease: str,
    kg: Dict,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Doctor Agent 的 KG 工具接口：给定一个疾病名（可以不精确），返回该疾病在
    KG 中的症状列表，按权重（诊断重要性）降序排列。

    用法场景：
      - Doctor 确定假设后，用该疾病查 KG 中有哪些症状，
        从而知道接下来该针对性地问哪些症状来确认/排除假设。

    Args:
        disease : 疾病名（自然语言，会先 normalize + fuzzy 匹配）
        kg      : 已加载的知识图谱
        top_k   : 最多返回 top_k 个症状

    Returns:
        [(symptom_name, weight), ...] 按 weight 降序，symptom_name 为 KG normalized key
        空列表表示 KG 中找不到该疾病
    """
    if not disease:
        return []
    norm_d = _normalize(disease)

    # 精确匹配
    sym_dict = kg.get(norm_d)
    if sym_dict is None:
        # fuzzy 匹配
        matches = fuzzy_match_disease(disease, kg, top_k=1)
        if matches and matches[0][1] >= 0.2:
            sym_dict = kg.get(matches[0][0])
    if not sym_dict:
        return []

    items = sorted(sym_dict.items(), key=lambda x: -x[1])
    return items[:top_k]


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
