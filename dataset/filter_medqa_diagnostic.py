"""
Filter medqa_test_dataset.jsonl to keep only genuine diagnostic questions
(i.e. questions whose answer is a disease/diagnosis name, and whose
atomic_facts contain symptom-like information).

Output:
  medqa_diag_test.jsonl   -- filtered diagnostic-only MedQA test set

Filtering criteria (all must pass):
  1. Question text contains a diagnosis-intent keyword
  2. The answer_info (correct option text) looks like a disease/condition name
     (not a drug, procedure, lab value, management action, etc.)
  3. atomic_facts is non-empty and contains ≥ 1 symptom-like fact
  4. (Optional) The answer_info matches a disease in master_kg.json
     → reported as coverage stat but NOT used as a hard filter
       (would be too restrictive; keeps zero-coverage cases for generalization)
"""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
KG_PATH    = SCRIPT_DIR.parent / "origin_dataset" / "master_kg.json"
IN_FILE    = SCRIPT_DIR / "medqa_test_dataset.jsonl"
OUT_FILE   = SCRIPT_DIR.parent / "diagprm_dataset" / "medqa_diag_test.jsonl"

# ── Keyword lists ─────────────────────────────────────────────────────────────

# Q1: question text must contain ≥1 of these (case-insensitive)
DIAG_KEYWORDS = [
    "most likely diagnosis",
    "what is the diagnosis",
    "most likely cause",
    "what is the cause",
    "most likely etiology",
    "most likely underlying",
    "most likely responsible",
    "most likely condition",
    "most likely disorder",
    "most likely disease",
    "most likely syndrome",
    "what type of",         # e.g. "what type of shock"
    "which condition",
    "which disease",
    "which disorder",
    "which syndrome",
    "which diagnosis",
    "which of the following is the cause",
    "which of the following is the diagnosis",
    "which of the following best describes",
    "consistent with",      # "which is most consistent with"
    "most likely explanation",
]

# Q2: answer_info should NOT start with these → likely a drug/procedure/action
NON_DISEASE_STARTS = [
    "administer", "give", "start", "begin", "prescribe", "order",
    "perform", "obtain", "schedule", "refer", "discontinue", "stop",
    "reassure", "counsel", "observe", "monitor", "measure", "check",
    "tell", "report", "inform", "notify", "disclose",
    "inhibition", "activation", "stimulation", "generation",
    "cross-linking", "hyperstabilization",
    "surgical", "incision", "biopsy", "aspiration",
]

# Q2: answer_info token patterns that suggest non-disease answers
NON_DISEASE_PATTERNS = [
    r"^\d",                           # starts with a number (lab value, dose)
    r"\b(mg|mcg|ml|mmol|ng|iu)\b",   # dosage units
    r"\bagar\b|\bculture\b|\bstain\b|\bpcr\b",
    r"\bsurgery\b|\boperation\b|\brepair\b",
    r"\btherapy\b|\btreatment\b|\bmanagement\b",
    r"\btest\b|\bexam\b|\bstudy\b|\bimaging\b",
    r"\binhibitor\b|\bagonist\b|\bantagonist\b",
    r"\bplasmid\b|\bvector\b|\bprimer\b",
]

# Q3: atomic_fact symptom markers (at least 1 fact must match)
SYMPTOM_MARKERS = [
    r"\b(has|have|presents?|complains?|reports?|shows?|exhibits?|develops?)\b",
    r"\b(pain|fever|nausea|vomit|cough|dyspn|fatigue|weakness|swelling)\b",
    r"\b(weight loss|appetite|discharge|bleeding|rash|redness|edema)\b",
    r"\b(history of|diagnosed|positive|negative|elevated|decreased|low|high)\b",
    r"\b(temperature|blood pressure|pulse|heart rate|oxygen)\b",
    r"\b(laboratory|labs?|serum|level|count|result)\b",
]


def load_kg_diseases(path: Path) -> set:
    with open(path) as f:
        kg = json.load(f)
    return set(kg.keys())


def question_is_diagnostic(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DIAG_KEYWORDS)


def answer_looks_like_disease(answer_info: str) -> bool:
    a = answer_info.lower().strip()
    if not a:
        return False
    # Reject if starts with action verbs
    if any(a.startswith(kw) for kw in NON_DISEASE_STARTS):
        return False
    # Reject if matches non-disease patterns
    if any(re.search(pat, a) for pat in NON_DISEASE_PATTERNS):
        return False
    return True


def facts_contain_symptoms(atomic_facts: list) -> bool:
    if not atomic_facts:
        return False
    for fact in atomic_facts:
        fact_lower = fact.lower()
        if any(re.search(pat, fact_lower) for pat in SYMPTOM_MARKERS):
            return True
    return False


def get_question_text(record: dict) -> str:
    """Extract the question Problem field from the prompt."""
    for msg in record.get("prompt", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            m = re.search(r"Problem:\s*(.*?)(?:\nOptions:|$)", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            return content
    return ""


def main():
    kg_diseases = load_kg_diseases(KG_PATH)
    print(f"KG diseases loaded: {len(kg_diseases)}")

    records = []
    with open(IN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Total MedQA test records: {len(records)}")

    kept        = []
    stats       = {
        "total":              len(records),
        "fail_diag_keyword":  0,
        "fail_answer_type":   0,
        "fail_no_symptoms":   0,
        "passed":             0,
        "kg_covered":         0,
    }

    for rec in records:
        question    = get_question_text(rec)
        gt          = rec.get("ground_truth", {})
        answer_info = (gt.get("answer_info", "") or "").strip()
        atomic_facts = rec.get("extra_info", {}).get("atomic_facts", [])

        # Filter 1: diagnostic question
        if not question_is_diagnostic(question):
            stats["fail_diag_keyword"] += 1
            continue

        # Filter 2: answer looks like a disease name
        if not answer_looks_like_disease(answer_info):
            stats["fail_answer_type"] += 1
            continue

        # Filter 3: atomic_facts contain symptom info
        if not facts_contain_symptoms(atomic_facts):
            stats["fail_no_symptoms"] += 1
            continue

        stats["passed"] += 1
        if answer_info.lower() in kg_diseases:
            stats["kg_covered"] += 1
        kept.append(rec)

    print("\n=== Filtering Statistics ===")
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")
    if stats["passed"] > 0:
        print(f"  KG coverage rate       : "
              f"{100 * stats['kg_covered'] / stats['passed']:.1f}%")

    # Re-index
    for i, rec in enumerate(kept):
        rec["index"] = i

    with open(OUT_FILE, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(kept)} diagnostic records → {OUT_FILE}")


if __name__ == "__main__":
    main()
