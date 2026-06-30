#!/usr/bin/env python3
"""Build provenance-backed common-disease augment candidates from local med data.

This script does not overwrite DiagPRM RL/SFT files. It mines local datasets
that are already in diagprm_dataset and writes an auditable candidate table:

  - ATPO-style MEDIQ/MedQA/AIE/MedMCQA JSONL files with atomic_facts
  - PrimeKG disease-phenotype edges/weights

The output is intended for human review before updating clean_v2_common_aug.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DIAG_KEYWORDS = [
    "most likely diagnosis",
    "diagnosis",
    "condition",
    "disorder",
    "disease",
    "cause",
    "etiology",
    "explanation",
]

PATIENT_OBSERVABLE_RE = re.compile(
    r"\b("
    r"pain|fever|cough|dyspnea|shortness|breath|wheez|chest|nausea|vomit|"
    r"fatigue|weak|dizz|headache|rash|swelling|edema|bleeding|discharge|"
    r"urination|urine|thirst|hunger|weight|vision|palpitation|sweat|chills|"
    r"diarrhea|constipation|sore|stiff|numb|tingl|confusion|smoking|"
    r"alcohol|drinks?|history|reports?|complains?|develops?|experiences?|has|have"
    r")\b",
    re.I,
)

CLINICAL_TEST_RE = re.compile(
    r"\b("
    r"laboratory|serum|count|biopsy|culture|audiometry|radiograph|radiography|"
    r"x-?ray|ct|mri|scan|ultrasound|ecg|ekg|electrocardiogram|histology|"
    r"pathology|microscopy|genetic|mutation|chromosome|antibody|antigen|"
    r"level|mm hg|mg/dl|db|vital signs|physical exam|examination|auscultation|"
    r"murmur|breath sounds|spirometry|pulmonary function|fev1|fvc"
    r")\b",
    re.I,
)

NON_FACT_RE = re.compile(
    r"\b("
    r"physician orders|received|course of|medication|therapy|treatment|"
    r"following|which of the following|answer|option"
    r")\b",
    re.I,
)


TARGETS: dict[str, dict[str, list[str]]] = {
    "pneumonia": {
        "aliases": ["pneumonia", "bacterial pneumonia", "viral pneumonia"],
        "primekg_names": ["pneumonia", "viral pneumonia", "bacterial pneumonia"],
    },
    "asthma": {
        "aliases": ["asthma", "bronchial asthma"],
        "primekg_names": ["asthma"],
    },
    "copd": {
        "aliases": ["copd", "chronic obstructive pulmonary disease", "chronic obstructive airway disease"],
        "primekg_names": ["chronic obstructive pulmonary disease", "copd, severe early onset"],
    },
    "hypertension": {
        "aliases": ["hypertension", "essential hypertension", "hypertensive disease"],
        "primekg_names": ["essential hypertension"],
    },
    "type 2 diabetes mellitus": {
        "aliases": ["type 2 diabetes mellitus", "diabetes mellitus type 2", "type 2 diabetes", "diabetes"],
        "primekg_names": ["type 2 diabetes mellitus", "diabetes mellitus"],
    },
    "urinary tract infection": {
        "aliases": ["urinary tract infection", "uti", "cystitis", "pyelonephritis"],
        "primekg_names": ["urinary tract infection", "cystitis", "pyelonephritis"],
    },
    "gastroesophageal reflux disease": {
        "aliases": ["gastroesophageal reflux disease", "gerd", "acid reflux"],
        "primekg_names": ["gastroesophageal reflux disease"],
    },
    "migraine": {
        "aliases": ["migraine", "migraine headache", "migraine with aura"],
        "primekg_names": ["migraine with aura", "migraine without aura"],
    },
    "anemia": {
        "aliases": ["anemia", "iron deficiency anemia", "anaemia"],
        "primekg_names": ["anemia", "iron deficiency anemia"],
    },
    "influenza": {
        "aliases": ["influenza", "flu"],
        "primekg_names": ["influenza", "seasonal influenza", "avian influenza"],
    },
    "bronchitis": {
        "aliases": ["bronchitis", "acute bronchitis", "chronic bronchitis"],
        "primekg_names": ["bronchitis", "chronic bronchitis"],
    },
    "sepsis": {
        "aliases": ["sepsis", "septicemia", "septic shock"],
        "primekg_names": ["sepsis", "septic shock"],
    },
    "coronary artery disease": {
        "aliases": ["coronary artery disease", "coronary heart disease", "cad"],
        "primekg_names": ["coronary artery disease"],
    },
    "myocardial infarction": {
        "aliases": ["myocardial infarction", "heart attack", "acute myocardial infarction"],
        "primekg_names": ["myocardial infarction (disease)", "acute myocardial infarction"],
    },
    "heart failure": {
        "aliases": ["heart failure", "congestive heart failure", "failure heart"],
        "primekg_names": ["heart failure", "congestive heart failure"],
    },
    "stroke": {
        "aliases": ["stroke", "cerebrovascular accident", "ischemic stroke"],
        "primekg_names": ["stroke disorder", "ischemic stroke"],
    },
    "pulmonary embolism": {
        "aliases": ["pulmonary embolism", "embolism pulmonary"],
        "primekg_names": ["pulmonary embolism (disease)"],
    },
    "acute pancreatitis": {
        "aliases": ["acute pancreatitis", "pancreatitis"],
        "primekg_names": ["acute pancreatitis", "pancreatitis"],
    },
    "cholecystitis": {
        "aliases": ["cholecystitis", "acute cholecystitis"],
        "primekg_names": ["cholecystitis", "acute cholecystitis"],
    },
    "nephrolithiasis": {
        "aliases": ["nephrolithiasis", "kidney stone", "kidney stones", "renal calculus"],
        "primekg_names": ["nephrolithiasis", "kidney stone"],
    },
    "gout": {
        "aliases": ["gout", "gouty arthritis"],
        "primekg_names": ["gout"],
    },
    "osteoarthritis": {
        "aliases": ["osteoarthritis", "degenerative joint disease"],
        "primekg_names": ["osteoarthritis"],
    },
    "hypothyroidism": {
        "aliases": ["hypothyroidism"],
        "primekg_names": ["hypothyroidism", "congenital hypothyroidism"],
    },
    "hyperthyroidism": {
        "aliases": ["hyperthyroidism", "graves disease", "graves' disease"],
        "primekg_names": ["hyperthyroidism", "graves disease"],
    },
    "depression": {
        "aliases": ["depression", "major depressive disorder", "depressive disorder"],
        "primekg_names": ["depression"],
    },
    "anxiety": {
        "aliases": ["anxiety", "anxiety disorder", "panic disorder"],
        "primekg_names": ["anxiety disorder", "panic disorder"],
    },
}


@dataclass
class CandidateFact:
    text: str
    disease: str
    surface_type: str
    source_counts: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)
    primekg_weight: float | None = None

    def add_example(self, source: str, example: dict[str, Any], limit: int = 3) -> None:
        self.source_counts[source] += 1
        if len(self.examples) < limit:
            self.examples.append(example)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def strip_fact_prefix(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text)
    return text.strip()


def canonical_fact(text: str) -> str:
    text = strip_fact_prefix(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def surface_type_for_fact(text: str) -> str:
    if NON_FACT_RE.search(text):
        return "exclude"
    if CLINICAL_TEST_RE.search(text):
        return "clinical_or_test"
    if PATIENT_OBSERVABLE_RE.search(text):
        return "patient_observable"
    return "clinical_or_test"


def build_alias_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    for disease, cfg in TARGETS.items():
        for alias in cfg["aliases"]:
            out[normalize_text(alias)] = disease
    return out


def match_answer(answer: str, alias_lookup: dict[str, str]) -> str | None:
    norm = normalize_text(answer)
    if norm in alias_lookup:
        return alias_lookup[norm]
    # Allow conservative parenthetical variants, e.g. "myocardial infarction (disease)".
    norm_no_paren = normalize_text(re.sub(r"\([^)]*\)", "", answer))
    if norm_no_paren in alias_lookup:
        return alias_lookup[norm_no_paren]
    return None


def get_question_from_prompt(record: dict[str, Any]) -> str:
    for msg in record.get("prompt", []):
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            m = re.search(r"Problem:\s*(.*?)(?:\nOptions:|$)", content, re.S)
            return m.group(1).strip() if m else content.strip()
    return ""


def is_diagnostic_question(text: str) -> bool:
    low = text.lower()
    return any(keyword in low for keyword in DIAG_KEYWORDS)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def add_candidate(
    candidates: dict[str, dict[str, CandidateFact]],
    disease: str,
    fact_text: str,
    source: str,
    example: dict[str, Any],
    primekg_weight: float | None = None,
) -> None:
    fact = canonical_fact(fact_text)
    if len(fact) < 3:
        return
    surface_type = surface_type_for_fact(fact)
    if surface_type == "exclude":
        return
    key = normalize_text(fact)
    if not key:
        return
    disease_bucket = candidates[disease]
    if key not in disease_bucket:
        disease_bucket[key] = CandidateFact(text=fact, disease=disease, surface_type=surface_type)
    item = disease_bucket[key]
    if item.surface_type != "patient_observable" and surface_type == "patient_observable":
        item.surface_type = surface_type
    if primekg_weight is not None:
        item.primekg_weight = max(float(primekg_weight), item.primekg_weight or 0.0)
    item.add_example(source, example)


def mine_atomic_fact_jsonl(
    path: Path,
    source_name: str,
    candidates: dict[str, dict[str, CandidateFact]],
    alias_lookup: dict[str, str],
    require_diag_question: bool,
) -> dict[str, int]:
    stats = Counter()
    if not path.exists():
        return {"missing": 1}
    for line_no, record in iter_jsonl(path):
        stats["rows"] += 1
        question = str(record.get("question") or get_question_from_prompt(record))
        if require_diag_question and not is_diagnostic_question(question):
            stats["skip_non_diag_question"] += 1
            continue

        answer = str(record.get("answer") or "")
        ground_truth = record.get("ground_truth")
        if isinstance(ground_truth, dict):
            answer = str(ground_truth.get("answer_info") or answer)
        disease = match_answer(answer, alias_lookup)
        if not disease:
            stats["skip_answer_not_target"] += 1
            continue

        atomic_facts = record.get("atomic_facts")
        if atomic_facts is None:
            atomic_facts = record.get("extra_info", {}).get("atomic_facts", [])
        if not isinstance(atomic_facts, list) or not atomic_facts:
            stats["skip_no_atomic_facts"] += 1
            continue

        stats["matched_cases"] += 1
        for fact in atomic_facts:
            add_candidate(
                candidates=candidates,
                disease=disease,
                fact_text=str(fact),
                source=source_name,
                example={
                    "path": str(path),
                    "line": line_no,
                    "answer": answer,
                    "question": question[:240],
                },
            )
            stats["facts_seen"] += 1
    return dict(stats)


def mine_primekg(
    path: Path,
    candidates: dict[str, dict[str, CandidateFact]],
    max_per_disease: int,
) -> dict[str, int]:
    stats = Counter()
    if not path.exists():
        return {"missing": 1}
    kg = json.loads(path.read_text(encoding="utf-8"))
    norm_to_name = {normalize_text(name): name for name in kg}
    for disease, cfg in TARGETS.items():
        rows: list[tuple[str, float, str]] = []
        for name in cfg["primekg_names"]:
            prime_name = norm_to_name.get(normalize_text(name))
            if not prime_name:
                continue
            for symptom, weight in kg.get(prime_name, {}).items():
                rows.append((str(symptom), float(weight), prime_name))
        rows.sort(key=lambda x: (-x[1], normalize_text(x[0])))
        for symptom, weight, prime_name in rows[:max_per_disease]:
            add_candidate(
                candidates=candidates,
                disease=disease,
                fact_text=symptom,
                source="PrimeKG:disease_phenotype_positive",
                primekg_weight=weight,
                example={"path": str(path), "primekg_disease": prime_name, "weight": weight},
            )
            stats["facts_seen"] += 1
        if rows:
            stats["matched_diseases"] += 1
    return dict(stats)


def score_candidate(item: CandidateFact) -> float:
    source_bonus = 0.0
    if any(src.startswith("MEDIQ") for src in item.source_counts):
        source_bonus += 2.0
    if any(src.startswith("ATPO") for src in item.source_counts):
        source_bonus += 1.5
    if any(src.startswith("PrimeKG") for src in item.source_counts):
        source_bonus += 1.0
    surface_bonus = 2.0 if item.surface_type == "patient_observable" else 0.0
    support = sum(item.source_counts.values())
    kg_weight = item.primekg_weight or 0.0
    return surface_bonus + source_bonus + min(support, 5) * 0.4 + kg_weight


def make_outputs(
    candidates: dict[str, dict[str, CandidateFact]],
    source_stats: dict[str, dict[str, int]],
    out_dir: Path,
    top_k: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "description": "Provenance-backed common-disease augment candidates mined from local med datasets.",
        "source_stats": source_stats,
        "targets": TARGETS,
        "diseases": {},
    }
    md_lines = [
        "# Common Disease Augment Candidates From Local Med Sources",
        "",
        "This is a review file. It does not overwrite RL/SFT datasets.",
        "",
        "Recommended rule: use `patient_observable` facts for DiagPRM reward anchors; keep `clinical_or_test` as evidence or evaluation context only.",
        "",
        "## Source Stats",
        "",
    ]
    for source, stats in source_stats.items():
        md_lines.append(f"- `{source}`: {json.dumps(stats, ensure_ascii=False)}")
    md_lines.extend(["", "## Candidates", ""])

    for disease in TARGETS:
        items = list(candidates.get(disease, {}).values())
        items.sort(key=lambda item: (-score_candidate(item), item.surface_type, normalize_text(item.text)))
        selected = items[:top_k]
        payload["diseases"][disease] = {
            "num_candidates": len(items),
            "top_candidates": [
                {
                    "text": item.text,
                    "surface_type": item.surface_type,
                    "score": round(score_candidate(item), 3),
                    "source_counts": dict(item.source_counts),
                    "primekg_weight": item.primekg_weight,
                    "examples": item.examples,
                }
                for item in selected
            ],
        }
        md_lines.append(f"### {disease}")
        md_lines.append("")
        md_lines.append(f"Total candidates: {len(items)}")
        md_lines.append("")
        if not selected:
            md_lines.append("_No local candidate found._")
            md_lines.append("")
            continue
        md_lines.append("| rank | surface | fact | sources |")
        md_lines.append("|---:|---|---|---|")
        for rank, item in enumerate(selected, 1):
            sources = ", ".join(f"{k}:{v}" for k, v in item.source_counts.items())
            fact = item.text.replace("|", "\\|")
            md_lines.append(f"| {rank} | {item.surface_type} | {fact} | {sources} |")
        md_lines.append("")

    (out_dir / "common_disease_med_source_augments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "common_disease_med_source_augments.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=Path("diagprm_dataset"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("diagprm_dataset/common_disease_med_source_augments"),
    )
    parser.add_argument("--top_k", type=int, default=14)
    parser.add_argument("--primekg_max_per_disease", type=int, default=16)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    candidates: dict[str, dict[str, CandidateFact]] = defaultdict(dict)
    alias_lookup = build_alias_lookup()
    source_stats: dict[str, dict[str, int]] = {}

    jsonl_sources = [
        ("MEDIQ:dev", dataset_dir / "mediQ" / "medqa_dev_convo.jsonl", True),
        ("MEDIQ:test", dataset_dir / "mediQ" / "medqa_test_convo.jsonl", True),
        ("ATPO:MedQA-test", dataset_dir / "MedQA" / "medqa_test_dataset.jsonl", True),
        ("ATPO:MedicalExam", dataset_dir / "MedicalExam" / "aie_test_dataset.jsonl", True),
        ("ATPO:MedMCQA-test", dataset_dir / "mcqa_test_dataset.jsonl", True),
        ("ATPO:MedQA-diagnostic-filtered", dataset_dir / "medqa_diag_test.jsonl", False),
    ]
    for source_name, path, require_diag in jsonl_sources:
        source_stats[source_name] = mine_atomic_fact_jsonl(
            path=path,
            source_name=source_name,
            candidates=candidates,
            alias_lookup=alias_lookup,
            require_diag_question=require_diag,
        )

    source_stats["PrimeKG"] = mine_primekg(
        path=dataset_dir / "PrimeKG" / "disease_symptom_weighted.json",
        candidates=candidates,
        max_per_disease=args.primekg_max_per_disease,
    )

    make_outputs(candidates, source_stats, args.output_dir.resolve(), top_k=args.top_k)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "num_target_diseases": len(TARGETS),
                "num_diseases_with_candidates": sum(1 for d in TARGETS if candidates.get(d)),
                "source_stats": source_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
