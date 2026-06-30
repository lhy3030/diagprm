#!/usr/bin/env python3
"""Curate QA-mined common-disease facts into disease-level KG augments.

Input is the auditable output from build_common_disease_augments_from_med_sources.py.
The script keeps only patient-observable, disease-relevant facts and rewrites
case-specific sentences into short patient-facing KG facts. It intentionally
does not force coverage for every disease; missing diseases should be filled
from stronger disease-level sources or manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FALLBACK_REVIEWED_FACTS: dict[str, list[str]] = {
    # Used only when --allow_fallback_seed is set. These are conservative
    # patient-observable facts to prevent the QA evidence pool from leaving
    # important common diseases empty.
    "hypertension": [
        "repeated high blood pressure readings",
        "headaches",
        "dizziness or lightheadedness",
        "blurred vision",
        "shortness of breath",
        "chest discomfort",
    ],
    "type 2 diabetes mellitus": [
        "increased thirst",
        "frequent urination",
        "unexplained weight loss",
        "fatigue",
        "blurry vision",
        "numbness or tingling in the feet",
    ],
    "urinary tract infection": [
        "burning or pain when urinating",
        "frequent urination",
        "urgent need to urinate",
        "lower abdominal or pelvic pain",
        "cloudy or foul-smelling urine",
        "blood in the urine",
    ],
    "gastroesophageal reflux disease": [
        "heartburn",
        "burning pain behind the breastbone",
        "acid or sour taste in the mouth",
        "regurgitation of food or fluid",
        "symptoms worse after meals",
        "symptoms worse when lying down",
    ],
    "migraine": [
        "moderate to severe headache",
        "throbbing or pulsing headache",
        "nausea or vomiting",
        "sensitivity to light",
        "sensitivity to sound",
        "visual aura or flashing lights before headache",
    ],
    "anemia": [
        "fatigue",
        "weakness",
        "pale skin",
        "shortness of breath with activity",
        "dizziness or lightheadedness",
        "fast or pounding heartbeat",
    ],
    "sepsis": [
        "fever or very low temperature",
        "fast heart rate",
        "rapid breathing",
        "confusion or altered mental status",
        "severe weakness",
        "low blood pressure or fainting",
    ],
    "heart failure": [
        "shortness of breath with activity",
        "shortness of breath when lying flat",
        "waking up short of breath at night",
        "leg or ankle swelling",
        "fatigue",
        "reduced exercise tolerance",
    ],
    "coronary artery disease": [
        "chest pain or pressure with exertion",
        "chest pain relieved by rest",
        "shortness of breath with activity",
        "pain spreading to arm neck jaw or back",
        "sweating with chest discomfort",
        "nausea with chest discomfort",
    ],
    "cholecystitis": [
        "right upper abdominal pain",
        "pain after fatty meals",
        "pain radiating to the right shoulder or back",
        "fever",
        "nausea or vomiting",
        "loss of appetite",
    ],
    "nephrolithiasis": [
        "severe flank pain",
        "pain radiating to the groin",
        "blood in the urine",
        "nausea or vomiting",
        "pain comes in waves",
        "burning with urination",
    ],
    "gout": [
        "sudden severe joint pain",
        "red swollen joint",
        "warmth over the joint",
        "pain often in the big toe",
        "recurrent attacks",
        "joint tenderness to light touch",
    ],
}

FALLBACK_REVIEWED_FACTS.update(
    {
        "pneumonia": [
            "fever or chills",
            "productive cough",
            "shortness of breath",
            "pleuritic chest pain",
            "fatigue or feeling very weak",
            "recent cold or respiratory infection",
            "smoking history",
            "night sweats",
        ],
        "asthma": [
            "wheezing",
            "shortness of breath",
            "chest tightness",
            "cough that is worse at night or early morning",
            "symptoms triggered by exercise",
            "symptoms triggered by dust pollen pets or smoke",
            "relief after using an inhaler",
            "history of allergies or eczema",
        ],
        "copd": [
            "smoking history",
            "long-term cough",
            "coughing up phlegm most days",
            "shortness of breath with exertion",
            "wheezing",
            "frequent chest infections",
            "symptoms gradually worsening over years",
            "unintentional weight loss",
        ],
        "influenza": [
            "sudden fever",
            "chills",
            "body aches",
            "dry cough",
            "sore throat",
            "runny or stuffy nose",
            "fatigue",
            "headache",
        ],
        "bronchitis": [
            "cough",
            "coughing up mucus",
            "chest discomfort",
            "wheezing",
            "shortness of breath",
            "fatigue",
            "low-grade fever",
            "symptoms after a cold",
        ],
        "myocardial infarction": [
            "severe chest pressure or tightness",
            "pain spreading to left arm neck jaw or back",
            "shortness of breath",
            "sweating",
            "nausea or vomiting",
            "lightheadedness or fainting",
            "symptoms lasting more than a few minutes",
            "history of coronary artery disease",
        ],
        "stroke": [
            "sudden weakness or numbness on one side",
            "face drooping",
            "trouble speaking or slurred speech",
            "sudden confusion",
            "sudden vision loss",
            "sudden trouble walking",
            "loss of balance or coordination",
            "symptoms started abruptly",
        ],
        "pulmonary embolism": [
            "sudden shortness of breath",
            "sharp chest pain worse with breathing",
            "coughing up blood",
            "fast heart rate",
            "lightheadedness or fainting",
            "recent leg swelling or calf pain",
            "recent surgery hospitalization or long travel",
            "history of blood clots",
        ],
        "acute pancreatitis": [
            "severe upper abdominal pain",
            "abdominal pain radiating to the back",
            "nausea or vomiting",
            "pain worse after eating",
            "fever",
            "fast heart rate",
            "heavy alcohol use",
            "history of gallstones",
        ],
        "osteoarthritis": [
            "joint pain worse with use",
            "joint stiffness after rest",
            "morning stiffness lasting less than thirty minutes",
            "reduced range of motion",
            "joint swelling or bony enlargement",
            "grinding or clicking in the joint",
            "symptoms gradually worsening over years",
            "older age",
        ],
        "depression": [
            "persistent low mood",
            "loss of interest or pleasure",
            "sleeping too much or insomnia",
            "fatigue or low energy",
            "poor concentration",
            "changes in appetite or weight",
            "feelings of worthlessness or guilt",
            "symptoms lasting at least two weeks",
        ],
        "anxiety": [
            "excessive worry",
            "restlessness",
            "muscle tension",
            "trouble sleeping",
            "difficulty concentrating",
            "irritability",
            "palpitations",
            "panic attacks",
        ],
    }
)

KG_DISEASE_ALIASES: dict[str, list[str]] = {
    "copd": ["chronic obstructive pulmonary disease"],
    "urinary tract infection": ["infection urinary tract"],
    "pulmonary embolism": ["embolism pulmonary"],
    "heart failure": ["failure heart", "failure heart congestive"],
    "depression": ["depression mental"],
    "anxiety": ["anxiety state"],
}


DROP_RE = re.compile(
    r"\b("
    r"blood pressure is|pulse is|respirations are|temperature is|bmi is|cm |kg |"
    r"radiograph|radiography|x-?ray|ct|mri|scan|ultrasound|ecg|ekg|"
    r"laboratory|serum|count|biopsy|culture|audiometry|test is|shows no abnormalities|"
    r"no abnormalities|operation|surgery|medications include|takes [a-z]+|"
    r"does not use illicit|does not have any children|mother died|menopause|"
    r"physician|recommended|within normal limits|abdomen is soft"
    r")\b",
    re.I,
)

NEGATION_RE = re.compile(r"\b(no|not|denies|without|never|does not|has not|had not)\b", re.I)

PATTERN_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"productive cough|coughing up (yellow|green|mucus|phlegm|sputum)", re.I), "productive cough"),
    (re.compile(r"dry cough|nonproductive cough", re.I), "dry cough"),
    (re.compile(r"\bcough\b", re.I), "cough"),
    (re.compile(r"chest pain.*inspiration|pain.*inspiration|worse with breathing|pleuritic", re.I), "pleuritic chest pain"),
    (re.compile(r"shortness of breath.*exertion|dyspnea.*exertion", re.I), "shortness of breath with exertion"),
    (re.compile(r"shortness of breath|dyspnea|breathing difficult", re.I), "shortness of breath"),
    (re.compile(r"wheez", re.I), "wheezing"),
    (re.compile(r"fever", re.I), "fever"),
    (re.compile(r"chills", re.I), "chills"),
    (re.compile(r"night sweats?", re.I), "night sweats"),
    (re.compile(r"weight loss", re.I), "unintentional weight loss"),
    (re.compile(r"smok|pack-year|pack of cigarettes", re.I), "smoking history"),
    (re.compile(r"alcohol|beers? per day|drinks", re.I), "heavy alcohol use"),
    (re.compile(r"nausea.*vomit|vomit.*nausea", re.I), "nausea or vomiting"),
    (re.compile(r"vomit", re.I), "vomiting"),
    (re.compile(r"nausea", re.I), "nausea"),
    (re.compile(r"abdominal pain.*back|pain radiates to the back", re.I), "abdominal pain radiating to the back"),
    (re.compile(r"abdominal pain|epigastric pain", re.I), "abdominal pain"),
    (re.compile(r"leg swelling|pitting edema|edema below the knees", re.I), "leg swelling"),
    (re.compile(r"joint swelling", re.I), "joint swelling"),
    (re.compile(r"joint pain|pain and crepitus", re.I), "joint pain"),
    (re.compile(r"headaches?", re.I), "headache"),
    (re.compile(r"fatigue", re.I), "fatigue"),
    (re.compile(r"weakness", re.I), "weakness"),
    (re.compile(r"palpitations?", re.I), "palpitations"),
    (re.compile(r"family history|sister has asthma", re.I), "family history"),
]

DISEASE_FACT_DENYLIST: dict[str, set[str]] = {
    "asthma": {"family history"},
    "pulmonary embolism": {"headache", "night sweats"},
    "bronchitis": {"hypertension"},
}


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def expand_kg_aliases(reviewed: dict[str, list[str]]) -> dict[str, list[str]]:
    expanded = dict(reviewed)
    for disease, aliases in KG_DISEASE_ALIASES.items():
        if disease not in reviewed:
            continue
        for alias in aliases:
            expanded.setdefault(alias, reviewed[disease])
    return expanded


def rewrite_fact(text: str) -> str | None:
    if DROP_RE.search(text):
        return None
    for pattern, replacement in PATTERN_RULES:
        if pattern.search(text):
            return replacement
    return None


def is_negative_only(text: str, rewritten: str) -> bool:
    if not NEGATION_RE.search(text):
        return False
    # A negated symptom is not a positive disease-level symptom. Smoking/alcohol
    # history is often phrased positively in the same sentence, so keep it.
    return rewritten not in {"smoking history", "heavy alcohol use"}


def load_candidates(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def curate(
    mined: dict[str, Any],
    min_facts_per_disease: int,
    max_facts_per_disease: int,
    allow_fallback_seed: bool,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    reviewed: dict[str, list[str]] = {}
    report: dict[str, Any] = {
        "min_facts_per_disease": min_facts_per_disease,
        "max_facts_per_disease": max_facts_per_disease,
        "allow_fallback_seed": allow_fallback_seed,
        "diseases": {},
    }

    for disease, payload in mined.get("diseases", {}).items():
        fact_sources: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, str]] = []

        for item in payload.get("top_candidates", []):
            if item.get("surface_type") != "patient_observable":
                rejected.append({"text": item.get("text", ""), "reason": "not_patient_observable"})
                continue
            text = str(item.get("text", ""))
            rewritten = rewrite_fact(text)
            if not rewritten:
                rejected.append({"text": text, "reason": "no_rewrite_rule_or_drop"})
                continue
            if is_negative_only(text, rewritten):
                rejected.append({"text": text, "reason": "negative_case_fact"})
                continue
            if rewritten in DISEASE_FACT_DENYLIST.get(disease, set()):
                rejected.append({"text": text, "reason": "disease_specific_denylist"})
                continue

            key = normalize_key(rewritten)
            if key not in fact_sources:
                fact_sources[key] = {
                    "fact": rewritten,
                    "source_counts": defaultdict(int),
                    "examples": [],
                }
            for source, count in item.get("source_counts", {}).items():
                fact_sources[key]["source_counts"][source] += int(count)
            if len(fact_sources[key]["examples"]) < 3:
                fact_sources[key]["examples"].extend(item.get("examples", [])[: 3 - len(fact_sources[key]["examples"])])

        facts_with_scores = []
        for value in fact_sources.values():
            source_counts = dict(value["source_counts"])
            support = sum(source_counts.values())
            med_support = sum(v for k, v in source_counts.items() if not k.startswith("PrimeKG"))
            facts_with_scores.append((value["fact"], support + med_support * 2, source_counts, value["examples"]))
        facts_with_scores.sort(key=lambda x: (-x[1], normalize_key(x[0])))

        facts = [fact for fact, _, _, _ in facts_with_scores[:max_facts_per_disease]]
        used_fallback = False
        if allow_fallback_seed and len(facts) < max_facts_per_disease:
            before_fallback = len(facts)
            for fact in FALLBACK_REVIEWED_FACTS.get(disease, []):
                if normalize_key(fact) not in {normalize_key(x) for x in facts}:
                    facts.append(fact)
                if len(facts) >= max_facts_per_disease:
                    break
            used_fallback = len(facts) > before_fallback

        if len(facts) >= min_facts_per_disease:
            reviewed[disease] = facts[:max_facts_per_disease]

        report["diseases"][disease] = {
            "input_candidates": payload.get("num_candidates", 0),
            "accepted_facts": facts[:max_facts_per_disease],
            "num_accepted": min(len(facts), max_facts_per_disease),
            "included": disease in reviewed,
            "used_fallback_seed": used_fallback,
            "evidence": [
                {
                    "fact": fact,
                    "score": score,
                    "source_counts": source_counts,
                    "examples": examples,
                }
                for fact, score, source_counts, examples in facts_with_scores[:max_facts_per_disease]
            ],
            "rejected": rejected[:30],
        }

    return reviewed, report


def write_outputs(reviewed: dict[str, list[str]], report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_reviewed = expand_kg_aliases(reviewed)
    payload = {
        "description": "Reviewed common-disease patient-observable KG augments derived from local QA evidence.",
        "usage": "Pass augments to build_common_disease_augmented_kg.py via --augment_json.",
        "augments": expanded_reviewed,
        "canonical_augments": reviewed,
        "kg_aliases": KG_DISEASE_ALIASES,
    }
    (output_dir / "reviewed_common_disease_augments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reviewed_common_disease_augments_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Reviewed Common Disease Augments",
        "",
        "These facts are disease-level, patient-observable KG augment candidates derived from QA-mined evidence.",
        "",
        "| disease | facts | fallback |",
        "|---|---|---:|",
    ]
    for disease, facts in expanded_reviewed.items():
        fallback = report["diseases"].get(disease, {}).get("used_fallback_seed", False)
        lines.append(f"| {disease} | {', '.join(facts)} | {fallback} |")
    lines.append("")
    (output_dir / "reviewed_common_disease_augments.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mined_json",
        type=Path,
        default=Path("diagprm_dataset/common_disease_med_source_augments/common_disease_med_source_augments.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("diagprm_dataset/common_disease_med_source_augments/reviewed"),
    )
    parser.add_argument("--min_facts_per_disease", type=int, default=2)
    parser.add_argument("--max_facts_per_disease", type=int, default=10)
    parser.add_argument(
        "--allow_fallback_seed",
        action="store_true",
        help="Fill sparse high-value common diseases with conservative seed facts; report marks these.",
    )
    args = parser.parse_args()

    mined = load_candidates(args.mined_json.resolve())
    reviewed, report = curate(
        mined,
        min_facts_per_disease=args.min_facts_per_disease,
        max_facts_per_disease=args.max_facts_per_disease,
        allow_fallback_seed=args.allow_fallback_seed,
    )
    write_outputs(reviewed, report, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "num_diseases": len(reviewed),
                "num_facts": sum(len(v) for v in reviewed.values()),
                "diseases": sorted(reviewed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
