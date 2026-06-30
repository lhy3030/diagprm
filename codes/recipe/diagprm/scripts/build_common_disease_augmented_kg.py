#!/usr/bin/env python3
"""Build a common-disease augmented DiagPRM dataset.

This script keeps the original clean_v2 files intact and creates a sibling
dataset directory whose common diseases use a curated patient-observable fact
set. The goal is to make KG-supervised turn-level rewards easier to hit in
natural diagnostic dialogue.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


COMMON_AUGMENTS: dict[str, list[str]] = {
    "pneumonia": [
        "fever or chills",
        "cough",
        "coughing up yellow or green mucus",
        "shortness of breath",
        "chest pain that is worse with breathing or coughing",
        "fatigue or feeling very weak",
        "rapid breathing",
        "sweating or night sweats",
        "recent cold or respiratory infection",
        "confusion in an older patient",
    ],
    "asthma": [
        "wheezing",
        "shortness of breath",
        "chest tightness",
        "cough that is worse at night or early morning",
        "symptoms triggered by exercise",
        "symptoms triggered by dust pollen pets or smoke",
        "recurrent episodes of breathing difficulty",
        "relief after using an inhaler",
        "history of allergies or eczema",
        "difficulty speaking during severe attacks",
    ],
    "chronic obstructive pulmonary disease": [
        "long-term cough",
        "coughing up phlegm most days",
        "shortness of breath with activity",
        "wheezing",
        "chest tightness",
        "history of smoking",
        "frequent chest infections",
        "symptoms gradually worsening over years",
        "fatigue with exertion",
        "unintentional weight loss in advanced disease",
    ],
    "chronic obstructive airway disease": [
        "long-term cough",
        "coughing up phlegm most days",
        "shortness of breath with activity",
        "wheezing",
        "chest tightness",
        "history of smoking",
        "frequent chest infections",
        "symptoms gradually worsening over years",
        "fatigue with exertion",
        "unintentional weight loss in advanced disease",
    ],
    "hypertension": [
        "repeated high blood pressure readings",
        "headaches",
        "dizziness or lightheadedness",
        "blurred vision",
        "chest discomfort",
        "shortness of breath",
        "nosebleeds with very high blood pressure",
        "family history of high blood pressure",
        "high salt diet or weight gain",
        "kidney disease or diabetes history",
    ],
    "hypertensive disease": [
        "repeated high blood pressure readings",
        "headaches",
        "dizziness or lightheadedness",
        "blurred vision",
        "chest discomfort",
        "shortness of breath",
        "nosebleeds with very high blood pressure",
        "family history of high blood pressure",
        "high salt diet or weight gain",
        "kidney disease or diabetes history",
    ],
    "type 2 diabetes mellitus": [
        "increased thirst",
        "frequent urination",
        "increased hunger",
        "unexplained weight loss",
        "fatigue",
        "blurry vision",
        "slow-healing cuts or infections",
        "numbness or tingling in the feet",
        "history of obesity or weight gain",
        "family history of diabetes",
    ],
    "diabetes": [
        "increased thirst",
        "frequent urination",
        "increased hunger",
        "unexplained weight loss",
        "fatigue",
        "blurry vision",
        "slow-healing cuts or infections",
        "numbness or tingling in the feet",
        "history of obesity or weight gain",
        "family history of diabetes",
    ],
    "infection urinary tract": [
        "burning or pain when urinating",
        "frequent urination",
        "urgent need to urinate",
        "lower abdominal or pelvic pain",
        "cloudy or foul-smelling urine",
        "blood in the urine",
        "fever",
        "flank or back pain",
        "new confusion in an older patient",
        "recent urinary catheter or urinary tract procedure",
    ],
    "gastroesophageal reflux disease": [
        "heartburn",
        "burning pain behind the breastbone",
        "acid or sour taste in the mouth",
        "regurgitation of food or fluid",
        "symptoms worse after meals",
        "symptoms worse when lying down",
        "chronic cough or throat clearing",
        "hoarse voice",
        "difficulty swallowing",
        "relief with antacids",
    ],
    "migraine": [
        "moderate to severe headache",
        "headache on one side of the head",
        "throbbing or pulsing headache",
        "nausea or vomiting",
        "sensitivity to light",
        "sensitivity to sound",
        "visual aura or flashing lights before headache",
        "headache worsened by activity",
        "recurrent similar headaches",
        "need to rest in a dark room",
    ],
    "anemia": [
        "fatigue",
        "weakness",
        "pale skin",
        "shortness of breath with activity",
        "dizziness or lightheadedness",
        "fast or pounding heartbeat",
        "cold hands or feet",
        "headaches",
        "heavy menstrual bleeding or blood loss",
        "poor diet or low iron intake",
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
        "symptoms started suddenly",
        "recent contact with someone with flu-like illness",
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
        "sore throat",
        "cough lasting more than several days",
    ],
    "upper respiratory infection": [
        "runny or stuffy nose",
        "sore throat",
        "cough",
        "sneezing",
        "low-grade fever",
        "headache",
        "fatigue",
        "postnasal drip",
        "symptoms started gradually",
        "recent sick contact",
    ],
    "cellulitis": [
        "red area of skin",
        "skin warmth",
        "skin swelling",
        "skin pain or tenderness",
        "spreading redness",
        "fever",
        "recent cut scrape or wound",
        "pus or drainage",
        "leg or foot involvement",
        "history of diabetes or poor circulation",
    ],
    "sepsis": [
        "fever or very low temperature",
        "fast heart rate",
        "rapid breathing",
        "confusion or altered mental status",
        "severe weakness",
        "low blood pressure or fainting",
        "suspected infection",
        "decreased urination",
        "cold or mottled skin",
        "recent hospitalization or procedure",
    ],
    "septicemia": [
        "fever or very low temperature",
        "fast heart rate",
        "rapid breathing",
        "confusion or altered mental status",
        "severe weakness",
        "low blood pressure or fainting",
        "suspected infection",
        "decreased urination",
        "cold or mottled skin",
        "recent hospitalization or procedure",
    ],
    "coronary artery disease": [
        "chest pain or pressure with exertion",
        "chest pain relieved by rest",
        "shortness of breath with activity",
        "pain spreading to arm neck jaw or back",
        "sweating with chest discomfort",
        "nausea with chest discomfort",
        "history of high cholesterol",
        "history of smoking",
        "diabetes or high blood pressure history",
        "family history of early heart disease",
    ],
    "coronary arteriosclerosis": [
        "chest pain or pressure with exertion",
        "chest pain relieved by rest",
        "shortness of breath with activity",
        "pain spreading to arm neck jaw or back",
        "sweating with chest discomfort",
        "nausea with chest discomfort",
        "history of high cholesterol",
        "history of smoking",
        "diabetes or high blood pressure history",
        "family history of early heart disease",
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
        "risk factors such as smoking diabetes or high blood pressure",
        "sense of impending doom",
    ],
    "failure heart": [
        "shortness of breath with activity",
        "shortness of breath when lying flat",
        "waking up short of breath at night",
        "leg or ankle swelling",
        "rapid weight gain from fluid",
        "fatigue",
        "reduced exercise tolerance",
        "persistent cough or wheeze",
        "history of heart disease or high blood pressure",
        "palpitations",
    ],
    "failure heart congestive": [
        "shortness of breath with activity",
        "shortness of breath when lying flat",
        "waking up short of breath at night",
        "leg or ankle swelling",
        "rapid weight gain from fluid",
        "fatigue",
        "reduced exercise tolerance",
        "persistent cough or wheeze",
        "history of heart disease or high blood pressure",
        "palpitations",
    ],
    "stroke": [
        "sudden weakness or numbness on one side",
        "face drooping",
        "trouble speaking or slurred speech",
        "sudden confusion",
        "sudden vision loss",
        "sudden trouble walking",
        "loss of balance or coordination",
        "sudden severe headache",
        "symptoms started abruptly",
        "history of atrial fibrillation high blood pressure or prior stroke",
    ],
    "accident cerebrovascular": [
        "sudden weakness or numbness on one side",
        "face drooping",
        "trouble speaking or slurred speech",
        "sudden confusion",
        "sudden vision loss",
        "sudden trouble walking",
        "loss of balance or coordination",
        "sudden severe headache",
        "symptoms started abruptly",
        "history of atrial fibrillation high blood pressure or prior stroke",
    ],
    "deep vein thrombosis": [
        "one-sided leg swelling",
        "calf pain or tenderness",
        "leg warmth",
        "red or discolored skin on the leg",
        "recent surgery or hospitalization",
        "recent long travel or immobility",
        "history of blood clots",
        "pregnancy or recent childbirth",
        "use of estrogen therapy or birth control pills",
        "leg pain worse when standing or walking",
    ],
    "embolism pulmonary": [
        "sudden shortness of breath",
        "sharp chest pain worse with breathing",
        "coughing up blood",
        "fast heart rate",
        "lightheadedness or fainting",
        "recent leg swelling or calf pain",
        "recent surgery hospitalization or long travel",
        "history of blood clots",
        "low oxygen level",
        "anxiety or sense of doom with breathing symptoms",
    ],
    "acute pancreatitis": [
        "severe upper abdominal pain",
        "abdominal pain radiating to the back",
        "nausea or vomiting",
        "pain worse after eating",
        "abdominal tenderness",
        "fever",
        "fast heart rate",
        "history of gallstones",
        "heavy alcohol use",
        "recent very high triglycerides",
    ],
    "pancreatitis": [
        "severe upper abdominal pain",
        "abdominal pain radiating to the back",
        "nausea or vomiting",
        "pain worse after eating",
        "abdominal tenderness",
        "fever",
        "fast heart rate",
        "history of gallstones",
        "heavy alcohol use",
        "recent very high triglycerides",
    ],
    "cholecystitis": [
        "right upper abdominal pain",
        "pain after fatty meals",
        "pain radiating to the right shoulder or back",
        "fever",
        "nausea or vomiting",
        "right upper abdominal tenderness",
        "history of gallstones",
        "loss of appetite",
        "yellowing of the skin or eyes",
        "dark urine or pale stools",
    ],
    "cholelithiasis": [
        "right upper abdominal pain",
        "pain after fatty meals",
        "pain radiating to the right shoulder or back",
        "nausea or vomiting",
        "episodes of biliary colic",
        "history of gallstones",
        "pain lasting minutes to hours",
        "bloating after meals",
        "yellowing of the skin or eyes",
        "dark urine or pale stools",
    ],
    "nephrolithiasis": [
        "severe flank pain",
        "pain radiating to the groin",
        "blood in the urine",
        "nausea or vomiting",
        "pain comes in waves",
        "burning with urination",
        "frequent urination",
        "history of kidney stones",
        "dehydration or low fluid intake",
        "restlessness due to pain",
    ],
    "gout": [
        "sudden severe joint pain",
        "red swollen joint",
        "warmth over the joint",
        "pain often in the big toe",
        "pain starts at night",
        "recurrent attacks",
        "history of high uric acid",
        "recent alcohol or rich food intake",
        "kidney disease or diuretic use",
        "joint tenderness to light touch",
    ],
    "osteoarthritis": [
        "joint pain worse with use",
        "joint stiffness after rest",
        "morning stiffness lasting less than thirty minutes",
        "reduced range of motion",
        "joint swelling or bony enlargement",
        "grinding or clicking in the joint",
        "knee hip hand or spine involvement",
        "symptoms gradually worsening over years",
        "older age",
        "prior joint injury or overuse",
    ],
    "hypothyroidism": [
        "fatigue",
        "weight gain",
        "cold intolerance",
        "constipation",
        "dry skin",
        "hair thinning",
        "slow heart rate",
        "depressed mood",
        "heavy or irregular menstrual periods",
        "puffy face or hoarse voice",
    ],
    "hyperthyroidism": [
        "weight loss despite normal or increased appetite",
        "heat intolerance",
        "palpitations or fast heartbeat",
        "tremor",
        "anxiety or irritability",
        "increased sweating",
        "frequent bowel movements",
        "trouble sleeping",
        "neck swelling",
        "bulging eyes or eye irritation",
    ],
    "depression mental": [
        "persistent low mood",
        "loss of interest or pleasure",
        "sleeping too much or insomnia",
        "fatigue or low energy",
        "poor concentration",
        "changes in appetite or weight",
        "feelings of worthlessness or guilt",
        "slowed movement or agitation",
        "thoughts of death or self-harm",
        "symptoms lasting at least two weeks",
    ],
    "anxiety state": [
        "excessive worry",
        "restlessness",
        "muscle tension",
        "trouble sleeping",
        "difficulty concentrating",
        "irritability",
        "palpitations",
        "shortness of breath during anxiety",
        "panic attacks",
        "avoidance of feared situations",
    ],
}


def weight_for_rank(idx: int) -> float:
    if idx < 4:
        return 0.45
    if idx < 8:
        return 0.40
    return 0.35


def load_augments(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return COMMON_AUGMENTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_augments = payload.get("augments", payload) if isinstance(payload, dict) else {}
    augments: dict[str, list[str]] = {}
    for disease, facts in raw_augments.items():
        if not isinstance(facts, list):
            continue
        cleaned = []
        seen = set()
        for fact in facts:
            text = str(fact).strip()
            key = text.lower()
            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)
        if cleaned:
            augments[str(disease).strip().lower()] = cleaned
    if not augments:
        raise ValueError(f"No augments found in {path}")
    return augments


def make_augmented_record(record: dict[str, Any], augments: dict[str, list[str]], replace: bool) -> dict[str, Any]:
    disease = str(record.get("disease", "")).strip().lower()
    if disease not in augments:
        return record

    row = deepcopy(record)
    new_pool = {fact: weight_for_rank(i) for i, fact in enumerate(augments[disease])}

    if not replace:
        old_pool = row.get("symptoms_pool", {})
        if isinstance(old_pool, dict):
            for fact, weight in old_pool.items():
                if fact not in new_pool:
                    new_pool[str(fact)] = float(weight)

    symptom_facts = [
        {
            "fact_id": f"F{i:03d}",
            "text": fact,
            "weight": float(weight),
            "source": "common_disease_augment",
            "surface_type": "patient_observable",
        }
        for i, (fact, weight) in enumerate(new_pool.items())
    ]
    row["symptoms_pool"] = new_pool
    row["symptom_facts"] = symptom_facts
    row["common_disease_augmented"] = True
    row["augment_policy"] = "replace" if replace else "append"

    initial = [x for x in row.get("initial_symptoms", []) if x in new_pool]
    if not initial:
        initial = list(new_pool.keys())[:2]
    row["initial_symptoms"] = initial[:3]
    row["chief_complaint"] = (
        "A patient presents to clinic for evaluation. The patient reports: "
        + ", ".join(row["initial_symptoms"])
        + "."
    )
    return row


def make_synthetic_common_record(
    disease: str,
    facts: list[str],
    meta: dict[str, Any],
    variant_idx: int,
    global_index: int,
) -> dict[str, Any]:
    symptoms_pool = {fact: weight_for_rank(i) for i, fact in enumerate(facts)}
    symptom_facts = [
        {
            "fact_id": f"F{i:03d}",
            "text": fact,
            "weight": float(weight),
            "source": "common_disease_augment_synthetic",
            "surface_type": "patient_observable",
        }
        for i, (fact, weight) in enumerate(symptoms_pool.items())
    ]
    offset = (variant_idx * 2) % max(1, len(facts))
    initial_symptoms = [facts[offset % len(facts)], facts[(offset + 1) % len(facts)]]
    chief_templates = [
        "A patient presents to clinic for evaluation. The patient reports: {symptoms}.",
        "A patient seeks medical attention today. The patient reports: {symptoms}.",
        "A patient comes in because of ongoing symptoms. The patient reports: {symptoms}.",
    ]
    return {
        "disease": disease,
        "disease_id": meta.get("disease_id") or f"COMMON_AUG_{global_index:04d}",
        "disease_group": meta.get("disease_group") or disease,
        "chief_complaint": chief_templates[variant_idx % len(chief_templates)].format(
            symptoms=", ".join(initial_symptoms)
        ),
        "initial_symptoms": initial_symptoms,
        "symptoms_pool": symptoms_pool,
        "symptom_facts": symptom_facts,
        "split": "train",
        "visible_kg_policy": "none_by_default",
        "data_source": "kg_common_disease_aug_synthetic",
        "index": global_index,
        "common_disease_augmented": True,
        "synthetic_common_aug_case": True,
        "augment_policy": "synthetic_train_oversample",
    }


def record_to_parquet_row(record: dict[str, Any], max_turns: int) -> dict[str, Any]:
    prompt = [
        {
            "role": "system",
            "content": f"You are an experienced diagnostic physician conducting a multi-turn consultation. Maximum turns: {max_turns}.",
        },
        {"role": "user", "content": record["chief_complaint"]},
    ]
    ground_truth = {
        "disease": record["disease"],
        "disease_id": record.get("disease_id", ""),
        "disease_group": record.get("disease_group", record["disease"]),
        "symptoms_pool": record["symptoms_pool"],
        "symptom_facts": record["symptom_facts"],
        "initial_symptoms": record["initial_symptoms"],
        "split": record.get("split", ""),
    }
    extra_info = {
        "index": record.get("index"),
        "disease_id": record.get("disease_id", ""),
        "disease_group": record.get("disease_group", record["disease"]),
        "chief_complaint": record["chief_complaint"],
        "initial_symptoms": record["initial_symptoms"],
        "n_symptoms_pool": len(record["symptoms_pool"]),
        "n_symptom_facts": len(record["symptom_facts"]),
        "visible_kg_policy": record.get("visible_kg_policy", "none_by_default"),
        "common_disease_augmented": bool(record.get("common_disease_augmented", False)),
        "synthetic_common_aug_case": bool(record.get("synthetic_common_aug_case", False)),
        "augment_policy": record.get("augment_policy", "none"),
    }
    return {
        "prompt": json.dumps(prompt, ensure_ascii=False),
        "reward_model": json.dumps({"ground_truth": ground_truth}, ensure_ascii=False),
        "data_source": record.get("data_source", "kg_clean_v2") + "_common_aug",
        "agent_name": "diagprm_interaction",
        "extra_info": json.dumps(extra_info, ensure_ascii=False),
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("diagprm_dataset/clean_v2"))
    parser.add_argument("--output_dir", type=Path, default=Path("diagprm_dataset/clean_v2_common_aug"))
    parser.add_argument("--policy", choices=["replace", "append"], default="replace")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument(
        "--augment_json",
        type=Path,
        default=None,
        help="Optional JSON with {'augments': {disease: [facts]}} or direct {disease: [facts]}.",
    )
    parser.add_argument(
        "--synthetic_train_cases_per_disease",
        type=int,
        default=0,
        help="Append this many curated common-disease synthetic cases per matched disease to train only.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replace = args.policy == "replace"
    augments = load_augments(args.augment_json.resolve() if args.augment_json else None)

    kg = json.loads((input_dir / "clean_master_kg.json").read_text(encoding="utf-8"))
    augmented_kg = deepcopy(kg)
    matched_kg = []
    for disease, facts in augments.items():
        if disease not in augmented_kg:
            continue
        matched_kg.append(disease)
        new_pool = {fact: weight_for_rank(i) for i, fact in enumerate(facts)}
        if not replace:
            for fact, weight in kg.get(disease, {}).items():
                if fact not in new_pool:
                    new_pool[str(fact)] = float(weight)
        augmented_kg[disease] = new_pool

    (output_dir / "clean_master_kg.json").write_text(
        json.dumps(augmented_kg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    augment_doc = {
        "description": "Curated common-disease patient-observable fact augment for DiagPRM.",
        "policy": args.policy,
        "augment_json": str(args.augment_json.resolve()) if args.augment_json else None,
        "num_common_diseases_defined": len(augments),
        "num_common_diseases_matched_in_kg": len(matched_kg),
        "matched_diseases": matched_kg,
        "augments": augments,
    }
    (output_dir / "common_disease_augments.json").write_text(
        json.dumps(augment_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    disease_meta: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        for row in iter_jsonl(input_dir / f"kg_{split}_dataset.jsonl"):
            key = str(row.get("disease", "")).strip().lower()
            if key and key not in disease_meta:
                disease_meta[key] = {
                    "disease_id": row.get("disease_id", ""),
                    "disease_group": row.get("disease_group", key),
                }

    split_stats = {}
    for split in ["train", "val", "test"]:
        in_jsonl = input_dir / f"kg_{split}_dataset.jsonl"
        out_jsonl = output_dir / f"kg_{split}_dataset.jsonl"
        records = [make_augmented_record(r, augments, replace=replace) for r in iter_jsonl(in_jsonl)]
        synthetic_added = 0
        if split == "train" and args.synthetic_train_cases_per_disease > 0:
            next_index = max(int(r.get("index", -1) or -1) for r in records) + 1
            for disease in matched_kg:
                facts = augments[disease]
                for variant_idx in range(args.synthetic_train_cases_per_disease):
                    records.append(
                        make_synthetic_common_record(
                            disease=disease,
                            facts=facts,
                            meta=disease_meta.get(disease, {}),
                            variant_idx=variant_idx,
                            global_index=next_index,
                        )
                    )
                    next_index += 1
                    synthetic_added += 1
        write_jsonl(out_jsonl, records)
        rows = [record_to_parquet_row(r, max_turns=args.max_turns) for r in records]
        pd.DataFrame(rows).to_parquet(output_dir / f"diagprm_{split}.parquet", index=False)
        split_stats[split] = {
            "rows": len(records),
            "augmented_rows": sum(1 for r in records if r.get("common_disease_augmented")),
            "synthetic_common_aug_rows": synthetic_added,
        }

    for filename in ["split_manifest.json", "clean_filter_report.md"]:
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policy": args.policy,
        "augment_json": str(args.augment_json.resolve()) if args.augment_json else None,
        "num_common_diseases_defined": len(augments),
        "num_common_diseases_matched_in_kg": len(matched_kg),
        "synthetic_train_cases_per_disease": args.synthetic_train_cases_per_disease,
        "matched_diseases": matched_kg,
        "split_stats": split_stats,
        "note": "For RL, set DIAGPRM_DATASET to this output_dir and KG_PATH to output_dir/clean_master_kg.json.",
    }
    (output_dir / "common_aug_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
