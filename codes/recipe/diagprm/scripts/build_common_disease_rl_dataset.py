#!/usr/bin/env python3
"""Build a common-disease-only DiagPRM RL dataset.

This creates a small, focused KG-grounded diagnostic dialogue benchmark from
curated common-disease augment facts. It is intended for early RL experiments
where final diagnosis rewards should be much less sparse than in the full
rare-disease KG.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.diagprm.scripts.build_common_disease_augmented_kg import (
    load_augments,
    record_to_parquet_row,
    weight_for_rank,
)


CHIEF_TEMPLATES = [
    "A patient presents to clinic for evaluation. The patient reports: {symptoms}.",
    "A patient seeks medical attention today. The patient reports: {symptoms}.",
    "A patient comes in because of ongoing symptoms. The patient reports: {symptoms}.",
    "A patient arrives for a diagnostic consultation. The patient reports: {symptoms}.",
    "A patient is worried about recent symptoms. The patient reports: {symptoms}.",
]

CANONICAL_DISEASE_NAMES = {
    "chronic obstructive pulmonary disease": "copd",
    "infection urinary tract": "urinary tract infection",
    "embolism pulmonary": "pulmonary embolism",
    "failure heart": "heart failure",
    "failure heart congestive": "heart failure",
    "depression mental": "depression",
    "anxiety state": "anxiety",
}


def norm(text: str) -> str:
    return str(text or "").strip().lower()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_alias_map(kg: dict[str, Any], augments: dict[str, list[str]]) -> dict[str, str]:
    """Map augment disease names to existing KG disease names."""
    keys = {norm(k): k for k in kg.keys()}
    aliases = {
        "urinary tract infection": "infection urinary tract",
        "pulmonary embolism": "embolism pulmonary",
        "anxiety": "anxiety state",
        "heart failure": "failure heart",
        "depression": "depression mental",
        "copd": "chronic obstructive pulmonary disease",
    }

    out: dict[str, str] = {}
    for disease in augments:
        d = norm(disease)
        if d in keys:
            out[disease] = keys[d]
        elif d in aliases and norm(aliases[d]) in keys:
            out[disease] = keys[norm(aliases[d])]
    return out


def canonical_disease_name(disease: str) -> str:
    return CANONICAL_DISEASE_NAMES.get(norm(disease), norm(disease))


def merge_facts(old: list[str], new: list[str]) -> list[str]:
    merged = []
    seen = set()
    for fact in list(old) + list(new):
        text = str(fact).strip()
        key = norm(text)
        if text and key not in seen:
            merged.append(text)
            seen.add(key)
    return merged


def make_record(
    disease: str,
    facts: list[str],
    split: str,
    case_idx: int,
    global_index: int,
    seed: int,
    min_initial: int,
    max_initial: int,
) -> dict[str, Any]:
    rng = random.Random(seed + global_index * 9973)
    clean_facts = []
    seen = set()
    for fact in facts:
        text = str(fact).strip()
        key = norm(text)
        if text and key not in seen:
            clean_facts.append(text)
            seen.add(key)
    if len(clean_facts) < 3:
        raise ValueError(f"Need at least 3 facts for {disease}, got {len(clean_facts)}")

    n_initial = min(max_initial, max(min_initial, 2), len(clean_facts))
    # Rotate first so every scenario starts from a different chief complaint.
    offset = (case_idx * 2) % len(clean_facts)
    rotated = clean_facts[offset:] + clean_facts[:offset]
    initial = rotated[:n_initial]

    # Shuffle the remaining facts while keeping the initial symptoms first.
    remaining = rotated[n_initial:]
    rng.shuffle(remaining)
    ordered_facts = initial + remaining

    symptoms_pool = {fact: weight_for_rank(i) for i, fact in enumerate(ordered_facts)}
    symptom_facts = [
        {
            "fact_id": f"F{i:03d}",
            "text": fact,
            "weight": float(weight),
            "source": "common_disease_rl",
            "surface_type": "patient_observable",
        }
        for i, (fact, weight) in enumerate(symptoms_pool.items())
    ]
    template = CHIEF_TEMPLATES[case_idx % len(CHIEF_TEMPLATES)]
    return {
        "disease": disease,
        "disease_id": f"COMMON_RL_{norm(disease).replace(' ', '_')}_{case_idx:03d}",
        "disease_group": disease,
        "chief_complaint": template.format(symptoms=", ".join(initial)),
        "initial_symptoms": initial,
        "symptoms_pool": symptoms_pool,
        "symptom_facts": symptom_facts,
        "split": split,
        "visible_kg_policy": "none_by_default",
        "data_source": "kg_common_disease_rl",
        "index": global_index,
        "common_disease_rl": True,
        "common_disease_augmented": True,
        "scenario_idx": case_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_dataset_dir",
        type=Path,
        default=Path("diagprm_dataset/clean_v2_common_aug_qa_plus_seed_v2"),
        help="Dataset directory containing clean_master_kg.json and common_disease_augments.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("diagprm_dataset/common_disease_rl_v1"),
    )
    parser.add_argument(
        "--augment_json",
        type=Path,
        default=None,
        help="Optional augment JSON. Defaults to base_dataset_dir/common_disease_augments.json.",
    )
    parser.add_argument("--train_cases_per_disease", type=int, default=24)
    parser.add_argument("--val_cases_per_disease", type=int, default=4)
    parser.add_argument("--test_cases_per_disease", type=int, default=4)
    parser.add_argument("--min_initial", type=int, default=2)
    parser.add_argument("--max_initial", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = args.base_dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    augment_path = args.augment_json.resolve() if args.augment_json else base_dir / "common_disease_augments.json"
    augments = load_augments(augment_path)
    base_kg = load_json(base_dir / "clean_master_kg.json")
    alias_map = build_alias_map(base_kg, augments)
    if not alias_map:
        raise ValueError("No common diseases matched the KG.")

    # Use natural canonical disease names so final diagnosis matching is less
    # brittle than rare-KG aliases such as "failure heart".
    disease_to_facts: dict[str, list[str]] = {}
    disease_to_matched_kg_name: dict[str, list[str]] = {}
    for augment_name, kg_name in alias_map.items():
        facts = augments[augment_name]
        if len(facts) >= 3:
            canonical = canonical_disease_name(augment_name)
            disease_to_facts[canonical] = merge_facts(disease_to_facts.get(canonical, []), facts)
            disease_to_matched_kg_name.setdefault(canonical, [])
            if kg_name not in disease_to_matched_kg_name[canonical]:
                disease_to_matched_kg_name[canonical].append(kg_name)

    common_kg = {
        disease: {fact: weight_for_rank(i) for i, fact in enumerate(facts)}
        for disease, facts in sorted(disease_to_facts.items())
    }
    (output_dir / "clean_master_kg.json").write_text(
        json.dumps(common_kg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    split_specs = {
        "train": args.train_cases_per_disease,
        "val": args.val_cases_per_disease,
        "test": args.test_cases_per_disease,
    }
    global_index = 0
    split_stats: dict[str, dict[str, Any]] = {}
    for split, cases_per_disease in split_specs.items():
        records = []
        for disease, facts in sorted(disease_to_facts.items()):
            for case_idx in range(cases_per_disease):
                records.append(
                    make_record(
                        disease=disease,
                        facts=facts,
                        split=split,
                        case_idx=case_idx,
                        global_index=global_index,
                        seed=args.seed,
                        min_initial=args.min_initial,
                        max_initial=args.max_initial,
                    )
                )
                global_index += 1
        random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[split]).shuffle(records)
        write_jsonl(output_dir / f"kg_{split}_dataset.jsonl", records)
        rows = [record_to_parquet_row(row, max_turns=args.max_turns) for row in records]
        pd.DataFrame(rows).to_parquet(output_dir / f"diagprm_{split}.parquet", index=False)
        split_stats[split] = {
            "rows": len(records),
            "cases_per_disease": cases_per_disease,
            "num_diseases": len(disease_to_facts),
        }

    augment_doc = {
        "description": "Common-disease-only RL dataset for DiagPRM.",
        "base_dataset_dir": str(base_dir),
        "augment_json": str(augment_path),
        "num_common_diseases": len(disease_to_facts),
        "diseases": sorted(disease_to_facts),
        "alias_map": alias_map,
        "canonical_to_matched_kg_names": disease_to_matched_kg_name,
        "split_stats": split_stats,
        "max_turns": args.max_turns,
        "note": "For RL, set DIAGPRM_DATASET to this output_dir and KG_PATH to output_dir/clean_master_kg.json.",
    }
    (output_dir / "common_disease_rl_report.json").write_text(
        json.dumps(augment_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "common_disease_augments_used.json").write_text(
        json.dumps({"augments": disease_to_facts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for filename in ["split_manifest.json", "clean_filter_report.md"]:
        src = base_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    print(json.dumps(augment_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
