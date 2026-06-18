"""
Build clean DiagPRM experiment data.

This script creates a leakage-controlled v2 dataset under:
  diagprm_dataset/clean_v2/

Outputs:
  clean_master_kg.json
  clean_filter_report.md
  split_manifest.json
  kg_train_dataset.jsonl
  kg_val_dataset.jsonl
  kg_test_dataset.jsonl
  diagprm_train.parquet
  diagprm_val.parquet
  diagprm_test.parquet

The parquet schema is compatible with recipe.diagprm.diagprm_agent_loop.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_KG_PATH = ROOT_DIR / "diagprm_dataset" / "master_kg.json"
DEFAULT_OUT_DIR = ROOT_DIR / "diagprm_dataset" / "clean_v2"


BAD_DISEASE_NAMES = {
    "patient", "patients", "male", "female", "adult", "child", "children",
    "infant", "pregnancy", "infection", "pain", "fever", "cough",
    "rash", "fatigue", "nausea", "vomiting", "diarrhea", "headache",
}

DISEASE_GROUP_STOPWORDS = {
    "acute", "chronic", "severe", "mild", "moderate", "primary",
    "secondary", "familial", "congenital", "acquired", "idiopathic",
    "type", "types", "stage", "grade", "syndrome", "disease",
    "disorder", "deficiency", "i", "ii", "iii", "iv", "v",
}

TYPE_PATTERNS = {
    "outcome": [
        r"\bdeath\b", r"\bmortality\b", r"\bsurvival\b", r"\blethal\b",
        r"\bfatal\b", r"\bmorbidity\b", r"\bprognosis\b",
        r"\bcomplication\b", r"\bcomplications\b", r"\bsequelae\b",
        r"\blegal blindness\b",
        r"\bprogression\b", r"\birreversible\b",
        r"\badverse outcome\b", r"\borgan failure\b", r"\bremission\b",
        r"\bprolonged course\b",
        r"\bmetastasis\b", r"\bmetastases\b", r"\bmetastatic\b",
        r"\brelapse\b", r"\bspread\b",
    ],
    "epidemiology": [
        r"\bincidence\b", r"\bprevalence\b", r"\bepidemi", r"\bcohort\b",
        r"\bpopulation\b", r"\brisk factor\b", r"\bmean age\b",
    ],
    "treatment": [
        r"\btreatment\b", r"\btherapy\b", r"\btherapies\b", r"\bmanagement\b",
        r"\bsurgery\b", r"\bsurgical\b", r"\boperation\b", r"\bmedication\b",
        r"\bdrug\b", r"\bresponse to\b",
    ],
    "lab_imaging_test": [
        r"\bimaging\b", r"\bradiograph", r"\bx ray\b", r"\bx-ray\b",
        r"\bct\b", r"\bmri\b", r"\bultrasound\b", r"\bbiopsy\b",
        r"\bassay\b", r"\btest\b", r"\bresult\b", r"\blevel\b",
        r"\bcount\b", r"\bconcentration\b", r"\bserum\b", r"\bplasma\b",
        r"\burine\b", r"\bantibody\b", r"\bantibodies\b",
        r"\bbcva\b", r"\bvisual acuity score\b",
        r"\bt2wi\b", r"\bt1wi\b", r"\bhyperintense\b", r"\bhypointense\b",
        r"\bsignal\b", r"\breferable\b",
        r"\belectroretinogram\b", r"\bflicker response\b", r"\bresponse\b",
        r"\bultrasonography\b", r"\bultrasound\b", r"\bdetection\b",
        r"\belevated\b", r"\bpth\b",
    ],
    "molecular_pathology": [
        r"\bmutation\b", r"\bgene\b", r"\bgenetic\b", r"\bchromosom",
        r"\bprotein\b", r"\bcell\b", r"\bcellular\b", r"\bapoptosis\b",
        r"\bactivation\b", r"\bexpression\b", r"\bpathway\b",
        r"\bmyeloid\b", r"\bkeratinocyte\b",
        r"\batrophy\b", r"\bdeposit\b", r"\bdeposits\b", r"\bscarring\b",
        r"\bneovascularization\b", r"\btubulation\b", r"\bepithelium\b",
        r"\bchoriocapillaris\b", r"\bchorioretinal\b", r"\bretinal pigment\b",
        r"\bretinal\b.*\batrophy\b", r"\bdystrophy\b",
        r"\bcortical\b", r"\blayer\b", r"\bmucosa\b",
        r"\bfibrotic changes\b", r"\bbiologically active substances\b",
        r"\bmalignancy\b", r"\bmalignant\b", r"\btumors\b", r"\btumours\b",
        r"\bbrain tumors\b", r"\bsubtypes\b", r"\bbronchus\b", r"\blobe\b",
        r"\bpatholog", r"\bpathophysiology\b", r"\bproteinopathy\b",
        r"\btdp\b", r"\baggregation\b", r"\baggregations\b",
        r"\bimmunoglobulin\b", r"\bo-methyltransferase\b", r"\bactivity\b",
        r"\btissue\b", r"\bcells\b", r"\bglomeruli\b", r"\bcorpuscles\b",
        r"\btufts\b", r"\bfoot-process\b", r"\beffacement\b",
    ],
    "administrative_research": [
        r"\bstudy\b", r"\bstudies\b", r"\breport\b", r"\breports\b",
        r"\binitial report\b", r"\bcriteria\b", r"\bdiagnosis\b",
        r"\bdisease\b", r"\bpresentation\b", r"\bfinding\b", r"\bfindings\b",
        r"\bsymptom onset\b", r"\bonset\b", r"\bsigns and symptoms\b",
        r"\bmanifestation\b", r"\bclinical progression\b",
        r"\brecurrence\b", r"\btumou?r\b", r"\bmass appearance\b",
        r"\bage-related changes\b",
        r"\bclinical manifestations\b", r"\bmanifestations\b",
        r"\bdisorders\b",
        r"\bclinical features\b", r"\btypical clinical features\b",
        r"\bspecial face\b",
    ],
    "demographic_state": [
        r"\bpregnancy\b", r"\bpediatric\b", r"\bneonatal\b",
        r"\bmale\b", r"\bfemale\b",
    ],
}

BAD_SYMPTOM_RE = re.compile("|".join(p for ps in TYPE_PATTERNS.values() for p in ps), re.I)


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 '\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def disease_group_key(disease: str) -> str:
    norm = normalize(re.sub(r"\([^)]*\)", " ", disease))
    toks = [t for t in norm.split() if t not in DISEASE_GROUP_STOPWORDS and not t.isdigit()]
    if not toks:
        toks = norm.split()
    return " ".join(toks[:3])


def classify_symptom(symptom: str) -> str:
    s = normalize(symptom)
    for label, patterns in TYPE_PATTERNS.items():
        if any(re.search(p, s, re.I) for p in patterns):
            return label
    return "patient_reportable"


def is_clean_symptom(symptom: str) -> bool:
    s = normalize(symptom)
    if len(s) < 3 or len(s) > 80:
        return False
    if "," in symptom:
        return False
    if len(s.split()) > 8:
        return False
    if re.search(r"\d+\s*(mg|g|dl|mmol|iu|ml|kg|cm|mm|%)\b", s, re.I):
        return False
    if BAD_SYMPTOM_RE.search(s):
        return False
    return classify_symptom(s) == "patient_reportable"


def is_clean_disease(disease: str) -> bool:
    d = normalize(disease)
    if len(d) < 4 or d in BAD_DISEASE_NAMES:
        return False
    if len(d.split()) > 8:
        return False
    if re.search(r"\b(patient|patients|male|female|adult|child|children)\b", d):
        return False
    return True


def load_kg(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    kg = {}
    for disease, symptoms in raw.items():
        d = normalize(disease)
        if not is_clean_disease(d):
            continue
        if isinstance(symptoms, dict):
            kg[d] = {normalize(s): float(w) for s, w in symptoms.items() if s}
        elif isinstance(symptoms, list):
            kg[d] = {normalize(s): 1.0 for s in symptoms if s}
    return kg


def clean_kg(kg: Dict[str, Dict[str, float]], min_symptoms: int, max_symptoms: int):
    clean = {}
    report = {
        "raw_diseases": len(kg),
        "raw_edges": sum(len(v) for v in kg.values()),
        "dropped_by_type": Counter(),
        "dropped_other": 0,
        "kept_edges": 0,
        "dropped_diseases_too_few": 0,
        "dropped_diseases_too_many": 0,
    }

    for disease, symptoms in kg.items():
        kept = {}
        for sym, weight in symptoms.items():
            label = classify_symptom(sym)
            if not is_clean_symptom(sym):
                if label == "patient_reportable":
                    report["dropped_other"] += 1
                else:
                    report["dropped_by_type"][label] += 1
                continue
            kept[sym] = max(float(weight), kept.get(sym, 0.0))

        if len(kept) < min_symptoms:
            report["dropped_diseases_too_few"] += 1
            continue
        if len(kept) > max_symptoms:
            kept = dict(sorted(kept.items(), key=lambda x: -x[1])[:max_symptoms])
            report["dropped_diseases_too_many"] += 1
        clean[disease] = dict(sorted(kept.items(), key=lambda x: -x[1]))

    report["clean_diseases"] = len(clean)
    report["kept_edges"] = sum(len(v) for v in clean.values())
    return clean, report


def split_by_group(diseases: List[str], train_size: int, val_size: int, test_size: int, rng: random.Random):
    group_to_diseases = defaultdict(list)
    for d in diseases:
        group_to_diseases[disease_group_key(d)].append(d)

    groups = list(group_to_diseases)
    rng.shuffle(groups)

    splits = {"train": [], "val": [], "test": []}
    targets = {"train": train_size, "val": val_size, "test": test_size}
    for split in ["test", "val", "train"]:
        for g in list(groups):
            if len(splits[split]) >= targets[split]:
                break
            splits[split].extend(group_to_diseases[g])
            groups.remove(g)

    if len(splits["train"]) < train_size:
        for g in groups:
            splits["train"].extend(group_to_diseases[g])
            if len(splits["train"]) >= train_size:
                break

    return {k: v[:targets[k]] for k, v in splits.items()}, group_to_diseases


def sample_initial_symptoms(symptoms: Dict[str, float], rng: random.Random, lo: int, hi: int) -> List[str]:
    sorted_syms = sorted(symptoms.items(), key=lambda x: -x[1])
    n = min(rng.randint(lo, hi), len(sorted_syms))
    initial = [sorted_syms[0][0]]
    if n > 1:
        pool = [s for s, _ in sorted_syms[1:max(2, len(sorted_syms) // 2)]]
        initial.extend(rng.sample(pool, min(n - 1, len(pool))))
    return initial


CC_TEMPLATES = [
    "A patient presents to clinic for evaluation. The patient reports: {symptoms}.",
    "A patient comes in because of ongoing symptoms. The patient reports: {symptoms}.",
    "A patient seeks medical attention today. The patient reports: {symptoms}.",
    "A patient arrives for a diagnostic consultation. The patient reports: {symptoms}.",
]


def build_symptom_facts(symptoms: Dict[str, float]) -> List[dict]:
    return [
        {"fact_id": f"F{i:03d}", "text": symptom, "weight": float(weight)}
        for i, (symptom, weight) in enumerate(symptoms.items())
    ]


def build_sample(idx: int, split: str, disease: str, symptoms: Dict[str, float], rng: random.Random):
    initial = sample_initial_symptoms(symptoms, rng, 1, 3)
    chief = rng.choice(CC_TEMPLATES).format(symptoms=", ".join(initial))
    return {
        "disease": disease,
        "disease_id": f"D{idx:06d}",
        "disease_group": disease_group_key(disease),
        "chief_complaint": chief,
        "initial_symptoms": initial,
        "symptoms_pool": symptoms,
        "symptom_facts": build_symptom_facts(symptoms),
        "split": split,
        "visible_kg_policy": "none_by_default",
        "data_source": "kg_clean_v2",
        "index": idx,
    }


def to_parquet_rows(samples: List[dict], max_turns: int):
    rows = []
    for s in samples:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an experienced diagnostic physician conducting a "
                    f"multi-turn consultation. Maximum turns: {max_turns}."
                ),
            },
            {"role": "user", "content": s["chief_complaint"]},
        ]
        reward_model = {
            "ground_truth": {
                "disease": s["disease"],
                "disease_id": s["disease_id"],
                "disease_group": s["disease_group"],
                "symptoms_pool": s["symptoms_pool"],
                "symptom_facts": s["symptom_facts"],
                "initial_symptoms": s["initial_symptoms"],
                "split": s["split"],
            }
        }
        extra_info = {
            "index": s["index"],
            "disease_id": s["disease_id"],
            "disease_group": s["disease_group"],
            "chief_complaint": s["chief_complaint"],
            "initial_symptoms": s["initial_symptoms"],
            "n_symptoms_pool": len(s["symptoms_pool"]),
            "n_symptom_facts": len(s["symptom_facts"]),
            "visible_kg_policy": s["visible_kg_policy"],
        }
        rows.append({
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "reward_model": json.dumps(reward_model, ensure_ascii=False),
            "data_source": s["data_source"],
            "agent_name": "diagprm_interaction",
            "extra_info": json.dumps(extra_info, ensure_ascii=False),
        })
    return rows


def write_jsonl(path: Path, samples: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict, splits: dict, clean_kg: dict, manifest: dict):
    counts = [len(v) for v in clean_kg.values()]
    type_counts = dict(report["dropped_by_type"])
    lines = [
        "# Clean KG Filter Report",
        "",
        "## Summary",
        "",
        f"- Raw diseases: {report['raw_diseases']}",
        f"- Raw edges: {report['raw_edges']}",
        f"- Clean diseases: {report['clean_diseases']}",
        f"- Clean edges: {report['kept_edges']}",
        f"- Avg clean symptoms/disease: {sum(counts) / max(len(counts), 1):.2f}",
        f"- Min/Max clean symptoms/disease: {min(counts) if counts else 0} / {max(counts) if counts else 0}",
        "",
        "## Dropped Symptom Types",
        "",
    ]
    for label, n in sorted(type_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {label}: {n}")
    lines.extend([
        f"- other string-quality filters: {report['dropped_other']}",
        "",
        "## Dropped Diseases",
        "",
        f"- Too few clean symptoms: {report['dropped_diseases_too_few']}",
        f"- Truncated because too many symptoms: {report['dropped_diseases_too_many']}",
        "",
        "## Split Sizes",
        "",
        f"- Train diseases: {len(splits['train'])}",
        f"- Val diseases: {len(splits['val'])}",
        f"- Test diseases: {len(splits['test'])}",
        f"- Disease group overlap train/val: {len(set(manifest['groups']['train']) & set(manifest['groups']['val']))}",
        f"- Disease group overlap train/test: {len(set(manifest['groups']['train']) & set(manifest['groups']['test']))}",
        f"- Disease group overlap val/test: {len(set(manifest['groups']['val']) & set(manifest['groups']['test']))}",
        "",
        "## Notes",
        "",
        "- Patient and Reward Manager may access hidden `symptoms_pool` and `disease`.",
        "- Patient emits hidden `fact_id`; Reward Manager resolves it through `symptom_facts` / `fact_id_to_text`.",
        "- Doctor policy receives only chief complaint and subsequent dialogue unless a KG tool condition is explicitly enabled.",
        "- Initial symptoms define S0 and should not be rewarded as Doctor-discovered evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_path", type=Path, default=DEFAULT_KG_PATH)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train_size", type=int, default=6000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--min_symptoms", type=int, default=5)
    parser.add_argument("--max_symptoms", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_kg = load_kg(args.kg_path)
    clean, report = clean_kg(raw_kg, args.min_symptoms, args.max_symptoms)
    diseases = sorted(clean)
    rng.shuffle(diseases)

    splits, _ = split_by_group(diseases, args.train_size, args.val_size, args.test_size, rng)
    manifest = {
        "seed": args.seed,
        "source_kg": str(args.kg_path),
        "clean_kg": "clean_master_kg.json",
        "sizes": {k: len(v) for k, v in splits.items()},
        "groups": {k: sorted({disease_group_key(d) for d in v}) for k, v in splits.items()},
        "diseases": {k: sorted(v) for k, v in splits.items()},
    }

    samples_by_split = {}
    global_idx = 0
    for split in ["train", "val", "test"]:
        samples = []
        for disease in splits[split]:
            samples.append(build_sample(global_idx, split, disease, clean[disease], rng))
            global_idx += 1
        samples_by_split[split] = samples

    with open(args.out_dir / "clean_master_kg.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    with open(args.out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_jsonl(args.out_dir / "kg_train_dataset.jsonl", samples_by_split["train"])
    write_jsonl(args.out_dir / "kg_val_dataset.jsonl", samples_by_split["val"])
    write_jsonl(args.out_dir / "kg_test_dataset.jsonl", samples_by_split["test"])

    for split, parquet_name in [
        ("train", "diagprm_train.parquet"),
        ("val", "diagprm_val.parquet"),
        ("test", "diagprm_test.parquet"),
    ]:
        rows = to_parquet_rows(samples_by_split[split], args.max_turns)
        pd.DataFrame(rows).to_parquet(args.out_dir / parquet_name, index=False)

    write_report(args.out_dir / "clean_filter_report.md", report, splits, clean, manifest)

    print(f"Saved clean data to {args.out_dir}")
    print(f"Clean diseases: {len(clean)}")
    print(f"Train/val/test: {len(samples_by_split['train'])}/{len(samples_by_split['val'])}/{len(samples_by_split['test'])}")
    print(f"Clean KG: {args.out_dir / 'clean_master_kg.json'}")
    print(f"Train parquet: {args.out_dir / 'diagprm_train.parquet'}")


if __name__ == "__main__":
    main()
