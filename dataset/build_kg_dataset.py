"""
Build DiagPRM training & test datasets from master_kg.json.

Output files (in diagprm_dataset/):
  kg_train_dataset.jsonl   -- training set  (~5000-8000 samples)
  kg_test_dataset.jsonl    -- hold-out test set (200 samples)

Each sample schema (minimal, RL-ready):
{
  "disease":         "polycythaemia vera",          # GT for reward
  "chief_complaint": "A patient presents ...",      # Doctor Agent turn-0 user msg
  "symptoms_pool":   {                              # Patient Agent's full fact sheet
    "microvascular circulation disturbances": 0.395,  # KG verbatim keys
    "transient ischaemic attack": 0.345,              # used directly for KG coverage delta
    ...
  },
  "initial_symptoms": [                            # top-1~3 KG verbatim strings
    "microvascular circulation disturbances",       # revealed at turn-0 to Doctor
    "transient ischaemic attack"
  ],
  "data_source": "kg",
  "index": 0
}

Design rationale:
  - symptoms_pool keys are KG verbatim strings.
  - initial_symptoms are the SAME verbatim strings (no sentence wrapping).
  - Patient Simulator receives initial_symptoms and is instructed to copy one
    verbatim into <fact>; the reward function matches <fact> directly against
    symptoms_pool keys -- no mismatch possible.
  - No MCQA wrapper (options / problem), no old system prompt.
  - .parquet generation is handled by prepare_diagprm_data.py (placeholder role).
"""

import json
import random
import re
import string
from pathlib import Path

# ── Symptom quality filter ────────────────────────────────────────────────────
_BAD_SYMPTOM_PATTERNS = [
    r"\b\d+[\.\-]year\b",
    r"\bsurvival\b",
    r"\bmortality\b",
    r"\bincidence\b",
    r"\bprevalence\b",
    r"\bprognosis\b",
    r"\bepidemi",
    r"^symptom\b",
    r"\bpresentation\b",
    r"\bdiagnosis\b",
    r"\bmanagement\b",
    r"\btreatment\b",
    r"\btherapy\b",
    r"\bhistory\b",
    r"\bstudy\b|\bstudies\b",
    r"\bimaging\b",
    r"\bradiograph",
    r"\bfinding[s]?\b",
    r"\bresult[s]?\b",
    r"\bbiopsy\b",
    r"\bcriteria\b",
    r"\bpatholog",
    r"^\d+[%\.]",
    r"^[\(\[]",
]
_BAD_SYMPTOM_RE = re.compile("|".join(_BAD_SYMPTOM_PATTERNS), re.IGNORECASE)


def is_clean_symptom(sym: str) -> bool:
    """Return True if the symptom string is a usable patient-reportable finding."""
    if len(sym) < 3 or len(sym) > 120:
        return False
    if _BAD_SYMPTOM_RE.search(sym):
        return False
    # Lab value with units
    if re.search(r"\d+\s*(mg|g|dl|mmol|iu|ml|kg|cm|mm)\b", sym, re.I):
        return False
    return True


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
KG_PATH    = SCRIPT_DIR.parent / "diagprm_dataset" / "master_kg.json"
TRAIN_OUT  = SCRIPT_DIR.parent / "diagprm_dataset" / "kg_train_dataset.jsonl"
TEST_OUT   = SCRIPT_DIR.parent / "diagprm_dataset" / "kg_test_dataset.jsonl"

# ── Hyper-params ───────────────────────────────────────────────────────────────
SEED                 = 42
MIN_CLEAN_SYMPTOMS   = 5           # disease must have >= this many clean symptoms
COMMON_SYMPTOM_RANGE = (10, 50)    # proxy for "common / learnable" diseases
KG_TEST_SIZE         = 200
INITIAL_REVEAL_COUNT = (1, 3)      # how many symptoms to reveal at turn-0

# Chief-complaint templates (vague -- real info comes from initial_symptoms)
_CC_TEMPLATES = [
    "A patient presents to the clinic with multiple complaints.",
    "A patient comes in with a history of several symptoms.",
    "A patient is brought to the emergency department with various symptoms.",
    "A patient seeks medical attention due to ongoing health issues.",
    "A patient presents with a complex medical history and seeks evaluation.",
]


def load_kg(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def clean_symptoms(symptoms: dict) -> dict:
    """Filter out noisy KG entries; return {symptom: weight} for clean ones."""
    return {s: w for s, w in symptoms.items() if is_clean_symptom(s)}


def sample_initial_symptoms(clean_pool: dict, rng: random.Random) -> list:
    """
    Select 1-3 symptoms to reveal at turn-0.

    Returns a list of KG verbatim symptom strings (NOT sentences).
    - The highest-weight symptom is always included.
    - Extra symptoms are sampled from the top-50% pool.

    These strings are passed as-is to the Patient Simulator and written into
    the chief_complaint so the Doctor has an initial foothold.
    """
    sorted_syms = sorted(clean_pool.items(), key=lambda x: -x[1])
    n_reveal = min(rng.randint(*INITIAL_REVEAL_COUNT), len(sorted_syms))

    revealed = [sorted_syms[0][0]]   # top-weight symptom always first
    top_half = [s for s, _ in sorted_syms[1 : max(2, len(sorted_syms) // 2)]]
    extra_n  = min(n_reveal - 1, len(top_half))
    revealed += rng.sample(top_half, extra_n)
    return revealed


def build_chief_complaint(disease: str, initial_symptoms: list) -> str:
    """
    Build a natural chief-complaint sentence from the vague template +
    the revealed initial symptoms (KG verbatim strings appended naturally).
    """
    template = random.Random(disease).choice(_CC_TEMPLATES)
    if initial_symptoms:
        syms_str = ", ".join(initial_symptoms)
        return f"{template} The patient reports: {syms_str}."
    return template


def build_sample(
    idx: int,
    disease: str,
    clean_syms: dict,
    rng: random.Random,
) -> dict:
    initial_symptoms = sample_initial_symptoms(clean_syms, rng)
    chief_complaint  = build_chief_complaint(disease, initial_symptoms)
    return {
        "disease":          disease,          # GT verbatim (lower-cased KG key)
        "chief_complaint":  chief_complaint,  # Doctor Agent turn-0 user content
        "symptoms_pool":    clean_syms,       # {verbatim_symptom: weight} -- full fact sheet
        "initial_symptoms": initial_symptoms, # verbatim KG strings revealed at turn-0
        "data_source":      "kg",
        "index":            idx,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    print(f"Loading KG from {KG_PATH} ...")
    kg = load_kg(KG_PATH)
    print(f"  Total diseases in KG: {len(kg)}")

    # ── Step 1: clean symptoms and filter diseases ────────────────────────────
    lo, hi         = COMMON_SYMPTOM_RANGE
    common_pool    = []
    extended_pool  = []
    noise_filtered = 0

    for disease, symptoms in kg.items():
        cleaned = clean_symptoms(symptoms)
        n       = len(cleaned)
        noise_filtered += len(symptoms) - n
        if n < MIN_CLEAN_SYMPTOMS:
            continue
        kg[disease] = cleaned
        if lo <= n <= hi:
            common_pool.append(disease)
        else:
            extended_pool.append(disease)

    print(f"  Noise-filtered symptom entries : {noise_filtered}")
    print(f"  Common pool  (clean syms {lo}-{hi}): {len(common_pool)}")
    print(f"  Extended pool (outside range)  : {len(extended_pool)}")

    # ── Step 2: select disease pool ───────────────────────────────────────────
    TARGET       = 8000 + KG_TEST_SIZE
    use_common   = common_pool[:]
    need_extra   = max(0, TARGET - len(use_common))
    use_extended = rng.sample(extended_pool, min(need_extra, len(extended_pool)))
    all_selected = use_common + use_extended
    rng.shuffle(all_selected)
    print(f"  Total selected diseases        : {len(all_selected)}")

    # ── Step 3: train / test split ────────────────────────────────────────────
    test_diseases  = rng.sample(common_pool, min(KG_TEST_SIZE, len(common_pool)))
    test_set       = set(test_diseases)
    train_diseases = [d for d in all_selected if d not in test_set]
    print(f"  Training diseases : {len(train_diseases)}")
    print(f"  Test diseases     : {len(test_diseases)}")

    # ── Step 4: build training samples ───────────────────────────────────────
    print("Building training samples ...")
    train_samples = []
    for i, disease in enumerate(train_diseases):
        sample = build_sample(i, disease, kg[disease], rng)
        train_samples.append(sample)

    with open(TRAIN_OUT, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written {len(train_samples)} samples → {TRAIN_OUT}")

    # ── Step 5: build test samples ────────────────────────────────────────────
    print("Building KG test samples ...")
    test_samples = []
    for i, disease in enumerate(test_diseases):
        sample = build_sample(i, disease, kg[disease], rng)
        sample["data_source"] = "kg_test"
        test_samples.append(sample)

    with open(TEST_OUT, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written {len(test_samples)} samples → {TEST_OUT}")

    # ── Step 6: sanity check ─────────────────────────────────────────────────
    print("\n=== Sanity Check (first training sample) ===")
    s = train_samples[0]
    print(f"  disease           : {s['disease']}")
    print(f"  chief_complaint   : {s['chief_complaint']}")
    print(f"  initial_symptoms  : {s['initial_symptoms']}")
    print(f"  symptoms_pool size: {len(s['symptoms_pool'])}")
    # Verify initial_symptoms are all keys in symptoms_pool
    for sym in s["initial_symptoms"]:
        assert sym in s["symptoms_pool"], f"initial_symptom '{sym}' not in symptoms_pool!"
    print("  initial_symptoms all in symptoms_pool: OK")

    # Verify no train/test overlap
    train_set_diseases = {s["disease"] for s in train_samples}
    test_set_diseases  = {s["disease"] for s in test_samples}
    overlap = train_set_diseases & test_set_diseases
    print(f"\n  Train/test disease overlap: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, f"Overlap detected: {overlap}"

    # Show clean top-5 for a well-known disease
    for demo in ["type 2 diabetes mellitus", "hypertension", "pneumonia", "depression"]:
        if demo in kg:
            top5 = sorted(kg[demo].items(), key=lambda x: -x[1])[:5]
            print(f"\n  {demo} clean top-5: {[sym for sym, _ in top5]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
