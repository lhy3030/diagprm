"""
Build DiagPRM training & test datasets from master_kg.json.

Output files (in the same directory as this script):
  kg_train_dataset.jsonl   -- KG-derived training set  (~5000-8000 samples)
  kg_test_dataset.jsonl    -- KG hold-out test set     (200 samples)

Each sample follows the ATPO merged_train_dataset format so it can be dropped
directly into the existing training pipeline with no other changes.

Sample schema:
{
  "prompt": [
    {"role": "system", "content": <SYSTEM_PROMPT>},
    {"role": "user",   "content": "<initial_info>\nProblem: ...\nOptions: {...}"}
  ],
  "ground_truth": {
    "answer": "A",
    "answer_info": "<disease_name>",
    "disease": "<disease_name>",
    "symptoms_pool": {<symptom: weight, ...>}   # kept for reward computation
  },
  "agent_name": "user_assistant_interaction",
  "extra_info": {
    "atomic_facts": ["1. The patient has ...", ...]   # sampled at build time;
                                                      # can also be re-sampled
                                                      # at training time
  },
  "data_source": "kg",
  "index": <int>
}
"""

import json
import random
import re
import string
from pathlib import Path

# ── Symptom quality filter ────────────────────────────────────────────────────
# These patterns identify KG entries that are NOT real patient symptoms.
_BAD_SYMPTOM_PATTERNS = [
    r"\b\d+[\.\-]year\b",      # "1-year survival", "10-year mortality"
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
_BAD_SYMPTOM_RE = re.compile(
    "|".join(_BAD_SYMPTOM_PATTERNS), re.IGNORECASE
)


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


# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
KG_PATH    = SCRIPT_DIR.parent / "origin_dataset" / "master_kg.json"
TRAIN_OUT  = SCRIPT_DIR.parent / "diagprm_dataset" / "kg_train_dataset.jsonl"
TEST_OUT   = SCRIPT_DIR.parent / "diagprm_dataset" / "kg_test_dataset.jsonl"

# ── hyper-params ───────────────────────────────────────────────────────────────
SEED                 = 42
MIN_CLEAN_SYMPTOMS   = 5      # disease must have ≥ this many CLEAN symptoms
COMMON_SYMPTOM_RANGE = (10, 50)  # proxy for common diseases
KG_TEST_SIZE         = 200
DISTRACTOR_COUNT     = 3
INITIAL_REVEAL_COUNT = (1, 3)   # how many symptoms to show at turn-0
STRONG_WEIGHT        = 0.5

# ── system prompt (identical to ATPO's) ───────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a professional medical assistant, possessing outstanding medical "
    "diagnostic reasoning and analytical abilities, as well as strong clinical "
    "inquiry and patient assessment skills.\n\n"
    "Below, the user will provide initial patient information at the beginning "
    "of the first round of conversation, pose a single-choice question "
    "(Problem: question description), and give 4 options (Options: option "
    "descriptions). Your task is to, based on the question description, the "
    "option descriptions, the currently available patient information, and your "
    "own knowledge, select the correct option.\n\n"
    "Note: The initial patient information provided by the user in the first "
    "round is incomplete. You can ask the user questions to continuously obtain "
    "more patient information until you are confident enough to select the "
    "correct option.\n\n"
    "In each round of dialogue, you must first determine: Based on the question "
    "description, the option descriptions, the currently available patient "
    "information, and your own knowledge, do you have enough confidence to "
    "select the correct option?\n"
    "    - If you are not confident enough, output a specific question in the "
    "following format:\n"
    "      Question: [The specific question you want to ask]\n"
    "    - If you are confident enough, output your selection in the following "
    "format:\n"
    "      Final Answer: [Your chosen option]\n\n"
    "Important Notes:\n"
    "1. In each round of conversation, you must make a clear decision — either "
    "choose an option or ask a question. Do not be vague. When responding or "
    "asking, you must strictly follow the corresponding format.\n"
    "2. When choosing an option, you can only choose one from the provided "
    "options (e.g., A, B, C, etc.), and cannot choose multiple or include any "
    "other content.\n"
    "3. When asking a question, you can only ask one specific question at a "
    "time, cannot repeat questions that have already been asked, and cannot "
    "include any other content."
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_kg(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def clean_symptoms(symptoms: dict) -> dict:
    """Filter out noisy KG entries; return {symptom: weight} for clean ones."""
    return {s: w for s, w in symptoms.items() if is_clean_symptom(s)}


def title_case(name: str) -> str:
    return string.capwords(name)


def symptom_to_fact(symptom: str, idx: int) -> str:
    """Convert KG symptom string → readable atomic fact sentence."""
    s = symptom.strip().lower()
    # "pain, referred" → "referred pain"
    if re.match(r"^[^,]+,\s+\S", s):
        parts = [p.strip() for p in s.split(",", 1)]
        s = f"{parts[1]} {parts[0]}"
    return f"{idx}. The patient has {s}."


def sample_initial_info(disease: str) -> str:
    """Vague chief-complaint opener, deterministic per disease."""
    templates = [
        "A patient presents to the clinic with multiple complaints.",
        "A patient comes in with a history of several symptoms.",
        "A patient is brought to the emergency department with various symptoms.",
        "A patient seeks medical attention due to ongoing health issues.",
        "A patient presents with a complex medical history and seeks evaluation.",
    ]
    return random.Random(disease).choice(templates)


def build_options(correct_disease: str, option_pool: list,
                  rng: random.Random) -> tuple:
    """
    Build options dict {A,B,C,D}.
    Correct disease is placed at a random letter.
    3 distractors are sampled from option_pool.
    """
    letters = ["A", "B", "C", "D"]
    correct_idx = rng.randint(0, 3)
    candidates  = [d for d in option_pool if d != correct_disease]
    distractors = rng.sample(candidates, DISTRACTOR_COUNT)
    items       = distractors[:]
    items.insert(correct_idx, correct_disease)
    options = {l: title_case(items[i]) for i, l in enumerate(letters)}
    return options, letters[correct_idx]


def sample_atomic_facts(clean_pool: dict, rng: random.Random) -> list:
    """
    Reveal 1-3 clean symptoms as the initial turn-0 atomic_facts.
    Always include the highest-weight symptom so the doctor has a foothold.
    """
    sorted_syms = sorted(clean_pool.items(), key=lambda x: -x[1])
    n_reveal    = min(rng.randint(*INITIAL_REVEAL_COUNT), len(sorted_syms))

    revealed    = [sorted_syms[0][0]]          # top symptom always first
    top_half    = [s for s, _ in sorted_syms[1:max(2, len(sorted_syms) // 2)]]
    extra_n     = min(n_reveal - 1, len(top_half))
    revealed   += rng.sample(top_half, extra_n)

    return [symptom_to_fact(s, i + 1) for i, s in enumerate(revealed)]


def build_sample(idx: int, disease: str, clean_syms: dict,
                 option_pool: list, rng: random.Random) -> dict:
    options, correct_letter = build_options(disease, option_pool, rng)
    initial_info = sample_initial_info(disease)
    atomic_facts = sample_atomic_facts(clean_syms, rng)
    user_content = (
        f"{initial_info}\n"
        f"Problem: What is the most likely diagnosis?\n"
        f"Options: {json.dumps(options)}"
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "ground_truth": {
            "answer":        correct_letter,
            "answer_info":   title_case(disease),
            "disease":       disease,
            "symptoms_pool": clean_syms,   # full clean pool for reward function
        },
        "agent_name": "user_assistant_interaction",
        "extra_info": {
            "atomic_facts": atomic_facts,  # partial reveal at turn 0
        },
        "data_source": "kg",
        "index": idx,
    }


# ── main ──────────────────────────────────────────────────────────────────────

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
        kg[disease] = cleaned   # replace with clean version in-place
        if lo <= n <= hi:
            common_pool.append(disease)
        else:
            extended_pool.append(disease)

    print(f"  Noise-filtered symptom entries: {noise_filtered}")
    print(f"  Common-disease pool  (clean symptoms {lo}-{hi}): {len(common_pool)}")
    print(f"  Extended pool        (clean symptoms outside range): {len(extended_pool)}")

    # ── Step 2: select disease pool ───────────────────────────────────────────
    TARGET = 8000 + KG_TEST_SIZE
    use_common   = common_pool[:]
    need_extra   = max(0, TARGET - len(use_common))
    use_extended = rng.sample(extended_pool, min(need_extra, len(extended_pool)))
    all_selected = use_common + use_extended
    rng.shuffle(all_selected)
    print(f"  Total selected diseases: {len(all_selected)}")

    # ── Step 3: train / test split ────────────────────────────────────────────
    # Hold out KG_TEST_SIZE diseases from common_pool
    test_diseases  = rng.sample(common_pool, min(KG_TEST_SIZE, len(common_pool)))
    test_set       = set(test_diseases)
    train_diseases = [d for d in all_selected if d not in test_set]
    print(f"  Training diseases: {len(train_diseases)}")
    print(f"  Test diseases:     {len(test_diseases)}")

    option_pool = all_selected   # draw distractors from full selected pool

    # ── Step 4: build training samples ───────────────────────────────────────
    print("Building training samples ...")
    train_samples = []
    for i, disease in enumerate(train_diseases):
        sample = build_sample(i, disease, kg[disease], option_pool, rng)
        train_samples.append(sample)

    with open(TRAIN_OUT, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written {len(train_samples)} samples → {TRAIN_OUT}")

    # ── Step 5: build KG hold-out test samples ────────────────────────────────
    print("Building KG test samples ...")
    test_samples = []
    for i, disease in enumerate(test_diseases):
        sample = build_sample(i, disease, kg[disease], option_pool, rng)
        sample["data_source"] = "kg_test"
        test_samples.append(sample)

    with open(TEST_OUT, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written {len(test_samples)} samples → {TEST_OUT}")

    # ── Step 6: sanity check ─────────────────────────────────────────────────
    print("\n=== Sanity Check (first training sample) ===")
    s = train_samples[0]
    print(f"  disease      : {s['ground_truth']['disease']}")
    print(f"  correct_opt  : {s['ground_truth']['answer']} = "
          f"{s['ground_truth']['answer_info']}")
    opts = json.loads(s['prompt'][1]['content'].split('Options: ', 1)[1])
    print(f"  options      : {opts}")
    print(f"  atomic_facts : {s['extra_info']['atomic_facts']}")
    n_pool = len(s['ground_truth']['symptoms_pool'])
    print(f"  symptoms_pool: {n_pool} clean symptoms")

    # Verify no overlap
    train_set_diseases = {s['ground_truth']['disease'] for s in train_samples}
    test_set_diseases  = {s['ground_truth']['disease'] for s in test_samples}
    overlap = train_set_diseases & test_set_diseases
    print(f"\n  Train/test overlap: {len(overlap)} diseases (must be 0)")
    assert len(overlap) == 0, f"Overlap: {overlap}"

    # Show a few clean symptoms from a well-known disease
    for demo in ["type 2 diabetes mellitus", "hypertension", "pneumonia",
                 "depression"]:
        if demo in kg:
            top5 = sorted(kg[demo].items(), key=lambda x: -x[1])[:5]
            print(f"\n  {demo} clean top-5: {[s for s, _ in top5]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
