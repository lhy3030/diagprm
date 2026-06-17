"""
DiagPRM data preparation script.

Converts kg_train_dataset.jsonl (new minimal schema) into a verl-compatible
parquet file.  The parquet is used only as a placeholder for the verl dataloader;
all real fields (disease, chief_complaint, symptoms_pool, initial_symptoms) are
passed through as-is so the DiagPRMAgentLoop can read them directly.

New minimal jsonl schema (produced by build_kg_dataset.py):
{
  "disease":          "polycythaemia vera",
  "chief_complaint":  "A patient presents ...",
  "symptoms_pool":    {"microvascular circulation disturbances": 0.395, ...},
  "initial_symptoms": ["microvascular circulation disturbances", ...],
  "data_source":      "kg",
  "index":            0
}

Output parquet schema (one row per sample):
{
  "prompt":           JSON string of [{"role":"system",...},{"role":"user",...}]
  "reward_model":     JSON string of {"ground_truth":{"disease":...,"symptoms_pool":{...}}}
  "data_source":      "diagprm_kg"
  "agent_name":       "diagprm_interaction"
  "extra_info":       JSON string of {"disease":..., "initial_symptoms":[...], ...}
}

The verl dataloader deserialises "prompt" into raw_prompt; the agent loop then
reads chief_complaint, disease, symptoms_pool, initial_symptoms from extra_info
(passed through kwargs) rather than parsing the prompt string.

Usage:
  python recipe/diagprm/data_utils/prepare_diagprm_data.py \
    --input /path/to/kg_train_dataset.jsonl \
    --output_dir /path/to/output \
    --val_ratio 0.05 \
    --max_turns 10
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from recipe.diagprm.prompts import DOCTOR_SYSTEM_PROMPT, DOCTOR_INITIAL_PROMPT


def build_prompt_messages(chief_complaint: str, max_turns: int = 10) -> List[Dict]:
    """Build the initial [system, user] message list for the verl prompt field."""
    return [
        {
            "role": "system",
            "content": DOCTOR_SYSTEM_PROMPT.format(
                max_turns=max_turns,
                current_turn=1,
            ),
        },
        {
            "role": "user",
            "content": DOCTOR_INITIAL_PROMPT.format(chief_complaint=chief_complaint),
        },
    ]


def process_record(record: Dict, max_turns: int = 10) -> Optional[Dict]:
    """
    Convert a single kg_train_dataset.jsonl record into the parquet row format.

    Handles both the new minimal schema and the old ATPO MCQA schema for
    backward compatibility.
    """
    # ── New minimal schema ──────────────────────────────────────────────────
    disease          = record.get("disease", "")
    chief_complaint  = record.get("chief_complaint", "")
    symptoms_pool    = record.get("symptoms_pool", {})
    initial_symptoms = record.get("initial_symptoms", [])
    data_source      = record.get("data_source", "kg")

    # ── Old ATPO MCQA schema fallback ───────────────────────────────────────
    if not disease or not chief_complaint:
        # Try to extract from the old nested format
        ground_truth = record.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            disease = ground_truth.get("disease", ground_truth.get("answer_info", ""))
            symptoms_pool    = ground_truth.get("symptoms_pool", {})
            initial_symptoms = ground_truth.get("initial_symptoms", [])
            # Legacy atomic_facts (sentence form) -- kept for back-compat only
            if not initial_symptoms:
                initial_symptoms = ground_truth.get("atomic_facts", [])
        # Extract chief_complaint from prompt user message
        for msg in record.get("prompt", []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                # Strip MCQA "Problem:" and "Options:" sections
                m = re.search(r'\nProblem:', content)
                if m:
                    chief_complaint = content[:m.start()].strip()
                else:
                    chief_complaint = content.strip()
                break

    if not disease or len(chief_complaint) < 10:
        return None

    # ── Build parquet row ───────────────────────────────────────────────────
    prompt_messages = build_prompt_messages(chief_complaint, max_turns=max_turns)

    reward_model_info = {
        "ground_truth": {
            "disease":          disease,
            "symptoms_pool":    symptoms_pool,
            "initial_symptoms": initial_symptoms,
        }
    }

    extra_info = {
        # Top-level fields repeated here so the agent loop can access them
        # directly via kwargs without re-parsing reward_model JSON.
        "disease":          disease,
        "chief_complaint":  chief_complaint,
        "symptoms_pool":    symptoms_pool,
        "initial_symptoms": initial_symptoms,
        "original_data_source": data_source,
        "n_symptoms_pool":  len(symptoms_pool),
        "n_initial_symptoms": len(initial_symptoms),
    }

    return {
        "prompt":       prompt_messages,
        "reward_model": reward_model_info,
        "data_source":  f"diagprm_{data_source}",
        "agent_name":   "diagprm_interaction",
        "extra_info":   extra_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DiagPRM training data")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/liuhaoyu/iclr_2027/diagprm/diagprm_dataset/kg_train_dataset.jsonl",
        help="Input JSONL file (new minimal schema from build_kg_dataset.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/liuhaoyu/iclr_2027/diagprm/diagprm_dataset",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use as validation set",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum number of dialogue turns",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading input: {args.input}")
    records = []
    skipped = 0

    with open(args.input) as f:
        for i, line in enumerate(tqdm(f, desc="Processing")):
            if args.max_samples and i >= args.max_samples:
                break
            try:
                raw = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            processed = process_record(raw, max_turns=args.max_turns)
            if processed is None:
                skipped += 1
                continue

            records.append(processed)

    print(f"\nProcessed : {len(records):,}")
    print(f"Skipped   : {skipped:,}")

    # Train / val split
    import random
    random.seed(42)
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records   = records[:n_val]
    train_records = records[n_val:]
    print(f"Train: {len(train_records):,}  Val: {len(val_records):,}")

    def save_parquet(data: List[Dict], path: str):
        rows = []
        for rec in data:
            rows.append({
                "prompt":       json.dumps(rec["prompt"],       ensure_ascii=False),
                "reward_model": json.dumps(rec["reward_model"], ensure_ascii=False),
                "data_source":  rec["data_source"],
                "agent_name":   rec["agent_name"],
                "extra_info":   json.dumps(rec["extra_info"],   ensure_ascii=False),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        print(f"Saved {len(df):,} records → {path}")

    save_parquet(train_records, os.path.join(args.output_dir, "diagprm_train.parquet"))
    save_parquet(val_records,   os.path.join(args.output_dir, "diagprm_val.parquet"))

    # Sample printout
    print("\n--- Sample record ---")
    s = train_records[0]
    ei = s["extra_info"]
    print(f"disease          : {ei['disease']}")
    print(f"chief_complaint  : {ei['chief_complaint'][:120]}")
    print(f"initial_symptoms : {ei['initial_symptoms']}")
    print(f"symptoms_pool sz : {ei['n_symptoms_pool']}")
    print("Done!")


if __name__ == "__main__":
    main()
