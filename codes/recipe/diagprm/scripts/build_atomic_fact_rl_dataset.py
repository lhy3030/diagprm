#!/usr/bin/env python3
"""Build ATPO-style atomic-fact RL datasets from local medical QA sources.

The output keeps the standard verl/ATPO columns:
  prompt, reward_model, data_source, agent_name, extra_info

Each sample also carries a normalized list of hidden atomic facts in extra_info.
The agent sees only an incomplete initial context plus the multiple-choice
question/options; the user/case simulator can answer from hidden atomic facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.diagprm.prompts.atomic_fact import (
    ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT,
    ATOMIC_FACT_SFT_MAX_TURNS,
)

DEFAULT_SYSTEM_PROMPT = ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT.format(
    max_turns=ATOMIC_FACT_SFT_MAX_TURNS
)


ATPO_STYLE_SOURCES = {
    "medqa_test": "MedQA/medqa_test_dataset.jsonl",
    "medicalexam_test": "MedicalExam/aie_test_dataset.jsonl",
    "medmcqa_test": "MedMCQA/mcqa_test_dataset.jsonl",
    "medqa_diag_test": "medqa_diag_test.jsonl",
}


MEDIQ_SOURCES = {
    "mediq_dev": "mediQ/medqa_dev_convo.jsonl",
    "mediq_test": "mediQ/medqa_test_convo.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_atomic_fact(text: Any) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^\s*(?:[-*]|\d+[\).\s:]+)\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_facts(facts: Any) -> list[str]:
    if not isinstance(facts, list):
        return []
    out = []
    seen = set()
    for fact in facts:
        s = normalize_atomic_fact(fact)
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def stable_uid(*parts: Any) -> str:
    text = "||".join(str(x) for x in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def get_options_text(options: Any) -> str:
    if isinstance(options, dict):
        return json.dumps(options, ensure_ascii=False)
    return str(options or "")


def get_initial_context_from_prompt(prompt: list[dict[str, Any]]) -> str:
    if not prompt:
        return ""
    user = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
    return str(user).split("Problem:", 1)[0].strip()


def make_user_content(initial_context: str, question: str, options: Any) -> str:
    pieces = []
    if initial_context:
        pieces.append(str(initial_context).strip())
    pieces.append(f"Problem: {str(question).strip()}")
    pieces.append(f"Options: {get_options_text(options)}")
    return "\n".join(pieces)


def make_prompt(initial_context: str, question: str, options: Any) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": make_user_content(initial_context, question, options)},
    ]


def make_record(
    *,
    prompt: list[dict[str, Any]],
    answer_idx: Any,
    answer_info: Any,
    atomic_facts: list[str],
    source_name: str,
    source_path: str,
    source_index: Any,
    question: Any,
    options: Any,
    split: str,
    initial_context: str,
    original: dict[str, Any],
) -> dict[str, Any]:
    uid = stable_uid(source_name, source_index, question, answer_idx, answer_info)
    fact_dicts = [
        {"fact_id": f"F{i:03d}", "text": fact, "weight": 1.0}
        for i, fact in enumerate(atomic_facts)
    ]
    reward_model = {
        "style": "rule",
        "ground_truth": {
            "answer": str(answer_idx).strip(),
            "answer_info": str(answer_info or "").strip(),
        },
    }
    extra_info = {
        "index": source_index,
        "uid": uid,
        "split": split,
        "source_name": source_name,
        "source_path": source_path,
        "prompt": prompt,
        "question": str(question or "").strip(),
        "options": options,
        "answer_idx": str(answer_idx).strip(),
        "answer_info": str(answer_info or "").strip(),
        "initial_context": initial_context,
        "atomic_facts": atomic_facts,
        "atomic_fact_items": fact_dicts,
        "n_atomic_facts": len(atomic_facts),
        "original_keys": sorted(original.keys()),
    }
    return {
        "prompt": prompt,
        "reward_model": reward_model,
        "data_source": f"atomic_fact_{source_name}",
        "agent_name": "atomic_fact_interaction",
        "extra_info": extra_info,
        "uid": uid,
        "source_name": source_name,
        "source_index": source_index,
        "split": split,
        "question": str(question or "").strip(),
        "options": options,
        "answer_idx": str(answer_idx).strip(),
        "answer_info": str(answer_info or "").strip(),
        "initial_context": initial_context,
        "atomic_facts": atomic_facts,
        "n_atomic_facts": len(atomic_facts),
    }


def convert_atpo_style(path: Path, source_name: str, split: str) -> tuple[list[dict[str, Any]], Counter]:
    rows = []
    stats = Counter()
    for local_idx, obj in enumerate(read_jsonl(path)):
        facts = normalize_facts((obj.get("extra_info") or {}).get("atomic_facts"))
        if not facts:
            stats["skipped_no_atomic_facts"] += 1
            continue
        prompt = obj.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            stats["skipped_bad_prompt"] += 1
            continue
        gt = obj.get("ground_truth") or {}
        answer_idx = gt.get("answer")
        answer_info = gt.get("answer_info", "")
        if answer_idx is None:
            stats["skipped_no_answer"] += 1
            continue
        user_content = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
        question = ""
        m = re.search(r"Problem:\s*(.*?)(?:\nOptions:|$)", str(user_content), re.S)
        if m:
            question = m.group(1).strip()
        options = {}
        m = re.search(r"Options:\s*(.*)$", str(user_content), re.S)
        if m:
            options = m.group(1).strip()
        initial_context = get_initial_context_from_prompt(prompt)
        source_index = obj.get("index", local_idx)
        rows.append(
            make_record(
                prompt=prompt,
                answer_idx=answer_idx,
                answer_info=answer_info,
                atomic_facts=facts,
                source_name=source_name,
                source_path=str(path),
                source_index=source_index,
                question=question,
                options=options,
                split=split,
                initial_context=initial_context,
                original=obj,
            )
        )
        stats["kept"] += 1
    return rows, stats


def convert_atpo_merged_train(path: Path, split: str) -> tuple[list[dict[str, Any]], Counter]:
    rows = []
    stats = Counter()
    for local_idx, obj in enumerate(read_jsonl(path)):
        facts = normalize_facts((obj.get("extra_info") or {}).get("atomic_facts"))
        if not facts:
            stats["skipped_no_atomic_facts"] += 1
            continue
        prompt = obj.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            stats["skipped_bad_prompt"] += 1
            continue
        gt = obj.get("ground_truth") or {}
        answer_idx = gt.get("answer")
        answer_info = gt.get("answer_info", "")
        if answer_idx is None:
            stats["skipped_no_answer"] += 1
            continue
        source_base = str(obj.get("data_source") or "atpo")
        source_name = f"atpo_train_{source_base}"
        user_content = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
        question = ""
        m = re.search(r"Problem:\s*(.*?)(?:\nOptions:|$)", str(user_content), re.S)
        if m:
            question = m.group(1).strip()
        options = {}
        m = re.search(r"Options:\s*(.*)$", str(user_content), re.S)
        if m:
            options = m.group(1).strip()
        initial_context = get_initial_context_from_prompt(prompt)
        source_index = obj.get("index", local_idx)
        rows.append(
            make_record(
                prompt=prompt,
                answer_idx=answer_idx,
                answer_info=answer_info,
                atomic_facts=facts,
                source_name=source_name,
                source_path=str(path),
                source_index=source_index,
                question=question,
                options=options,
                split=split,
                initial_context=initial_context,
                original=obj,
            )
        )
        stats[f"kept_{source_base}"] += 1
        stats["kept"] += 1
    return rows, stats


def convert_mediq(path: Path, source_name: str, split: str) -> tuple[list[dict[str, Any]], Counter]:
    rows = []
    stats = Counter()
    for local_idx, obj in enumerate(read_jsonl(path)):
        facts = normalize_facts(obj.get("atomic_facts"))
        if not facts:
            stats["skipped_no_atomic_facts"] += 1
            continue
        context = obj.get("context") or []
        if isinstance(context, list) and context:
            initial_context = str(context[0]).strip()
        else:
            initial_context = str(obj.get("question", "")).strip()
        question = obj.get("question", "")
        options = obj.get("options", {})
        answer_idx = obj.get("answer_idx")
        answer_info = obj.get("answer", "")
        if answer_idx is None:
            stats["skipped_no_answer"] += 1
            continue
        source_index = obj.get("id", local_idx)
        rows.append(
            make_record(
                prompt=make_prompt(initial_context, question, options),
                answer_idx=answer_idx,
                answer_info=answer_info,
                atomic_facts=facts,
                source_name=source_name,
                source_path=str(path),
                source_index=source_index,
                question=question,
                options=options,
                split=split,
                initial_context=initial_context,
                original=obj,
            )
        )
        stats["kept"] += 1
    return rows, stats


def split_train_val(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_ratio))) if rows else 0
    val = rows[:n_val]
    train = rows[n_val:]
    for row in train:
        row["split"] = "train"
        row["extra_info"]["split"] = "train"
    for row in val:
        row["split"] = "val"
        row["extra_info"]["split"] = "val"
    return train, val


def to_parquet_safe(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    counts = Counter(row["source_name"] for row in rows)
    facts = [int(row["n_atomic_facts"]) for row in rows]
    answers = Counter(row["answer_idx"] for row in rows)
    return {
        "n": len(rows),
        "sources": dict(sorted(counts.items())),
        "answer_idx_counts": dict(sorted(answers.items())),
        "n_atomic_facts_min": min(facts),
        "n_atomic_facts_max": max(facts),
        "n_atomic_facts_avg": sum(facts) / len(facts),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Atomic-Fact RL Dataset Report",
        "",
        "This dataset follows the ATPO/MEDIQ setting: each case has an incomplete initial context, a multiple-choice medical question, answer options, and hidden atomic facts.",
        "",
        "## Splits",
        "",
        "| split | n | sources | avg facts |",
        "|---|---:|---|---:|",
    ]
    for split in ["train", "val", "test", "test_medqa", "test_medmcqa", "test_medicalexam", "test_medqa_diag", "test_mediq"]:
        info = manifest["splits"].get(split)
        if not info:
            continue
        lines.append(
            f"| {split} | {info['n']} | {json.dumps(info.get('sources', {}), ensure_ascii=False)} | {info.get('n_atomic_facts_avg', 0):.2f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `train` and `val` are split from local `mediQ/medqa_dev_convo.jsonl` because it already contains atomic facts.",
        "- `test_medqa` uses the ATPO-style `MedQA/medqa_test_dataset.jsonl` file.",
        "- `test_medmcqa` uses `MedMCQA/mcqa_test_dataset.jsonl`.",
        "- `test_medicalexam` uses `MedicalExam/aie_test_dataset.jsonl`.",
        "- `test_medqa_diag` is kept as an additional diagnostic subset, not one of the three main ATPO test sets.",
        "- `test_mediq` keeps the raw MEDIQ test-convo format converted with the same converter; it may overlap with ATPO-style MedQA test and is not included in the default combined `test` split.",
        "",
        "## Schema",
        "",
        "- `prompt`: chat messages visible to the actor.",
        "- `reward_model.ground_truth.answer`: correct option letter.",
        "- `extra_info.atomic_facts`: hidden case facts for the user/case simulator.",
        "- `extra_info.atomic_fact_items`: facts with stable fact ids.",
        "- `agent_name`: `atomic_fact_interaction`, compatible with the AtomicFact agent loop.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path(__file__).resolve().parents[4] / "diagprm_dataset")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--atpo_train_path",
        type=Path,
        default=None,
        help="Optional ATPO merged_train_dataset.jsonl. If present, it is used as the default train source.",
    )
    parser.add_argument(
        "--train_source",
        choices=["auto", "atpo_merged", "mediq_dev"],
        default="auto",
        help="Training source. auto uses ATPO merged train when available, otherwise mediQ dev.",
    )
    parser.add_argument(
        "--val_source",
        choices=["train_split", "atpo_test"],
        default="train_split",
        help="train_split holds out val_ratio from train; atpo_test uses the original ATPO-style test split as val.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir or dataset_root / "atomic_fact_rl_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stats: dict[str, dict[str, int]] = {}

    atpo_train_path = args.atpo_train_path or dataset_root.parent / "dataset" / "merged_train_dataset.jsonl"
    use_atpo_train = args.train_source == "atpo_merged" or (
        args.train_source == "auto" and atpo_train_path.exists()
    )
    if use_atpo_train:
        atpo_train_rows, stats = convert_atpo_merged_train(atpo_train_path, "train")
        source_stats["atpo_merged_train"] = dict(stats)
        all_train_rows = atpo_train_rows
        train_source_note = f"ATPO merged train: {atpo_train_path}"
    else:
        mediq_dev_path = dataset_root / MEDIQ_SOURCES["mediq_dev"]
        mediq_dev_rows, stats = convert_mediq(mediq_dev_path, "mediq_dev", "train")
        source_stats["mediq_dev"] = dict(stats)
        all_train_rows = mediq_dev_rows
        train_source_note = f"mediQ dev: {mediq_dev_path}"

    test_by_name: dict[str, list[dict[str, Any]]] = {}
    for source_name, rel in ATPO_STYLE_SOURCES.items():
        rows, stats = convert_atpo_style(dataset_root / rel, source_name, "test")
        source_stats[source_name] = dict(stats)
        test_by_name[source_name] = rows

    mediq_test_rows, stats = convert_mediq(dataset_root / MEDIQ_SOURCES["mediq_test"], "mediq_test", "test")
    source_stats["mediq_test"] = dict(stats)
    test_by_name["mediq_test"] = mediq_test_rows

    # Main ATPO-style combined test: MedicalExam, MedQA, and MedMCQA.
    test_rows = (
        test_by_name.get("medqa_test", [])
        + test_by_name.get("medicalexam_test", [])
        + test_by_name.get("medmcqa_test", [])
    )

    if args.val_source == "atpo_test":
        train_rows = all_train_rows
        val_rows = test_rows
        val_source_note = "ATPO-style combined test split"
    else:
        train_rows, val_rows = split_train_val(all_train_rows, args.val_ratio, args.seed)
        val_source_note = f"held-out train split, val_ratio={args.val_ratio}"

    split_rows = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "test_medqa": test_by_name.get("medqa_test", []),
        "test_medmcqa": test_by_name.get("medmcqa_test", []),
        "test_medicalexam": test_by_name.get("medicalexam_test", []),
        "test_medqa_diag": test_by_name.get("medqa_diag_test", []),
        "test_mediq": test_by_name.get("mediq_test", []),
    }

    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"atomic_fact_{split}.jsonl", rows)
        to_parquet_safe(output_dir / f"atomic_fact_{split}.parquet", rows)

    # Compatibility aliases for scripts that expect train/val/test parquet names.
    to_parquet_safe(output_dir / "diagprm_train.parquet", train_rows)
    to_parquet_safe(output_dir / "diagprm_val.parquet", val_rows)
    to_parquet_safe(output_dir / "diagprm_test.parquet", test_rows)

    manifest = {
        "name": "atomic_fact_rl_v1",
        "description": "ATPO-style medical QA cases with hidden atomic facts for turn-level credit assignment.",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_source": train_source_note,
        "val_source": val_source_note,
        "source_stats": source_stats,
        "splits": {split: summarize(rows) for split, rows in split_rows.items()},
        "files": {
            split: {
                "jsonl": str(output_dir / f"atomic_fact_{split}.jsonl"),
                "parquet": str(output_dir / f"atomic_fact_{split}.parquet"),
            }
            for split in split_rows
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "README.md", manifest)

    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    print(f"[OK] Wrote atomic-fact RL dataset to {output_dir}")


if __name__ == "__main__":
    main()
