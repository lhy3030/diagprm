#!/usr/bin/env python3
"""Merge and lightly filter DiagPRM SFT JSONL runs.

The script keeps three artifacts:
  - merged raw rows: all parseable rows, with source metadata
  - deduplicated rows: exact duplicate trajectories removed
  - filtered rows: rows that pass conservative SFT quality filters

It also writes a report and a small inspectable sample file.
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

try:
    import pandas as pd
except Exception:  # pragma: no cover - parquet is optional
    pd = None


GENERIC_ANSWER_RE = re.compile(r"\b(that has been present|that has been happening)\b", re.I)
MIXED_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
THIRD_PERSON_RE = re.compile(r"\bthe patient\b", re.I)
DISEASE_STATE_RE = re.compile(
    r"\b(the disease|my disease|condition has spread|spread to other parts|"
    r"metastas|advanced stage|later stages?|stage [ivx0-9]+)\b",
    re.I,
)
HIGH_RISK_PATIENT_TEXT_RE = re.compile(
    r"\b("
    r"serum|plasma|mutation|chromosome|gene|protein|biopsy|histolog|"
    r"pseudorosettes|neurosecretory|hydrops|radiosensitivity|hypoplasia|"
    r"agenesis|dysplasia|malformation|lymphoedema|lymphedema|interferon|"
    r"cytokines|ectatic capillaries|aortopulmonary|pulmonary venous|"
    r"magnesium in (?:my )?urine|calcium in (?:my )?urine"
    r")\b",
    re.I,
)
QUESTION_ANSWER_MISMATCH_RE = re.compile(r"^(what|when|where|why|how|at what)\b", re.I)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def discover_jsonl(run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        if run_dir.is_file():
            paths.append(run_dir)
            continue
        paths.extend(sorted(run_dir.glob("worker_*/20*/diagprm_sft_train.jsonl")))
        paths.extend(sorted(run_dir.glob("*/diagprm_sft_train.jsonl")))
        if run_dir.name == "diagprm_sft_train.jsonl":
            paths.append(run_dir)
    return sorted(set(paths))


def read_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(path.open("r", encoding="utf-8"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"file": str(path), "line": line_no, "error": str(exc)})
                continue
            row["_source_file"] = str(path)
            row["_source_line"] = line_no
            rows.append(row)
    return rows, errors


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = stable_json_hash(
            {
                "disease_id": row.get("disease_id", ""),
                "messages": row.get("messages", []),
                "candidate_signature": row.get("candidate_signature", []),
            }
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        row["_merge_hash"] = key
        deduped.append(row)
    return deduped, dropped


def patient_messages(row: dict[str, Any]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in row.get("messages", [])
        if message.get("role") == "user"
    ]


def assistant_messages(row: dict[str, Any]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in row.get("messages", [])
        if message.get("role") == "assistant"
    ]


def count_generic_answers(row: dict[str, Any]) -> int:
    return sum(1 for text in patient_messages(row) if GENERIC_ANSWER_RE.search(text))


def has_question_answer_mismatch(row: dict[str, Any]) -> bool:
    messages = row.get("messages", [])
    for idx, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        try:
            action = json.loads(message.get("content", "{}"))
        except json.JSONDecodeError:
            continue
        question = str(action.get("question", "")).strip()
        if not question or not QUESTION_ANSWER_MISMATCH_RE.search(question):
            continue
        next_msg = messages[idx + 1]
        if next_msg.get("role") == "user" and normalize_text(next_msg.get("content", "")).startswith("yes"):
            return True
    return False


def filter_reasons(row: dict[str, Any], max_generic_answers: int) -> list[str]:
    reasons: list[str] = []
    metrics = row.get("candidate_metrics", {}) if isinstance(row.get("candidate_metrics"), dict) else {}
    if metrics.get("final_correct") is not True:
        reasons.append("final_not_correct")
    if float(metrics.get("kg_coverage", 0.0)) < 0.5:
        reasons.append("low_kg_coverage")
    if int(metrics.get("num_new_facts", 0)) < 2:
        reasons.append("too_few_new_facts")
    if int(metrics.get("invalid_questions", 0)) > 0:
        reasons.append("invalid_questions")
    if int(metrics.get("repeated_questions", 0)) > 0:
        reasons.append("repeated_questions")
    if metrics.get("quality_reasons"):
        reasons.append("generator_quality_reasons")

    p_text = "\n".join(patient_messages(row))
    a_text = "\n".join(assistant_messages(row))
    disease = normalize_text(row.get("disease", ""))
    non_final_text = "\n".join(
        str(message.get("content", ""))
        for message in row.get("messages", [])[:-1]
        if message.get("role") != "system"
    )
    if disease and disease in normalize_text(non_final_text):
        reasons.append("disease_leak_before_final")
    if THIRD_PERSON_RE.search(p_text):
        reasons.append("third_person_patient")
    if MIXED_CHINESE_RE.search(p_text) or MIXED_CHINESE_RE.search(a_text):
        reasons.append("mixed_chinese")
    if DISEASE_STATE_RE.search(p_text):
        reasons.append("disease_state_or_stage_text")
    if HIGH_RISK_PATIENT_TEXT_RE.search(p_text):
        reasons.append("high_risk_patient_text")
    if count_generic_answers(row) > max_generic_answers:
        reasons.append("too_many_generic_answers")
    if has_question_answer_mismatch(row):
        reasons.append("question_answer_mismatch")
    return sorted(set(reasons))


def cap_per_disease(rows: list[dict[str, Any]], max_per_disease: int) -> tuple[list[dict[str, Any]], int]:
    if max_per_disease <= 0:
        return rows, 0
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    dropped = 0
    for row in sorted(
        rows,
        key=lambda r: (
            str(r.get("disease_id", "")),
            int(r.get("candidate_rank", 999)),
            -float(r.get("candidate_metrics", {}).get("score", 0.0)),
        ),
    ):
        did = str(row.get("disease_id", ""))
        if counts[did] >= max_per_disease:
            dropped += 1
            continue
        counts[did] += 1
        kept.append(row)
    return kept, dropped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sample(path: Path, rows: list[dict[str, Any]], n: int, seed: int) -> None:
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(sample, start=1):
            f.write(f"===== SAMPLE {idx} | {row.get('disease')} | {row.get('disease_id')} =====\n")
            f.write(f"source: {row.get('_source_file')}:{row.get('_source_line')}\n")
            f.write(f"metrics: {json.dumps(row.get('candidate_metrics', {}), ensure_ascii=False)}\n")
            for message in row.get("messages", []):
                if message.get("role") == "system":
                    continue
                f.write(f"{message.get('role')}: {message.get('content')}\n")
            f.write("\n")


def build_report(
    paths: list[Path],
    parse_errors: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    dedup_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    filter_reason_counts: Counter,
    duplicate_dropped: int,
    cap_dropped: int,
) -> dict[str, Any]:
    def metric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [
            row.get("candidate_metrics", {}).get(key)
            for row in rows
            if isinstance(row.get("candidate_metrics", {}).get(key), (int, float))
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "input_files": [str(path) for path in paths],
        "parse_errors": parse_errors[:20],
        "parse_error_count": len(parse_errors),
        "raw_rows": len(raw_rows),
        "dedup_rows": len(dedup_rows),
        "duplicate_dropped": duplicate_dropped,
        "filtered_rows": len(filtered_rows),
        "cap_dropped": cap_dropped,
        "unique_disease_ids_raw": len({row.get("disease_id") for row in raw_rows}),
        "unique_disease_ids_filtered": len({row.get("disease_id") for row in filtered_rows}),
        "filter_reason_counts": dict(filter_reason_counts.most_common()),
        "raw_metrics": {
            "avg_kg_coverage": metric_mean(raw_rows, "kg_coverage"),
            "avg_new_facts": metric_mean(raw_rows, "num_new_facts"),
            "avg_turns": metric_mean(raw_rows, "num_turns"),
            "avg_score": metric_mean(raw_rows, "score"),
        },
        "filtered_metrics": {
            "avg_kg_coverage": metric_mean(filtered_rows, "kg_coverage"),
            "avg_new_facts": metric_mean(filtered_rows, "num_new_facts"),
            "avg_turns": metric_mean(filtered_rows, "num_turns"),
            "avg_score": metric_mean(filtered_rows, "score"),
        },
        "generic_answer_rows_raw": sum(1 for row in raw_rows if count_generic_answers(row) > 0),
        "generic_answer_rows_filtered": sum(1 for row in filtered_rows if count_generic_answers(row) > 0),
        "noise_rows_raw": sum(1 for row in raw_rows if any(t.get("is_noise") for t in row.get("sft_trace", []))),
        "noise_rows_filtered": sum(1 for row in filtered_rows if any(t.get("is_noise") for t in row.get("sft_trace", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_generic_answers", type=int, default=1)
    parser.add_argument("--max_per_disease", type=int, default=2)
    parser.add_argument("--sample_size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = discover_jsonl(args.run_dir)
    if not paths:
        raise SystemExit("No diagprm_sft_train.jsonl files found.")

    raw_rows, parse_errors = read_rows(paths)
    dedup_rows, duplicate_dropped = dedupe_rows(raw_rows)

    filtered: list[dict[str, Any]] = []
    reason_counts: Counter = Counter()
    rejected: list[dict[str, Any]] = []
    for row in dedup_rows:
        reasons = filter_reasons(row, args.max_generic_answers)
        if reasons:
            reason_counts.update(reasons)
            row["_filter_reasons"] = reasons
            rejected.append(row)
            continue
        filtered.append(row)

    filtered, cap_dropped = cap_per_disease(filtered, args.max_per_disease)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "diagprm_sft_merged_raw.jsonl"
    dedup_path = args.output_dir / "diagprm_sft_merged_dedup.jsonl"
    filtered_path = args.output_dir / "diagprm_sft_merged_filtered.jsonl"
    rejected_path = args.output_dir / "diagprm_sft_rejected.jsonl"
    sample_path = args.output_dir / "diagprm_sft_filtered_samples.txt"
    report_path = args.output_dir / "merge_report.json"

    write_jsonl(raw_path, raw_rows)
    write_jsonl(dedup_path, dedup_rows)
    write_jsonl(filtered_path, filtered)
    write_jsonl(rejected_path, rejected)
    write_sample(sample_path, filtered, args.sample_size, args.seed)

    if pd is not None:
        pd.DataFrame(filtered).to_parquet(args.output_dir / "diagprm_sft_merged_filtered.parquet", index=False)

    report = build_report(
        paths=paths,
        parse_errors=parse_errors,
        raw_rows=raw_rows,
        dedup_rows=dedup_rows,
        filtered_rows=filtered,
        filter_reason_counts=reason_counts,
        duplicate_dropped=duplicate_dropped,
        cap_dropped=cap_dropped,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[OK] Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
