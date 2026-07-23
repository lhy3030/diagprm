#!/usr/bin/env python3
"""Rewrite AtomicFact RL dataset system prompts without changing examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.diagprm.prompts.atomic_fact import (
    ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT,
    ATOMIC_FACT_SFT_MAX_TURNS,
)


SYSTEM_PROMPT = ATOMIC_FACT_DOCTOR_SYSTEM_PROMPT.format(
    max_turns=ATOMIC_FACT_SFT_MAX_TURNS
)
ALIASES = {
    "atomic_fact_train": "diagprm_train.parquet",
    "atomic_fact_val": "diagprm_val.parquet",
    "atomic_fact_test": "diagprm_test.parquet",
}


def rewrite_prompt(prompt: Any, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"{source}: prompt must be a non-empty message list")
    messages = [dict(message) for message in prompt]
    if messages[0].get("role") != "system":
        raise ValueError(f"{source}: prompt must start with a system message")
    messages[0]["content"] = SYSTEM_PROMPT
    return messages


def rewrite_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    changed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            old_prompt = row.get("prompt")
            new_prompt = rewrite_prompt(old_prompt, source=f"{path}:{line_no}")
            changed += int(old_prompt[0].get("content") != SYSTEM_PROMPT)
            row["prompt"] = new_prompt
            rows.append(row)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)
    return rows, changed


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temp_path, index=False)
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)

    summary = {}
    jsonl_paths = sorted(args.dataset_dir.glob("atomic_fact_*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No atomic_fact_*.jsonl files in {args.dataset_dir}")

    for jsonl_path in jsonl_paths:
        rows, changed = rewrite_jsonl(jsonl_path)
        stem = jsonl_path.stem
        parquet_path = jsonl_path.with_suffix(".parquet")
        write_parquet(parquet_path, rows)
        alias_name = ALIASES.get(stem)
        if alias_name:
            write_parquet(args.dataset_dir / alias_name, rows)
        summary[stem] = {"rows": len(rows), "changed_system_prompts": changed}

    print(json.dumps({
        "dataset_dir": str(args.dataset_dir),
        "max_turns": ATOMIC_FACT_SFT_MAX_TURNS,
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "splits": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
