#!/usr/bin/env python3
"""Build final-turn-only ATPO SFT data.

The released ATPO SFT JSONL stores one next-action sample per dialogue prefix:
intermediate rows teach `Question: ...`, while the final row contains the full
dialogue history in `prompt` and teaches the terminal `Final Answer: ...`.

For our AtomicFact SFT warm start, keep only the terminal rows so each original
trajectory contributes one SFT sample ending with an answer.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


FINAL_RE = re.compile(r"</think>\s*Final Answer\s*:\s*([A-Z])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--train_name", default="atpo_sft_final_only_train.parquet")
    parser.add_argument("--val_name", default="atpo_sft_final_only_val.parquet")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Keep source order before splitting. Default is seeded shuffle.",
    )
    return parser.parse_args()


def read_final_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "kept_final_answer": 0,
        "dropped_question": 0,
        "dropped_invalid": 0,
    }
    with path.open("r", encoding="utf-8") as f:
        for source_row_idx, line in enumerate(f):
            if not line.strip():
                continue
            stats["total"] += 1
            obj = json.loads(line)
            if "prompt" not in obj or "response" not in obj:
                raise ValueError(
                    f"Row {source_row_idx} must contain prompt/response keys: {obj.keys()}"
                )

            response = str(obj["response"])
            if FINAL_RE.search(response):
                row = dict(obj)
                row["prompt"] = str(row["prompt"])
                row["response"] = response
                row["source_row_idx"] = source_row_idx
                rows.append(row)
                stats["kept_final_answer"] += 1
            elif "Question:" in response:
                stats["dropped_question"] += 1
            else:
                stats["dropped_invalid"] += 1
    return rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val_ratio must be in [0, 1)")

    rows, stats = read_final_rows(args.input_jsonl)
    if not rows:
        raise ValueError(f"No final-answer rows found in {args.input_jsonl}")

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)

    n_val = max(1, int(len(rows) * args.val_ratio)) if args.val_ratio > 0 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name
    train_jsonl = train_path.with_suffix(".jsonl")
    val_jsonl = val_path.with_suffix(".jsonl")

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "train_parquet": str(train_path),
        "val_parquet": str(val_path),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "stats": stats,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "policy": "kept only rows whose response matches </think> Final Answer: [A-Z]",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
