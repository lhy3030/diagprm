#!/usr/bin/env python3
"""Build no-think SFT data for AtomicFact/DiagPRM.

Transforms the original SFT JSONL (which has <think>...</think> blocks in every
response) into a no-think version:

1. Strips all <think>...</think> blocks from *both* the response and any
   assistant turns inside the prompt.
2. Inserts Qwen3's no-think marker into the system prompt:
   append  "\n/no_think"  at the end of the system content.
3. Keeps ALL turn types (Question and Final Answer), unlike the
   final-only builder.

Output files (per split):
  - <name>.parquet   – for verl.trainer.fsdp_sft_trainer
  - <name>.jsonl     – for human inspection
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches a full <think>...</think> block (possibly multiline, greedy=False)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Qwen3 im_start/im_end markers
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

NO_THINK_SUFFIX = "\n/no_think"


def strip_think(text: str) -> str:
    """Remove all <think>...</think> blocks and collapse extra leading whitespace."""
    stripped = THINK_RE.sub("", text)
    # Collapse any leading whitespace/newlines that the removed block left behind
    stripped = re.sub(r"^\s+", "", stripped)
    return stripped


def inject_no_think_into_system(prompt: str) -> str:
    """Append /no_think to the system turn content in a rendered chat string.

    The Qwen3 chat template produces:
        <|im_start|>system\\n<content><|im_end|>\\n

    We want to change the system content to end with "\\n/no_think".
    """
    # Find system block: <|im_start|>system\n ... <|im_end|>
    pattern = re.compile(
        r"(<\|im_start\|>system\n)(.*?)(<\|im_end\|>)",
        re.DOTALL,
    )
    def replace_system(m: re.Match) -> str:
        content = m.group(2)
        # Only add if not already there
        if not content.rstrip().endswith("/no_think"):
            content = content.rstrip() + NO_THINK_SUFFIX
        return m.group(1) + content + m.group(3)

    return pattern.sub(replace_system, prompt, count=1)


def clean_prompt(prompt: str) -> str:
    """Remove think blocks from assistant turns inside the prompt, and add /no_think to system."""
    # 1. Remove think blocks from all assistant turns embedded in the prompt
    #    (these appear in multi-turn prompts as completed assistant messages)
    cleaned = THINK_RE.sub("", prompt)

    # Fix up leading whitespace inside assistant turns after stripping think:
    # Pattern: <|im_start|>assistant\n<think>...</think>\nQuestion: ...
    # After removal: <|im_start|>assistant\n\nQuestion: ...
    # Clean to: <|im_start|>assistant\nQuestion: ...
    cleaned = re.sub(r"(<\|im_start\|>assistant\n)\n+", r"\1", cleaned)

    # 2. Inject /no_think into system prompt
    cleaned = inject_no_think_into_system(cleaned)

    return cleaned


def clean_response(response: str) -> str:
    """Remove think block from the response (only one think block expected at start)."""
    return strip_think(response)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "total": 0,
        "kept_question": 0,
        "kept_final_answer": 0,
        "dropped_invalid": 0,
    }
    with path.open("r", encoding="utf-8") as f:
        for source_idx, line in enumerate(f):
            if not line.strip():
                continue
            stats["total"] += 1
            obj = json.loads(line)
            if "prompt" not in obj or "response" not in obj:
                raise ValueError(f"Row {source_idx}: missing prompt/response keys")

            raw_prompt = str(obj["prompt"])
            raw_response = str(obj["response"])

            new_prompt = clean_prompt(raw_prompt)
            new_response = clean_response(raw_response)

            if not new_response.strip():
                stats["dropped_invalid"] += 1
                continue

            if "Final Answer" in new_response:
                stats["kept_final_answer"] += 1
            elif "Question:" in new_response:
                stats["kept_question"] += 1
            else:
                stats["dropped_invalid"] += 1
                continue

            row: dict[str, Any] = dict(obj)
            row["prompt"] = new_prompt
            row["response"] = new_response
            row["source_row_idx"] = source_idx
            rows.append(row)

    return rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, type=Path,
                        help="Original SFT JSONL with think blocks")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--train_name", default="atomic_fact_sft_nothink_train.parquet")
    parser.add_argument("--val_name", default="atomic_fact_sft_nothink_val.parquet")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Disable seeded shuffle before split")
    parser.add_argument("--final_only", action="store_true",
                        help="If set, keep only Final Answer rows (like original builder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)

    rows, stats = read_rows(args.input_jsonl)

    if args.final_only:
        rows = [r for r in rows if "Final Answer" in r["response"]]
        stats["after_final_only_filter"] = len(rows)

    if not rows:
        raise ValueError("No rows remain after filtering.")

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)

    n_val = max(1, int(len(rows) * args.val_ratio)) if args.val_ratio > 0 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    write_jsonl(train_path.with_suffix(".jsonl"), train_rows)
    write_jsonl(val_path.with_suffix(".jsonl"), val_rows)

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "train_parquet": str(train_path),
        "val_parquet": str(val_path),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "final_only": args.final_only,
        "stats": stats,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "policy": (
            "Stripped <think>...</think> from all prompts and responses; "
            "injected /no_think into system turn; kept all turn types."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
