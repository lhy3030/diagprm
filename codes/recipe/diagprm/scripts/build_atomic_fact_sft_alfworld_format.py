#!/usr/bin/env python3
"""Export AtomicFact/ATPO final SFT rows in an ALFWorld-style JSON format.

Target shape:

[
  {
    "id": "atomic_fact_0",
    "game_file": "atomic_fact_sft/source_row_3",
    "conversations": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]

The ATPO source stores rendered Qwen chat-template strings. This script parses
complete `<|im_start|>...<|im_end|>` messages from `prompt` and appends the
terminal `response` as the final assistant message.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MSG_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n([\s\S]*?)<\|im_end\|>",
    re.IGNORECASE,
)
FINAL_RE = re.compile(r"</think>\s*Final Answer\s*:\s*[A-Z]\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--id_prefix", default="atomic_fact")
    parser.add_argument("--game_file_prefix", default="atomic_fact_sft")
    parser.add_argument(
        "--keep_question_rows",
        action="store_true",
        help="Do not filter to terminal Final Answer rows. Default keeps finals only.",
    )
    return parser.parse_args()


def strip_im_end(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s*<\|im_end\|>\s*$", "", text)
    return text.strip()


def parse_rendered_prompt(prompt: str) -> list[dict[str, str]]:
    messages = [
        {"role": match.group(1).lower(), "content": match.group(2).strip()}
        for match in MSG_RE.finditer(prompt)
    ]
    if not messages:
        return [{"role": "user", "content": prompt.strip()}]
    return messages


def convert_row(
    obj: dict[str, Any],
    item_idx: int,
    source_row_idx: int,
    id_prefix: str,
    game_file_prefix: str,
) -> dict[str, Any]:
    conversations = parse_rendered_prompt(str(obj["prompt"]))
    conversations.append({
        "role": "assistant",
        "content": strip_im_end(str(obj["response"])),
    })
    return {
        "id": f"{id_prefix}_{item_idx}",
        "game_file": f"{game_file_prefix}/source_row_{source_row_idx}",
        "conversations": conversations,
    }


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)

    items: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_non_final": 0,
    }
    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            stats["total"] += 1
            obj = json.loads(line)
            if "prompt" not in obj or "response" not in obj:
                raise ValueError(f"Row {line_idx} must contain prompt/response keys: {obj.keys()}")
            response = str(obj["response"])
            if not args.keep_question_rows and not FINAL_RE.search(response):
                stats["dropped_non_final"] += 1
                continue
            source_row_idx = int(obj.get("source_row_idx", line_idx))
            items.append(
                convert_row(
                    obj=obj,
                    item_idx=len(items),
                    source_row_idx=source_row_idx,
                    id_prefix=args.id_prefix,
                    game_file_prefix=args.game_file_prefix,
                )
            )
            stats["kept"] += 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "output_json": str(args.output_json),
        "format": "alfworld_sft_json_list",
        "stats": stats,
    }
    args.output_json.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
