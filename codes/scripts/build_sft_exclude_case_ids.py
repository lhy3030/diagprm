#!/usr/bin/env python3
"""Build a disease_id exclude list from an interrupted sharded SFT run.

The generator shuffles cases, shards by idx % stride, then processes cases in
order. Given each worker's progress file, this script reconstructs which cases
were already attempted, including skipped cases that do not appear in JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy_seed_per_worker",
        action="store_true",
        help="Use seed+worker_id to reconstruct older runs launched before shard seed was fixed.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timestamp", default="", help="Optional worker subdir timestamp, e.g. 20260627_145636.")
    args = parser.parse_args()

    cases = load_jsonl(args.dataset_jsonl)
    attempted: set[str] = set()
    summary: dict[str, Any] = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "run_dir": str(args.run_dir),
        "seed": args.seed,
        "legacy_seed_per_worker": args.legacy_seed_per_worker,
        "num_workers": args.num_workers,
        "workers": {},
    }

    for worker_id in range(args.num_workers):
        worker_seed = args.seed + worker_id if args.legacy_seed_per_worker else args.seed
        shuffled = list(cases)
        random.Random(worker_seed).shuffle(shuffled)
        worker_dir = args.run_dir / f"worker_{worker_id}"
        if args.timestamp:
            worker_dirs = [worker_dir / args.timestamp]
        else:
            worker_dirs = sorted(p for p in worker_dir.iterdir() if p.is_dir()) if worker_dir.exists() else []
        if not worker_dirs:
            summary["workers"][str(worker_id)] = {"processed_cases": 0, "status": "missing_worker_dir"}
            continue
        current_dir = worker_dirs[-1]
        progress_path = current_dir / "diagprm_sft_train_progress.json"
        if not progress_path.exists():
            summary["workers"][str(worker_id)] = {
                "processed_cases": 0,
                "status": "missing_progress",
                "dir": str(current_dir),
            }
            continue

        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        processed_cases = int(progress.get("processed_cases", 0))
        shard = [case for idx, case in enumerate(shuffled) if idx % args.num_workers == worker_id]
        shard_attempted = shard[:processed_cases]
        ids = [str(case.get("disease_id", "")) for case in shard_attempted if case.get("disease_id")]
        attempted.update(ids)
        summary["workers"][str(worker_id)] = {
            "dir": str(current_dir),
            "processed_cases": processed_cases,
            "seed": worker_seed,
            "shard_size": len(shard),
            "attempted_ids": len(ids),
            "progress": progress,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(attempted)) + "\n", encoding="utf-8")
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary["attempted_unique"] = len(attempted)
    summary["output"] = str(args.output)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
