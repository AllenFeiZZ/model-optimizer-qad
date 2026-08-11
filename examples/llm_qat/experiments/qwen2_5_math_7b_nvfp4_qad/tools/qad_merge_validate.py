#!/usr/bin/env python3
"""Merge BF16 QAD generation shards and enforce prompt-disjointness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--expected", type=int, required=True)
    return parser.parse_args()


def read_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError(f"{path}:{line_number}: invalid messages")
                assistant = messages[-1]
                if assistant.get("role") != "assistant" or not assistant.get("content", "").strip():
                    raise ValueError(f"{path}:{line_number}: empty/missing assistant response")
                records.append(record)
    return records


def prompt_hash(record: dict) -> str:
    prompt = record["messages"][:-1]
    canonical = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(round(fraction * (len(ordered) - 1)), len(ordered) - 1)]


def main() -> None:
    args = parse_args()
    records = read_records(args.parts)
    holdout = read_records([args.holdout])
    if len(records) != args.expected:
        raise ValueError(f"expected {args.expected} records, found {len(records)}")

    train_hashes = [prompt_hash(record) for record in records]
    holdout_hashes = {prompt_hash(record) for record in holdout}
    duplicate_count = len(train_hashes) - len(set(train_hashes))
    overlap_count = len(set(train_hashes) & holdout_hashes)
    if duplicate_count or overlap_count:
        raise ValueError(
            f"data isolation failed: duplicate_prompts={duplicate_count}, "
            f"holdout_overlap={overlap_count}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")
    os.replace(temporary, output)

    lengths = [len(record["messages"][-1]["content"]) for record in records]
    print(
        json.dumps(
            {
                "records": len(records),
                "unique_prompts": len(set(train_hashes)),
                "holdout_records": len(holdout),
                "holdout_overlap": overlap_count,
                "empty_responses": 0,
                "response_chars_mean": round(mean(lengths), 1),
                "response_chars_p50": percentile(lengths, 0.50),
                "response_chars_p95": percentile(lengths, 0.95),
                "response_chars_max": max(lengths),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
