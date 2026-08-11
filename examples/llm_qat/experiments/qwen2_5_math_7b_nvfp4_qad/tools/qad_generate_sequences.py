#!/usr/bin/env python3
"""Generate BF16 teacher sequences for a small post-RL QAD experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30100/generate")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def generate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    input_ids: list[int],
    max_new_tokens: int,
) -> str:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
        },
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            async with semaphore:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
            text = result["text"]
            if isinstance(text, list):
                text = text[0]
            return text
        except Exception as error:  # retry transient server/HTTP failures
            last_error = error
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"generation failed after retries: {last_error}")


async def main_async(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    random.Random(args.seed).shuffle(records)
    records = records[args.start_index : args.start_index + args.num_samples]

    inputs: list[list[int]] = []
    for record in records:
        encoded = tokenizer.apply_chat_template(
            record["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs.append(list(encoded["input_ids"]))

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                generate_one(session, semaphore, args.url, ids, args.max_new_tokens)
            )
            for ids in inputs
        ]
        responses: list[str] = []
        for index, task in enumerate(tasks, start=1):
            responses.append(await task)
            if index % 16 == 0 or index == len(tasks):
                print(f"generated {index}/{len(tasks)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for record, response in zip(records, responses, strict=True):
            messages = list(record["prompt"])
            messages.append({"role": "assistant", "content": response})
            json.dump(
                {"messages": messages, "label": record.get("label")},
                stream,
                ensure_ascii=False,
            )
            stream.write("\n")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
