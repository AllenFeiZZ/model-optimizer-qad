#!/usr/bin/env python3
"""Compare pre/post-QAD students against a BF16 teacher on fixed sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

import modelopt.torch.opt as mto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def encode_record(tokenizer, record: dict, max_length: int) -> tuple[list[int], int]:
    messages = record["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True
    )["input_ids"]
    full = list(full[:max_length])
    response_start = min(len(prompt), len(full) - 1)
    return full, response_start


def forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_start: int,
    chunk_size: int = 64,
) -> tuple[float, int, int]:
    # logits at position i predict token i+1; response begins at response_start.
    start = max(0, response_start - 1)
    stop = teacher_logits.shape[1] - 1
    kl_sum = 0.0
    top1_same = 0
    token_count = 0
    for left in range(start, stop, chunk_size):
        right = min(stop, left + chunk_size)
        teacher_chunk = teacher_logits[0, left:right].to(student_logits.device).float()
        student_chunk = student_logits[0, left:right].float()
        teacher_logp = torch.log_softmax(teacher_chunk, dim=-1)
        student_logp = torch.log_softmax(student_chunk, dim=-1)
        kl_sum += (
            teacher_logp.exp().mul(teacher_logp - student_logp).sum(dim=-1).sum().item()
        )
        top1_same += (
            teacher_chunk.argmax(dim=-1) == student_chunk.argmax(dim=-1)
        ).sum().item()
        token_count += right - left
    return kl_sum, top1_same, token_count


def main() -> None:
    args = parse_args()
    mto.enable_huggingface_checkpointing()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher)
    records = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    # Use a fixed tail slice so every student sees exactly the same contexts.
    records = records[-args.num_samples :]
    encoded = [encode_record(tokenizer, record, args.max_length) for record in records]

    teacher = transformers.AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda:0")
    teacher.eval()

    for student_path in args.student:
        student = transformers.AutoModelForCausalLM.from_pretrained(
            student_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to("cuda:1")
        student.eval()
        enabled_quantizers = sum(
            1
            for module in student.modules()
            if "TensorQuantizer" in type(module).__name__
            and getattr(module, "is_enabled", False)
        )

        total_kl = 0.0
        total_same = 0
        total_tokens = 0
        with torch.no_grad():
            for index, (input_ids, response_start) in enumerate(encoded, start=1):
                teacher_input = torch.tensor([input_ids], device="cuda:0")
                student_input = teacher_input.to("cuda:1")
                teacher_logits = teacher(input_ids=teacher_input, use_cache=False).logits
                student_logits = student(input_ids=student_input, use_cache=False).logits
                kl_sum, top1_same, token_count = forward_kl(
                    teacher_logits, student_logits, response_start
                )
                total_kl += kl_sum
                total_same += top1_same
                total_tokens += token_count
                del teacher_logits, student_logits, teacher_input, student_input
                print(f"{Path(student_path).name}: {index}/{len(encoded)}", flush=True)

        print(
            json.dumps(
                {
                    "student": student_path,
                    "enabled_quantizers": enabled_quantizers,
                    "tokens": total_tokens,
                    "mean_forward_kl": total_kl / total_tokens,
                    "top1_agreement": total_same / total_tokens,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del student
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
