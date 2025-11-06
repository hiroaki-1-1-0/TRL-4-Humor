#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pairwise preference evaluation for base vs. fine-tuned models.

Task: For each question in a JSONL dataset (with fields question/text/star),
compare two candidate completions: the max-star text (chosen) vs. the min-star
text (rejected). A model "passes" a pair if it assigns higher likelihood to the
chosen completion than to the rejected completion when both are conditioned on
an identical, chat-templated prompt of the question.

This mirrors the construction used for DPO training in the provided script.

Usage examples:

# Minimal (evaluate one base and one fine-tuned model)
python scripts/eval_star_preference.py \
  --data_path texts/go/t2t_en_selected_1.jsonl \
  --base_model_path models/Qwen3-4B-Instruct-2507 \
  --ft_model_paths outputs/qwen3-4b-dpo

# Evaluate multiple fine-tuned checkpoints at once
python scripts/eval_star_preference.py \
  --data_path texts/go/t2t_en_selected_1.jsonl \
  --base_model_path models/Qwen3-4B-Instruct-2507 \
  --ft_model_paths outputs/qwen3-4b-dpo outputs/qwen3-4b-dpo/checkpoint-2000

Outputs a table with accuracy metrics.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------
# Data prep (mirrors training)
# --------------------------

def build_pairs_from_jsonl(jsonl_path: str, min_max_star: int = 10, limit: int = None) -> List[Dict[str, str]]:
    """
    From a JSONL of records {question, text, star, ...}, build DPO-like pairs:
      - For each question, pick chosen = max-star text; rejected = min-star text.
      - Skip questions whose max star < min_max_star.
      - Break ties deterministically by text length / lexicographic order.
    """
    groups = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("question") or rec.get("questions")
            t = rec.get("text")
            s = rec.get("star") if "star" in rec else rec.get("stars")
            if q is None or t is None or s is None:
                continue
            groups[q].append({"text": t, "star": int(s)})

    pairs = []
    for q, recs in groups.items():
        max_star = max(r["star"] for r in recs)
        if max_star < min_max_star:
            continue

        recs_max_sorted = sorted(
            recs, key=lambda r: (r["star"], len(r["text"]), r["text"]), reverse=True
        )
        chosen = recs_max_sorted[0]["text"].strip()

        recs_min_sorted = sorted(
            recs, key=lambda r: (r["star"], -len(r["text"]), r["text"]) 
        )
        rejected = recs_min_sorted[0]["text"].strip()

        if chosen == rejected:
            for cand in recs_min_sorted[1:]:
                if cand["text"].strip() != chosen:
                    rejected = cand["text"].strip()
                    break
            else:
                continue

        pairs.append({"question": q, "chosen": chosen, "rejected": rejected})

    pairs.sort(key=lambda x: (x["question"], len(x["chosen"]), len(x["rejected"])) )
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def make_prompt_from_question(tokenizer, question: str) -> str:
    """
    Render the user message with the model's chat template, ending at the
    assistant generation start. Works for Qwen Instruct; falls back to a simple
    tag-based prompt if the template is unavailable.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return f"<|user|>\n{question}\n<|assistant|>\n"


# --------------------------
# Scoring utilities
# --------------------------

def _completion_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
) -> Tuple[float, float, int]:
    """Return (sum_logprob, mean_logprob, length) over completion tokens.

    We compute NLL only for the completion span. The prompt tokens are masked
    with -100 so they don't contribute to the loss.
    """
    with torch.no_grad():
        # Tokenize prompt and completion separately to locate the boundary
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        comp_ids = tokenizer(completion, add_special_tokens=False).input_ids
        # Build joint sequence
        input_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
        # Labels: mask prompt, supervise completion
        labels = torch.full_like(input_ids, -100)
        labels[0, len(prompt_ids):] = input_ids[0, len(prompt_ids):]

        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, T, V]
        logprobs = torch.log_softmax(logits, dim=-1)
        # Shift for causal LM
        shifted_logprobs = logprobs[:, :-1, :]
        shifted_labels = labels[:, 1:]
        shifted_input_ids = input_ids[:, 1:]

        # Gather token logprobs only where labels are not -100
        mask = (shifted_labels != -100)
        target_token_ids = shifted_input_ids[mask]
        token_logprobs = shifted_logprobs[mask, target_token_ids]

        if token_logprobs.numel() == 0:
            return float("nan"), float("nan"), 0

        sum_lp = token_logprobs.sum().item()
        mean_lp = (token_logprobs.mean().item())
        return sum_lp, mean_lp, token_logprobs.numel()


def evaluate_model(
    model_path: str,
    data_pairs: List[Dict[str, str]],
    use_flash_attn: bool = False,
    max_pairs: int = None,
) -> Dict[str, float]:
    """Compute pairwise preference accuracy and margins for a model path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    # Build prompts once
    prompts = [make_prompt_from_question(tokenizer, p["question"]) for p in data_pairs]
    if max_pairs is not None:
        prompts = prompts[:max_pairs]
        data_pairs = data_pairs[:max_pairs]

    n = len(data_pairs)
    acc_sum = acc_mean = 0
    margins_sum = []  # sum-logprob margin (chosen - rejected)
    margins_mean = []  # mean-logprob margin

    for i, (prompt, pair) in enumerate(zip(prompts, data_pairs), 1):
        ch = pair["chosen"]
        rj = pair["rejected"]
        sum_ch, mean_ch, len_ch = _completion_logprob(model, tokenizer, prompt, ch, device)
        sum_rj, mean_rj, len_rj = _completion_logprob(model, tokenizer, prompt, rj, device)

        # If either completion failed to score, skip the pair
        if any(map(lambda x: (x != x) or (x is None), [sum_ch, sum_rj])):
            n -= 1
            continue

        acc_sum += 1 if (sum_ch > sum_rj) else 0
        acc_mean += 1 if (mean_ch > mean_rj) else 0
        margins_sum.append(sum_ch - sum_rj)
        margins_mean.append(mean_ch - mean_rj)

        if i % 50 == 0:
            print(f"Scored {i}/{len(prompts)} pairs...")

    # Aggregate
    import numpy as np
    def _safe_mean(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    return {
        "pairs": n,
        "acc_sum": acc_sum / n if n else float("nan"),
        "acc_mean": acc_mean / n if n else float("nan"),
        "avg_margin_sum": _safe_mean(margins_sum),
        "avg_margin_mean": _safe_mean(margins_mean),
    }


# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSONL with fields question/text/star (e.g., texts/go/t2t_en_selected_1.jsonl)")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model directory (e.g., models/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--ft_model_paths", type=str, nargs="*", default=[],
                        help="One or more fine-tuned model paths under outputs/")
    parser.add_argument("--min_max_star", type=int, default=10,
                        help="Skip questions whose maximum star < this threshold (must match training)")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of pairs for a quick run")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use flash_attention_2 if available")

    args = parser.parse_args()

    # Build evaluation pairs (same logic as training)
    pairs = build_pairs_from_jsonl(args.data_path, min_max_star=args.min_max_star, limit=args.limit)
    if not pairs:
        print("No evaluation pairs were constructed. Check data_path and min_max_star.")
        return
    print(f"[Eval] total_pairs={len(pairs)} (min_max_star={args.min_max_star})")

    # Evaluate base
    print("\n=== Base model ===")
    base_stats = evaluate_model(args.base_model_path, pairs, use_flash_attn=args.use_flash_attn, max_pairs=args.limit)
    for k, v in base_stats.items():
        print(f"{k}: {v}")

    # Evaluate fine-tuned models (and keep stats to avoid reloading twice)
    ft_stats_list = []
    for mpath in args.ft_model_paths:
        print(f"\n=== Fine-tuned: {mpath} ===")
        stats = evaluate_model(mpath, pairs, use_flash_attn=args.use_flash_attn, max_pairs=args.limit)
        ft_stats_list.append((mpath, stats))
        for k, v in stats.items():
            print(f"{k}: {v}")

    # Pretty summary table
    try:
        from tabulate import tabulate
        rows = [["Base", base_stats["pairs"], base_stats["acc_sum"], base_stats["acc_mean"], base_stats["avg_margin_sum"], base_stats["avg_margin_mean"]]]
        for mpath, stats in ft_stats_list:
            rows.append([os.path.basename(mpath.rstrip('/')), stats["pairs"], stats["acc_sum"], stats["acc_mean"], stats["avg_margin_sum"], stats["avg_margin_mean"]])
        print("\n" + tabulate(rows, headers=["model", "pairs", "acc(sum)", "acc(mean)", "avg_margin(sum)", "avg_margin(mean)"], floatfmt=".4f"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
