#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: GRPO fine-tuning for Qwen/Qwen3-4B-Thinking-2507 with TRL (LoRA, conversational prompts).

This trains a *Thinking* model (not Instruct) using GRPO with a rule-based reward:
we present a pair of answers and ask the model "Which answer is better?", then score
the model's decision against the known ground truth (chosen vs. rejected).

Key differences from your DPO script:
- Trainer: GRPOTrainer (online RL) instead of DPOTrainer.
- Model: Qwen/Qwen3-4B-Thinking-2507 (Thinking model; handles <think> ... </think>).
- Reward: rule-based judge that parses the assistant's final decision (1 or 2).
- Prompt format: conversational (tokenizer.apply_chat_template) with instructions to emit <answer>1</answer> or <answer>2</answer>.

Launch (3× RTX 6000 Ada, FlashAttention2, LoRA, no vLLM):

CUDA_VISIBLE_DEVICES=0,1,4 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=3 scripts/train_grpo_qwen3_thinking.py \
  --model_path models/Qwen3-4B-Thinking-2507 \
  --data_path texts/go/t2t_en_selected_9.jsonl \
  --output_dir outputs/qwen3-4b-thinking-grpo \
  --epochs 1 \
  --lr 5e-6 \
  --batch_size 1 \
  --grad_accum 8 \
  --num_generations 4 \
  --max_prompt_len 1024 \
  --max_completion_len 64 \
  --loss_type dapo \
  --scale_rewards batch \
  --use_flash_attn \
  --use_lora

(Optional) If you have vLLM installed, enabling guided decoding can drastically reduce useless <think> tokens:
  --use_vllm --vllm_mode colocate --vllm_guided_regex "^(<think>.*?</think>)?<answer>[12]</answer>$"

Notes:
- Qwen3 Thinking chat template often inserts <think> automatically. We strip <think>...</think> and any stray </think> before parsing.
- We strongly instruct the model to end with <answer>1</answer> or <answer>2</answer>. The reward function checks this first,
  and otherwise falls back to the last digit (1 or 2) found in the output.
"""

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer

try:
    from peft import LoraConfig
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def _supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 8 or (major == 7 and minor >= 5)


# -----------------------------
# Data: build (question, chosen, rejected) pairs from JSONL
# -----------------------------

def build_pairs_from_jsonl(jsonl_path: str, min_max_star: int = 10, limit: int = None) -> List[Dict[str, str]]:
    """
    Build DPO-style pairs (question, chosen, rejected) from a JSONL file where each line contains:
      {"question": str, "text": str, "star": int, ...}

    Heuristic (matching your DPO/eval scripts):
      - Group by question
      - chosen = text with max star (ties -> longer first, then lexicographically)
      - rejected = text with min star (ties -> shorter first, then lexicographically)
      - Skip questions where max(star) < min_max_star
    """
    groups = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            q = (obj.get("question") or "").strip()
            t = (obj.get("text") or "").strip()
            s = obj.get("star", None)
            if not q or not t or s is None:
                continue
            groups[q].append({"text": t, "star": int(s)})

    pairs = []
    for q, recs in groups.items():
        if not recs:
            continue
        max_star = max(r["star"] for r in recs)
        if max_star < min_max_star:
            continue

        # chosen: highest star; break ties by longer text then text
        recs_max_sorted = sorted(recs, key=lambda r: (r["star"], len(r["text"]), r["text"]), reverse=True)
        chosen = recs_max_sorted[0]["text"].strip()

        # rejected: lowest star; break ties by shorter text then text
        recs_min_sorted = sorted(recs, key=lambda r: (r["star"], -len(r["text"]), r["text"]))
        rejected = recs_min_sorted[0]["text"].strip()

        # if chosen == rejected, try next candidate
        if chosen == rejected:
            for cand in recs_min_sorted[1:]:
                if cand["text"].strip() != chosen:
                    rejected = cand["text"].strip()
                    break
            else:
                continue  # skip if we cannot get different texts

        pairs.append({"question": q, "chosen": chosen, "rejected": rejected})
        if limit and len(pairs) >= limit:
            break
    return pairs


# -----------------------------
# Prompting for Qwen3 Thinking
# -----------------------------

INSTRUCTION_EN = (
    "Which answer is better?\n"
    "Answer 1: \"{a1}\"\n"
    "Answer 2: \"{a2}\"\n\n"
    "Think step by step inside <think>...</think>, but in the very last line output "
    "your final decision as <answer>1</answer> or <answer>2</answer>.\n"
    "Do not write anything after </answer>."
)

CHAT_TEMPLATE_KWARGS = {
    "add_generation_prompt": True,  # ensure tokenizer adds assistant cue when formatting chat prompts
}

def build_user_content(question: str, answer_1: str, answer_2: str) -> str:
    body = f"Question: {question}\n\n" + INSTRUCTION_EN.format(a1=answer_1, a2=answer_2)
    return body


def derive_items_for_training(pairs: List[Dict[str, str]], seed: int) -> List[Dict]:
    """
    Randomly flip (chosen, rejected) so the model can't shortcut on position.
    Return items with `messages` (conversational format) and `ground_truth` (1 or 2).
    """
    rng = random.Random(seed)
    items = []
    for p in pairs:
        a = p["chosen"]
        b = p["rejected"]
        if rng.random() < 0.5:
            answer_1, answer_2, gt = a, b, 1
        else:
            answer_1, answer_2, gt = b, a, 2

        user_content = build_user_content(p["question"], answer_1, answer_2)
        messages = [{"role": "user", "content": user_content}]
        items.append({
            "prompt": messages,
            "ground_truth": gt,
            "question": p["question"],
            "answer_1": answer_1,
            "answer_2": answer_2,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS.copy(),
        })
    return items


# -----------------------------
# Reward function (rule-based)
# -----------------------------

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([12])\s*</answer>", re.IGNORECASE | re.DOTALL)
_TRAIL_12_RE = re.compile(r"([12])\s*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_END_THINK_RE = re.compile(r"</think>", re.IGNORECASE)

def _as_text_from_completion(completion) -> str:
    """
    GRPO can pass either strings (standard format) or list-of-messages (conversational).
    We normalize to a single string for parsing.
    """
    if isinstance(completion, list):
        # expect [{"role":"assistant","content":"..."}]
        if not completion:
            return ""
        x = completion[0]
        if isinstance(x, dict):
            return str(x.get("content", ""))
        return str(x)
    elif isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _strip_think(text: str) -> str:
    # remove <think>...</think> and any stray </think>
    text = _THINK_BLOCK_RE.sub("", text)
    text = _END_THINK_RE.sub("", text)
    return text


def _parse_choice(text: str) -> int:
    """
    Parse final decision from assistant output.
    Priority: <answer>1/2</answer> -> trailing 1/2 -> last 1/2 anywhere.
    Return 1, 2, or 0 if cannot parse.
    """
    if not text:
        return 0
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return int(m.group(1))
    m = _TRAIL_12_RE.search(text.strip())
    if m:
        return int(m.group(1))
    for ch in reversed(text):
        if ch in ("1", "2"):
            return int(ch)
    return 0


def pairwise_reward_func(completions, ground_truth, **kwargs):
    """
    Return a float reward per completion.
    - Correct decision -> 1.0
    - Wrong/invalid -> 0.0
    """
    # Normalize completions to raw strings and strip <think>
    contents = [_strip_think(_as_text_from_completion(c)) for c in completions]
    choices = [_parse_choice(c) for c in contents]

    # Expand ground_truth across num_generations
    if isinstance(ground_truth, (list, tuple)):
        n_prompts = len(ground_truth)
    else:
        ground_truth = [int(ground_truth)]
        n_prompts = 1

    n_completions = len(choices)
    if n_prompts == 0:
        return [0.0] * n_completions

    repeat = max(1, n_completions // n_prompts)
    gt_expanded = []
    for g in ground_truth:
        gt_expanded.extend([int(g)] * repeat)
    # Edge case: if due to rounding the list is short/long, adjust:
    if len(gt_expanded) < n_completions:
        gt_expanded.extend([int(ground_truth[-1])] * (n_completions - len(gt_expanded)))
    elif len(gt_expanded) > n_completions:
        gt_expanded = gt_expanded[:n_completions]

    rewards = [1.0 if (c in (1, 2) and c == gt) else 0.0 for c, gt in zip(choices, gt_expanded)]
    return rewards


# -----------------------------
# Main: GRPO training
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3-4b-thinking-grpo")
    parser.add_argument("--seed", type=int, default=42)

    # Training schedule
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)  # per-device
    parser.add_argument("--grad_accum", type=int, default=8)

    # Generation / GRPO
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_completion_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="dapo", choices=["grpo", "dapo", "dr_grpo"])
    parser.add_argument("--scale_rewards", type=str, default="batch", choices=["group", "batch", "none", "False", "false"])

    # Infra
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="server", choices=["server", "colocate"])
    parser.add_argument("--vllm_guided_regex", type=str, default=None)
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.3)
    parser.add_argument("--report_to", type=str, default=None)  # e.g., "wandb"
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing (disabled by default to avoid DDP+LoRA issues)")

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Data
    parser.add_argument("--min_max_star", type=int, default=10)
    parser.add_argument("--limit_pairs", type=int, default=None)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)

    has_cuda = torch.cuda.is_available()
    has_bf16 = _supports_bf16()
    torch_dtype = torch.float32
    if has_cuda:
        torch_dtype = torch.bfloat16 if has_bf16 else torch.float16

    attn_impl = "flash_attention_2" if (args.use_flash_attn and has_cuda) else "eager"

    # Build pairs
    pairs = build_pairs_from_jsonl(args.data_path, min_max_star=args.min_max_star, limit=args.limit_pairs)
    items = derive_items_for_training(pairs, seed=args.seed)

    # Conversational dataset (messages + ground_truth)
    train_ds = Dataset.from_list(items)

    # Tokenizer (we let GRPOTrainer handle model loading)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # LoRA config (optional)
    peft_config = None
    if args.use_lora:
        if not _HAS_PEFT:
            raise RuntimeError("peft is not installed but --use_lora was set.")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ],
        )

    # Scale rewards flag normalization
    if args.scale_rewards in ("False", "false", "none"):
        scale_rewards = False
    elif args.scale_rewards in ("batch", "group"):
        scale_rewards = args.scale_rewards
    else:
        scale_rewards = "batch"

    # Trainer config
    gradient_checkpointing_kwargs = {"use_reentrant": False} if args.gradient_checkpointing else None

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=os.path.basename(args.output_dir.rstrip("/")),
        report_to=(None if args.report_to in (None, "", "none") else args.report_to),

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=has_bf16,
        fp16=(has_cuda and not has_bf16),
        tf32=has_cuda,
        remove_unused_columns=False,  # keep extra columns for reward function
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,

        # GRPO specifics
        max_prompt_length=args.max_prompt_len,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=(None if args.top_k <= 0 else args.top_k),
        repetition_penalty=args.repetition_penalty,
        loss_type=args.loss_type,
        scale_rewards=scale_rewards,

        # vLLM (optional)
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_guided_decoding_regex=args.vllm_guided_regex,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,

        # from_pretrained kwargs
        model_init_kwargs={
            "attn_implementation": attn_impl,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            # For Thinking models, keep bf16 weights if possible
            "dtype": torch_dtype,
        },
    )

    trainer = GRPOTrainer(
        model=args.model_path,                       # local path ok
        args=training_args,
        reward_funcs=pairwise_reward_func,
        train_dataset=train_ds,
        processing_class=tok,                       # tokenizer for chat template
        peft_config=peft_config,
    )

    if args.gradient_checkpointing and training_args.gradient_checkpointing:
        enable_gckpt_kwargs = gradient_checkpointing_kwargs or {}
        model = trainer.model
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=enable_gckpt_kwargs)
            except TypeError:
                model.gradient_checkpointing_enable()

    # kick off training
    trainer.train()

    # save
    trainer.save_model(args.output_dir)
    if trainer.accelerator.is_main_process:
        tok.save_pretrained(args.output_dir)
        print("[Done] Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
