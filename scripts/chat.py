#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal chat client for a model trained with LoRA (e.g., Qwen3) using the
artifacts produced by the attached training script (trainer.save_model + tokenizer.save_pretrained).

Features
- Works with either a *merged full model* directory or a *LoRA adapter* directory
- Optional base model path when loading a LoRA adapter
- Streaming token-by-token output in the terminal
- Multi-turn chat using the tokenizer's chat template (apply_chat_template)
- Simple slash commands: /exit, /reset, /save <path>

Usage examples
--------------
# 1) If your output_dir contains a LoRA adapter (adapter_config.json is present):
CUDA_VISIBLE_DEVICES=0 python scripts/chat.py \
  --base_model models/Qwen3-4B-Instruct-2507 \
  --adapter_or_model outputs/qwen3-4b-dpo \
  --system "あなたは役に立つAIアシスタントです。" --temperature 0.7

# 2) If you've already merged the adapter and have a full model dir:
CUDA_VISIBLE_DEVICES=0 python scripts/chat.py \
  --adapter_or_model models/Qwen3-4B-Instruct-2507 \
  --system "あなたは役に立つAIアシスタントです。" --temperature 0.2

Tips
- Use --load_4bit or --load_8bit on smaller GPUs.
- To reset context during a session, type /reset
- To quit, type /exit
- To save the conversation, type /save transcript.jsonl

Requirements
- transformers >= 4.42
- accelerate >= 0.30
- peft >= 0.10
- torch (CUDA recommended)
- bitsandbytes (optional for 4/8-bit loading)

"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # type: ignore


def detect_is_adapter_dir(path: Path) -> bool:
    """Return True if path looks like a LoRA adapter directory."""
    return (path / "adapter_config.json").exists() or (path / "adapter_model.bin").exists() or (path / "adapter_model.safetensors").exists()


def build_model_and_tokenizer(
    adapter_or_model: str,
    base_model: str | None,
    load_8bit: bool,
    load_4bit: bool,
    dtype: str | None,
    device_map: str,
):
    target = Path(adapter_or_model)
    is_adapter = detect_is_adapter_dir(target)

    # Choose dtype
    torch_dtype = None
    if dtype:
        d = dtype.lower()
        if d == "float16" or d == "fp16":
            torch_dtype = torch.float16
        elif d == "bfloat16" or d == "bf16":
            torch_dtype = torch.bfloat16
        elif d == "float32" or d == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    quantization_config = None
    model_kwargs = {
        "device_map": device_map,
    }

    # 4/8-bit loading via bitsandbytes if requested
    if load_4bit or load_8bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("bitsandbytes is required for 4/8-bit loading. Please `pip install bitsandbytes`." ) from e
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
        # dtype is set implicitly by quantization config
    else:
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

    if is_adapter:
        if base_model is None:
            raise ValueError("`--base_model` is required when `--adapter_or_model` points to a LoRA adapter directory.")
        # Load base model, then attach adapter
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, **model_kwargs)
        if PeftModel is None:
            raise RuntimeError("peft is not installed. Please `pip install peft`. ")
        model = PeftModel.from_pretrained(model, adapter_or_model)
        # Prefer adapter's tokenizer if it exists (e.g., special tokens set)
        try:
            tokenizer_adapter = AutoTokenizer.from_pretrained(adapter_or_model, use_fast=True, trust_remote_code=True)
            # If the adapter dir actually contains tokenizer files, use them
            if (Path(adapter_or_model) / "tokenizer.json").exists() or (Path(adapter_or_model) / "tokenizer.model").exists():
                tokenizer = tokenizer_adapter
        except Exception:
            pass
    else:
        # Full model directory
        tokenizer = AutoTokenizer.from_pretrained(adapter_or_model, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(adapter_or_model, trust_remote_code=True, **model_kwargs)

    model.eval()
    return model, tokenizer


def build_messages(system_prompt: str | None, history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # include prior turns
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def tokens_stream_infer(model, tokenizer, messages: List[Dict[str, str]],
                        max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                        repetition_penalty: float, do_sample: bool, eos_token_id: int | None):
    # Prepare the chat-formatted prompt using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
        streamer=streamer,
    )

    # Run generation in a separate thread so we can stream tokens
    import threading

    def _generate():
        with torch.no_grad():
            model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})

    t = threading.Thread(target=_generate)
    t.start()
    return streamer, t


def main():
    parser = argparse.ArgumentParser(description="Terminal chat client for LoRA-trained models (Qwen3 et al.)")
    parser.add_argument("--adapter_or_model", type=str, required=True,
                        help="Path to LoRA adapter directory (contains adapter_config.json) OR a full model directory.")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path (required only when --adapter_or_model points to a LoRA adapter).")

    parser.add_argument("--system", type=str, default="あなたは役に立つAIアシスタントです。",
                        help="System prompt for the assistant.")
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device map for loading the model. 'auto' is recommended.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
                        help="Preferred dtype when not using 4/8-bit quantization.")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit (bitsandbytes).")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit (bitsandbytes).")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_sample", action="store_true", help="Disable sampling (greedy/beam-like).")

    args = parser.parse_args()

    # Load
    model, tokenizer = build_model_and_tokenizer(
        adapter_or_model=args.adapter_or_model,
        base_model=args.base_model,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    # EOS token handling (use tokenizer.eos_token_id when available)
    eos_id = None
    try:
        eos_id = tokenizer.eos_token_id
    except Exception:
        eos_id = None

    history: List[Dict[str, str]] = []

    print("\n💬 Terminal chat started. Type your message and press Enter.\n"
          "Commands: /exit, /reset, /save <path>\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break

        if not user:
            continue
        if user.startswith("/exit"):
            break
        if user.startswith("/reset"):
            history = []
            print("(context cleared)")
            continue
        if user.startswith("/save"):
            _, *rest = user.split(maxsplit=1)
            if rest:
                path = rest[0]
            else:
                path = "transcript.jsonl"
            try:
                transcript = []
                if args.system:
                    transcript.append({"role": "system", "content": args.system})
                transcript.extend(history)
                with open(path, "w", encoding="utf-8") as f:
                    for msg in transcript:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                print(f"(saved to {path})")
            except Exception as e:
                print(f"(save failed: {e})")
            continue

        # Build messages for this turn
        msgs = build_messages(args.system, history, user)
        print("Assistant:", end=" ", flush=True)
        streamer, thread = tokens_stream_infer(
            model,
            tokenizer,
            msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            eos_token_id=eos_id,
        )

        # Stream tokens as they arrive
        response_text = ""
        for token in streamer:
            sys.stdout.write(token)
            sys.stdout.flush()
            response_text += token
        thread.join()
        print("")

        # Push assistant reply into history
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": response_text})

    print("Bye!")


if __name__ == "__main__":
    main()