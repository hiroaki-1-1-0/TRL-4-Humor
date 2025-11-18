#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate base and fine-tuned Qwen-Instruct models on a pairwise preference task.

- Builds (question, chosen, rejected) pairs from JSONL files like texts/go/*.jsonl
  (same heuristic as your training script: pick max star as chosen, min star as rejected,
   require max(star) >= min_max_star, break ties deterministically).
- For each pair, randomly flip the order and ask the model:

    Question: {question}

    Which answer is better?
    Answer 1: "{answer_1}"
    Answer 2: "{answer_2}"

    At the end of your output, please output the number of the Better Answer (1 or 2).
    Do not write anything after that final number.

- Parses the final number ("1" or "2") from the model output to score correctness.
- Saves a .jsonl log with: {"question","answer 1","answer 2","choice"} to eval/{model_name}/.
- Prints accuracy for each model.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_pairwise_qwen3.py \
    --base_model_path models/Qwen3-4B-Thinking-2507 \
    --ft_model_path outputs/qwen3-4b-thinking-grpo \
    --seed 42 \
    --use_flash_attn \
    --skip_base_eval

Requirements:
  transformers >= 4.41, peft, torch, (optional) triton/flash-attn for --use_flash_attn.
"""

import argparse
import os
import json
import re
import glob
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime, timezone, timedelta

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Optional PEFT for LoRA adapters
try:
    from peft import AutoPeftModelForCausalLM, PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def build_pairs_from_jsonl(jsonl_path: str, min_max_star: int = 10, limit: int = None) -> List[Dict[str, str]]:
    """
    Build DPO-style pairs (question, chosen, rejected) from a JSONL file where each line contains:
      {"question": str, "text": str, "star": int, ...}

    Heuristic (mirrors scripts/train_dpo_qwen3_lora.py):
      - Group by question
      - chosen = text with max star (if ties, pick the longer text, then lexicographically)
      - rejected = text with min star (if ties, pick the shorter text, then lexicographically)
      - Skip questions where max(star) < min_max_star
    """
    groups = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            q = rec.get("question")
            t = rec.get("text")
            s = rec.get("star") if "star" in rec else rec.get("stars")
            if q is None or t is None or s is None:
                continue
            try:
                s_int = int(s)
            except Exception:
                # if star is weird, try to coerce or skip
                try:
                    s_int = int(float(s))
                except Exception:
                    continue
            groups[q].append({"text": str(t).strip(), "star": s_int})

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

        # guard: if same text, try next candidate
        if chosen == rejected:
            for cand in recs_min_sorted[1:]:
                if cand["text"].strip() != chosen:
                    rejected = cand["text"].strip()
                    break
            else:
                # if still identical, skip
                continue

        pairs.append({"question": q, "chosen": chosen, "rejected": rejected})

    # deterministic order for stability
    pairs.sort(key=lambda x: (x["question"], len(x["chosen"]), len(x["rejected"])))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def derive_eval_items(pairs: List[Dict[str, str]], seed: int) -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for p in pairs:
        a = p["chosen"]
        b = p["rejected"]
        if rng.random() < 0.5:
            items.append({"question": p["question"], "answer_1": a, "answer_2": b, "ground_truth": 1})
        else:
            items.append({"question": p["question"], "answer_1": b, "answer_2": a, "ground_truth": 2})
    return items


def apply_qwen_chat_template(tokenizer, content: str) -> str:
    """
    Wrap user content in Qwen Instruct chat template.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        # Fallback if template is missing
        return f"<|user|>\n{content}\n<|assistant|>\n"


def build_prompt(question: str, answer_1: str, answer_2: str) -> str:
    # Keep instruction in English as requested
    body = (
        f"Question: {question}\n\n"
        f"Which answer is better?\n"
        f"Answer 1: \"{answer_1}\"\n"
        f"Answer 2: \"{answer_2}\"\n\n"
        f"At the end of your output, please output the number of the Better Answer (1 or 2)."
        f"Do not write anything after that final number."
    )
    return body


_CHOICE_RE = re.compile(r"([12])\s*$")
JST = timezone(timedelta(hours=9))


def extract_last_choice(text: str) -> int:
    """
    Extract the final '1' or '2' at the end of the model output.
    Falls back to 'last 1/2 anywhere' if no strict trailing digit is found.
    Returns 1, 2, or 0 if cannot parse.
    """
    if not text:
        return 0
    # Try strict: trailing 1/2
    m = _CHOICE_RE.search(text.strip())
    if m:
        return int(m.group(1))
    # Fallback: scan from end for last 1/2
    for ch in reversed(text):
        if ch in ("1", "2"):
            return int(ch)
    return 0


def load_model_and_tokenizer(model_path: str, device: str, use_flash_attn: bool):
    """
    Load a (possibly PEFT) model from a path.
    - If path contains full model weights: use AutoModelForCausalLM
    - Else if adapters: use AutoPeftModelForCausalLM (preferred) or PeftModel over base (handled by caller)
    """
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = None
    errors = []

    # Try plain full model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "auto" else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        return model, tok, False  # not an adapter-only checkpoint
    except Exception as e:
        errors.append(f"AutoModelForCausalLM failed: {e}")

    # Try PEFT auto loader (adapter dir)
    if _HAS_PEFT:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "auto" else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            # attn_implementation isn't configurable via AutoPeftModelForCausalLM; acceptable for eval.
            return model, tok, True
        except Exception as e:
            errors.append(f"AutoPeftModelForCausalLM failed: {e}")

    raise RuntimeError("Failed to load model from path: %s\n%s" % (model_path, "\n".join(errors)))


def maybe_load_adapter_on_base(base_model_path: str, adapter_path: str, device: str, use_flash_attn: bool):
    """
    Fallback: explicitly compose base + adapter when adapter alone couldn't be loaded.
    """
    if not _HAS_PEFT:
        raise RuntimeError("peft is required to load adapters but not installed.")

    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto" if device == "auto" else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tok


def run_eval_one_model(
    model_path: str,
    eval_items: List[Dict],
    save_dir: str,
    model_display_name: str = None,
    device: str = "auto",
    use_flash_attn: bool = False,
    max_new_tokens: int = 262144,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    verbose_every: int = 50,
) -> Dict[str, float]:
    """
    Evaluate a single model on prepared eval items.
    Writes JSONL with {"question","answer 1","answer 2","choice"} to eval/{model_name}/.
    Returns dict with metrics.
    """
    model_name = model_display_name or os.path.basename(os.path.normpath(model_path))
    out_dir = os.path.join(save_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(out_dir, f"eval_{ts}.jsonl")
    meta_json = os.path.join(out_dir, f"metrics_{ts}.json")

    # Load model
    # First try to load path directly (covers full checkpoints or adapter dirs with AutoPeft)
    try:
        model, tokenizer, is_adapter_only = load_model_and_tokenizer(model_path, device, use_flash_attn)
    except Exception as e1:
        # If adapter-only and failed, try base+adapter fallback: requires BASE env var
        base_from_env = os.environ.get("BASE_MODEL_PATH", "")
        if base_from_env and _HAS_PEFT:
            model, tokenizer = maybe_load_adapter_on_base(base_from_env, model_path, device, use_flash_attn)
            is_adapter_only = True
        else:
            raise

    model.eval()
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.bfloat16)

    n = 0
    n_valid = 0
    n_correct = 0

    use_sampling = temperature is not None and temperature > 0.0
    generation_sampling_kwargs = {"do_sample": use_sampling}
    if use_sampling:
        generation_sampling_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        # Force the greedy-safe defaults so transformers doesn't warn about ignored sampler flags.
        generation_sampling_kwargs.update(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )

    progress = tqdm(total=len(eval_items), desc=f"[{model_name}] evaluating", unit="item")

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(eval_items, start=1):
            q = item["question"]
            a1 = item["answer_1"]
            a2 = item["answer_2"]
            content = build_prompt(q, a1, a2)
            prompt = apply_qwen_chat_template(tokenizer, content)

            inputs = tokenizer(prompt, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_sampling_kwargs,
                )
            gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].size(-1):], skip_special_tokens=True)

            choice = extract_last_choice(gen_text)  # 1, 2, or 0
            ground_truth = item["ground_truth"]  # 1 or 2
            is_valid = int(choice in (1, 2))
            is_correct = int(choice == ground_truth)

            if is_valid:
                n_valid += 1
                n_correct += is_correct

            n += 1

            record = {
                "question": q,
                "answer 1": a1,
                "answer 2": a2,
                "choice": choice,  # model's chosen better answer (1/2); 0=invalid
                # Extra, non-required fields can be useful for debugging:
                "ground_truth": ground_truth,
                "valid": is_valid,
                "output": gen_text,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress.update(1)
            progress.set_postfix({
                "valid": n_valid,
                "correct": n_correct,
                "acc": (n_correct / n_valid) if n_valid else 0.0,
            })

            if verbose_every and (idx % verbose_every == 0):
                progress.write(f"[{model_name}] {idx}/{len(eval_items)} processed... valid={n_valid}, correct={n_correct}")

    progress.close()

    acc = (n_correct / n_valid) if n_valid > 0 else 0.0
    metrics = {
        "model": model_name,
        "total": n,
        "valid": n_valid,
        "correct": n_correct,
        "accuracy": acc,
        "timestamp": ts,
        "save_dir": out_dir,
        "jsonl": out_jsonl,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Result] {model_name}: accuracy={acc:.4f} (valid {n_valid}/{n}) -> {out_jsonl}")
    return metrics


def collect_pairs_from_glob(data_glob: str, min_max_star: int, max_samples: int = None, seed: int = 42) -> List[Dict[str, str]]:
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No JSONL found for glob: {data_glob}")
    all_pairs = []
    for fp in files:
        pairs = build_pairs_from_jsonl(fp, min_max_star=min_max_star, limit=None)
        all_pairs.extend(pairs)
    # deterministic but reproducible shuffle
    rnd = random.Random(seed)
    rnd.shuffle(all_pairs)
    if max_samples is not None and max_samples > 0:
        all_pairs = all_pairs[:max_samples]
    return all_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", type=str, required=True, help="Path to the base model (e.g., models/Qwen3-4B-Instruct-2507)")
    ap.add_argument("--ft_model_path", type=str, required=True, help="Path to the fine-tuned model or adapter dir (e.g., outputs/qwen3-4b-dpo)")
    ap.add_argument("--data_glob", type=str, default="texts/go/t2t_en_selected_1.jsonl", help="Glob to evaluation JSONL(s)")
    ap.add_argument("--min_max_star", type=int, default=10, help="Minimum max(star) per question to include")
    ap.add_argument("--max_samples", type=int, default=0, help="Limit number of pairs after merge+shuffle (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for answer ordering & sampling")
    ap.add_argument("--use_flash_attn", action="store_true", help="Use flash_attention_2 if available")
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", type=str, default="auto", help="Device map: 'auto' or leave empty")
    ap.add_argument("--skip_base_eval", action="store_true", help="Skip evaluating the base model and run only the fine-tuned model")
    args = ap.parse_args()

    # Prepare eval data from glob
    pairs = collect_pairs_from_glob(args.data_glob, min_max_star=args.min_max_star, max_samples=(args.max_samples or None), seed=args.seed)
    eval_items = derive_eval_items(pairs, seed=args.seed)

    # Ensure eval/ directory
    save_root = "eval"
    os.makedirs(save_root, exist_ok=True)

    base_metrics = None

    # Temporarily set BASE_MODEL_PATH for adapter loading fallback
    os.environ["BASE_MODEL_PATH"] = args.base_model_path

    if args.skip_base_eval:
        print("=== Skipping base model evaluation (requested) ===")
    else:
        # 1) Base model
        print("=== Evaluating base model ===")
        base_metrics = run_eval_one_model(
            model_path=args.base_model_path,
            eval_items=eval_items,
            save_dir=save_root,
            model_display_name=os.path.basename(os.path.normpath(args.base_model_path)),
            device=args.device,
            use_flash_attn=args.use_flash_attn,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            verbose_every=50,
        )

    # 2) Fine-tuned model or adapter
    print("\n=== Evaluating fine-tuned model ===")
    ft_metrics = run_eval_one_model(
        model_path=args.ft_model_path,
        eval_items=eval_items,
        save_dir=save_root,
        model_display_name=os.path.basename(os.path.normpath(args.ft_model_path)),
        device=args.device,
        use_flash_attn=args.use_flash_attn,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        verbose_every=50,
    )

    # Summary
    print("\n=== Summary ===")
    summary = {}
    if base_metrics is not None:
        summary["base"] = base_metrics
    summary["fine_tuned"] = ft_metrics
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
