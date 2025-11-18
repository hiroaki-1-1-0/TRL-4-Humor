#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PYTHONPATH=/work/hiroaki/dev/TRL-4-Humor \
torchrun --standalone --nproc_per_node=5 scripts/train_dpo_qwen3_lora.py \
  --model_path models/Qwen3-4B-Instruct-2507 \
  --data_path texts/go/t2t_en_selected_9.jsonl \
  --output_dir outputs/qwen3-4b-instruct-dpo \
  --epochs 3 \
  --beta 0.1 \
  --lr 2e-5 \
  --batch_size 4 \
  --grad_accum 8 \
  --use_flash_attn
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from peft import LoraConfig


def build_pairs_from_jsonl(
    jsonl_path: str,
    min_max_star: int = 10,
    limit: int = None,
) -> List[Dict[str, str]]:
    """
    JSONL（各行: {question, text, star, ...}）から
    DPO の (prompt, chosen, rejected) ペアを構築。
      - question ごとに star 最大を chosen, 最小を rejected
      - question 単位で max(star) < min_max_star は除外
      - 同点が複数ある場合は text 長い方（chosen）/ 短い方（rejected）を優先（決定論）
    """
    groups = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # データ仕様に合わせてキーを参照（手元のファイルは question/text/star）
            q = rec.get("question") or rec.get("questions")
            t = rec.get("text")
            s = rec.get("star") if "star" in rec else rec.get("stars")
            if q is None or t is None or s is None:
                # 必須キーがない行はスキップ
                continue
            groups[q].append({"text": t, "star": int(s)})

    pairs = []
    for q, recs in groups.items():
        # フィルタ: 最大starが閾値未満は除外
        max_star = max(r["star"] for r in recs)
        if max_star < min_max_star:
            continue

        # chosen: star最大、同点なら text が長い/辞書順で決定
        recs_max_sorted = sorted(
            recs, key=lambda r: (r["star"], len(r["text"]), r["text"]), reverse=True
        )
        chosen = recs_max_sorted[0]["text"].strip()

        # rejected: star最小、同点なら text が短い/辞書順で決定
        recs_min_sorted = sorted(
            recs, key=lambda r: (r["star"], -len(r["text"]), r["text"])
        )
        rejected = recs_min_sorted[0]["text"].strip()

        # 念のため、chosen/rejected が同一なら別候補を探す
        if chosen == rejected:
            for cand in recs_min_sorted[1:]:
                if cand["text"].strip() != chosen:
                    rejected = cand["text"].strip()
                    break
            else:
                # どうしても差が出ないならスキップ
                continue

        pairs.append({"question": q, "chosen": chosen, "rejected": rejected})

    # 安定化のため deterministic にソート（任意）
    pairs.sort(key=lambda x: (x["question"], len(x["chosen"]), len(x["rejected"])))

    if limit is not None:
        pairs = pairs[:limit]

    return pairs


def make_prompt_from_question(tokenizer, question: str) -> str:
    """
    Qwen Instruct のチャットテンプレートで prompt を成形。
    DPOTrainer には "prompt"（ユーザ発話のみ）と "chosen/rejected"（アシスタント出力）を渡す。
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,  # assistant 開始位置までを含める
            tokenize=False,
        )
    except Exception:
        # もしテンプレートが未定義の場合のフォールバック
        return f"<|user|>\n{question}\n<|assistant|>\n"


def prepare_datasets(tokenizer, jsonl_path: str, val_ratio: float = 0.1):
    raw_pairs = build_pairs_from_jsonl(jsonl_path, min_max_star=10)

    # prompt を生成（chat template により system/user/assistant タグ化）
    prompts, chosens, rejecteds = [], [], []
    for item in raw_pairs:
        prompts.append(make_prompt_from_question(tokenizer, item["question"]))
        chosens.append(item["chosen"])
        rejecteds.append(item["rejected"])

    ds = Dataset.from_dict(
        {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}
    )

    # 固定シードで分割（小規模データのため 90/10）
    ds = ds.shuffle(seed=42)
    n_total = len(ds)
    n_val = max(1, int(n_total * val_ratio))
    ds_val = ds.select(range(n_val))
    ds_train = ds.select(range(n_val, n_total))

    stats = {
        "total_pairs": n_total,
        "train_pairs": len(ds_train),
        "val_pairs": len(ds_val),
    }
    return DatasetDict(train=ds_train, validation=ds_val), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-4B-Instruct-2507",
        help="ローカルに保存済みの Qwen3-4B-Instruct-2507 へのパス",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="texts/go/t2t_en_selected_9.jsonl",
        help="学習データ（JSONL, question/text/star）へのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3-4b-dpo",
        help="学習結果の出力先",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1, help="DPOのβ（温度）")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4, help="GPU毎のミニバッチ")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--use_flash_attn", action="store_true")
    args = parser.parse_args()

    # Tokenizer / Model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"

    attn_impl = "flash_attention_2" if args.use_flash_attn else "eager"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Dataset 準備
    dsets, stats = prepare_datasets(tokenizer, args.data_path, val_ratio=0.1)
    print(f"[Data] total={stats['total_pairs']}, train={stats['train_pairs']}, val={stats['val_pairs']}")

    # LoRA 設定（Qwen系: attention + MLP）
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    )

    # Trainer（DPO）
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        weight_decay=0.1,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,  # DDP安定化
        remove_unused_columns=False,       # DPO では必須
        beta=args.beta,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_target_len,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Noneだと内部で参照モデルを自動作成（LoRAならメモリ影響は小さめ）
        args=training_args,
        train_dataset=dsets["train"],
        eval_dataset=dsets["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        print("[Done] Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
