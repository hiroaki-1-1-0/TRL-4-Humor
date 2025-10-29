#!/usr/bin/env python3
"""
DPO training script for the reordered Oogiri dataset using Qwen/Qwen3-VL-8B-Instruct and TRL.

The script builds preference pairs by grouping rows that share the same `ID` and treating
`reordered_answer_type == "Human_top_tier"` as the preferred answer while
`reordered_answer_type == "Human_lower_tier"` acts as the rejected answer. Only rows whose
`dataset` column is `"OG"` participate in training.

When `prompt_type == "image"`, the script expects the corresponding image file to live in
`<image_root>/{image}.jpg` and injects it into the chat template using the Qwen processor.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import Dataset
from PIL import Image
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


LOGGER = logging.getLogger("train_dpo_qwen3vl")


TOP_TIER_LABEL = "Human_top_tier"
LOWER_TIER_LABEL = "Human_lower_tier"
TARGET_DATASET_NAME = "OG"


@dataclass
class PreferenceExample:
    """Container for a single DPO preference pair."""

    id_value: str
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
    prompt_type: str
    image_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id_value,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "prompt_type": self.prompt_type,
            "image_id": self.image_id,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO model on the reordered Oogiri dataset with Qwen3-VL.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("merged_Oogiri_Dataset_reordered.csv"),
        help="Input CSV produced by merging and reordering the Oogiri dataset.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("images/OG_image"),
        help="Directory that stores the image files referenced in the CSV.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Hugging Face model identifier to finetune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dpo-qwen3vl"),
        help="Where to store checkpoints and logs.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Proportion of IDs reserved for validation (0 disables validation).",
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed used for shuffling IDs before the split.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="DPOTrainer `per_device_train_batch_size`.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="DPOTrainer `per_device_eval_batch_size`.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps, useful for large models on limited hardware.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Optimizer learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of epochs for DPO training.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay factor.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Linear warmup ratio.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval (in optimizer steps).")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation interval when validation is enabled.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter.")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=2048,
        help="Maximum number of tokens retained from the prompt after tokenization.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="Maximum number of tokens retained from the completion after tokenization.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization via BitsAndBytes. Helps with VRAM usage but requires GPU support.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run training in bfloat16. Enable when hardware supports bf16 for best stability with Qwen3-VL.",
    )
    return parser.parse_args()


def collect_preference_examples(csv_path: Path) -> list[PreferenceExample]:
    df = pd.read_csv(csv_path)

    filtered = df[
        (df["dataset"].astype(str) == TARGET_DATASET_NAME)
        & df["reordered_answer_type"].isin([TOP_TIER_LABEL, LOWER_TIER_LABEL])
    ]

    examples: list[PreferenceExample] = []
    grouped = filtered.groupby("ID", sort=False)
    for id_value, group in grouped:
        positives = group[group["reordered_answer_type"] == TOP_TIER_LABEL]
        negatives = group[group["reordered_answer_type"] == LOWER_TIER_LABEL]
        if positives.empty or negatives.empty:
            continue

        prompt_types = group["prompt_type"].dropna().unique()
        prompt_type = prompt_types[0] if len(prompt_types) > 0 else "text"

        prompt_candidates = [
            value.strip()
            for value in group["prompt"].dropna().astype(str)
            if isinstance(value, str) and value.strip() and value.strip().lower() != "nan"
        ]
        if len(set(prompt_candidates)) > 1:
            LOGGER.warning("Multiple prompts detected for ID %s, using the first one.", id_value)

        prompt_text = prompt_candidates[0] if prompt_candidates else None
        if prompt_type == "image" and not prompt_text:
            prompt_text = "画像で一言"
        if prompt_type != "image" and not prompt_text:
            LOGGER.warning("Skipping ID %s because text prompt is missing.", id_value)
            continue

        image_value = None
        if prompt_type == "image":
            image_candidates = group["image"].dropna().unique()
            if len(image_candidates) == 0:
                LOGGER.warning("ID %s has image prompt type but no image value.", id_value)
            else:
                image_value = str(image_candidates[0])

        prompt_messages = [{"role": "user", "content": prompt_text}]

        for _, pos_row in positives.iterrows():
            chosen_msg = [{"role": "assistant", "content": str(pos_row["answer"])}]
            for _, neg_row in negatives.iterrows():
                rejected_msg = [{"role": "assistant", "content": str(neg_row["answer"])}]
                examples.append(
                    PreferenceExample(
                        id_value=str(id_value),
                        prompt=prompt_messages,
                        chosen=chosen_msg,
                        rejected=rejected_msg,
                        prompt_type=prompt_type,
                        image_id=image_value,
                    )
                )

    LOGGER.info("Collected %d preference pairs from %d unique IDs.", len(examples), grouped.ngroups)
    if not examples:
        raise ValueError("No valid preference examples were found. Verify the CSV contents.")

    return examples


def split_examples(
    examples: Iterable[PreferenceExample], val_split: float, seed: int
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    per_id: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        per_id.setdefault(example.id_value, []).append(example)

    ids = list(per_id.keys())
    random.Random(seed).shuffle(ids)

    if val_split <= 0.0 or len(ids) < 2:
        train_records = [item for id_value in ids for item in per_id[id_value]]
        return train_records, []

    val_count = max(1, int(math.ceil(len(ids) * val_split)))
    val_ids = set(ids[:val_count])
    train_ids = [id_value for id_value in ids if id_value not in val_ids]

    train_records = [item for id_value in train_ids for item in per_id[id_value]]
    val_records = [item for id_value in val_ids for item in per_id[id_value]]

    LOGGER.info(
        "Split: %d train pairs across %d IDs, %d validation pairs across %d IDs.",
        len(train_records),
        len(train_ids),
        len(val_records),
        len(val_ids),
    )
    return train_records, val_records


def build_dataset(
    examples: list[PreferenceExample],
    image_root: Path,
) -> Dataset:
    dataset = Dataset.from_list([example.to_dict() for example in examples])

    def _attach_images(batch: dict) -> dict:
        prompt_types = batch["prompt_type"]
        image_ids = batch["image_id"]
        images: list[list[Image.Image]] = []
        for prompt_type, image_id in zip(prompt_types, image_ids):
            if prompt_type == "image" and image_id:
                image_path = resolve_image_path(image_root, image_id)
                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Expected image for prompt type 'image' at {image_path}, but the file does not exist."
                    )
                image = Image.open(image_path).convert("RGB")
                images.append([image])
            else:
                images.append([])
        batch["images"] = images
        return batch

    dataset = dataset.map(_attach_images, batched=True, desc="Loading image assets")

    return dataset


def resolve_image_path(image_root: Path, image_id: str) -> Path:
    candidate = Path(image_id)
    if candidate.suffix:
        return image_root / candidate.name
    return image_root / f"{image_id}.jpg"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = collect_preference_examples(args.csv_path)
    train_examples, val_examples = split_examples(examples, args.val_split, args.seed)
    train_dataset = build_dataset(train_examples, args.image_root)
    eval_dataset = build_dataset(val_examples, args.image_root) if val_examples else None

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    model_kwargs = {}
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    LOGGER.info("Loading policy model %s", args.model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        **model_kwargs,
    )

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        bf16=args.bf16,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        run_name="qwen3vl-dpo",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir / "final")
    processor.save_pretrained(args.output_dir / "final")


if __name__ == "__main__":
    main()
