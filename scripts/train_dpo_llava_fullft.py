#!/usr/bin/env python3
"""
Working DPO training script for the reordered Oogiri dataset using LLaVA (llava-hf/llava-v1.6-mistral-7b-hf).

This script mirrors the structure of ``train_dpo_qwen3vl_work.py`` but targets the LLaVA vision-language
model family, which does not require the Qwen-specific image grid handling. It follows the TRL vision DPO
example so that multimodal inputs are processed correctly.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import pandas as pd
import torch
from datasets import Dataset, features
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, PreTrainedTokenizerBase

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


LOGGER = logging.getLogger("train_dpo_llava_work")

TOP_TIER_LABEL = "Human_top_tier"
LOWER_TIER_LABEL = "Human_lower_tier"
TARGET_DATASET_NAME = "OG"
DEFAULT_PROMPT_FALLBACK = "画像で一言"


# Enable logging in a Hugging Face Space (matches the reference script behaviour)
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class DatasetArguments:
    """Arguments that control local dataset preparation."""

    csv_path: Path = field(default=Path("merged_Oogiri_Dataset_reordered.csv"))
    image_root: Path = field(default=Path("images/OG_image"))
    val_split: float = field(default=0.05)
    dataset_seed: int = field(default=42, metadata={"help": "Seed used when splitting IDs into train/validation."})
    local_files_only: bool = field(default=False)
    prompt_fallback: str = field(default=DEFAULT_PROMPT_FALLBACK)


@dataclass
class PreferenceExample:
    """Container for a single DPO preference pair."""

    id_value: str
    prompt_messages: list[dict[str, Any]]
    chosen_text: str
    rejected_text: str
    image_id: str

    def to_record(self, processor: AutoProcessor, image_root: Path) -> dict[str, Any]:
        """Convert the in-memory representation into a dataset row."""
        prompt_text = processor.apply_chat_template(
            self.prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_path = resolve_image_path(image_root, self.image_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for ID {self.id_value}: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        return {
            "id": self.id_value,
            "prompt": prompt_text,
            "chosen": self.chosen_text,
            "rejected": self.rejected_text,
            "images": [image],
        }


class LlavaVisionDPOTrainer(DPOTrainer):
    """DPO trainer that tokenizes multimodal prompts via the provided processor."""

    @staticmethod
    def process_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        processor = processing_class
        tokenizer = processor.tokenizer

        processed_features = processor(images=features.get("images"), text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values[0]

        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output: dict[str, Any] = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if pixel_values is not None:
            output["pixel_values"] = pixel_values

        for key in ("pixel_attention_mask", "image_sizes", "token_type_ids"):
            if key in processed_features and processed_features[key] is not None:
                output[key] = processed_features[key][0]

        return output


def collect_preference_examples(
    csv_path: Path,
    prompt_fallback: str,
) -> list[PreferenceExample]:
    """Load the CSV and convert rows into DPO preference examples."""
    df = pd.read_csv(csv_path)

    filtered = df[
        (df["dataset"].astype(str) == TARGET_DATASET_NAME)
        & df["reordered_answer_type"].isin([TOP_TIER_LABEL, LOWER_TIER_LABEL])
    ]

    image_mask = filtered["prompt_type"].astype(str).str.lower() == "image"
    non_image_count = len(filtered) - int(image_mask.sum())
    if non_image_count:
        LOGGER.info("Dropping %d rows whose prompt_type is not 'image'.", non_image_count)

    filtered = filtered[image_mask]

    examples: list[PreferenceExample] = []
    grouped = filtered.groupby("ID", sort=False)
    for id_value, group in grouped:
        positives = group[group["reordered_answer_type"] == TOP_TIER_LABEL]
        negatives = group[group["reordered_answer_type"] == LOWER_TIER_LABEL]
        if positives.empty or negatives.empty:
            continue

        prompt_candidates = [
            candidate.strip()
            for candidate in group["prompt"].dropna().astype(str)
            if candidate and candidate.strip() and candidate.strip().lower() != "nan"
        ]
        if len(set(prompt_candidates)) > 1:
            LOGGER.warning("Multiple prompts detected for ID %s, using the first one.", id_value)

        prompt_text = prompt_candidates[0] if prompt_candidates else prompt_fallback
        image_value = str(id_value).strip()
        if not image_value:
            LOGGER.warning("ID %s has image prompt type but a blank ID value.", id_value)
            continue

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        for _, pos_row in positives.iterrows():
            chosen_answer = normalize_text_field(pos_row["answer"], prompt_fallback)
            if not chosen_answer:
                continue
            for _, neg_row in negatives.iterrows():
                rejected_answer = normalize_text_field(neg_row["answer"], prompt_fallback)
                if not rejected_answer:
                    continue
                examples.append(
                    PreferenceExample(
                        id_value=str(id_value),
                        prompt_messages=prompt_messages,
                        chosen_text=chosen_answer,
                        rejected_text=rejected_answer,
                        image_id=image_value,
                    )
                )

    LOGGER.info("Collected %d preference pairs from %d unique IDs.", len(examples), grouped.ngroups)
    if not examples:
        raise ValueError("No valid preference examples were found. Verify the CSV contents.")

    return examples


def normalize_text_field(value: Any, fallback: str) -> str:
    """Convert a CSV field into a clean string."""
    if isinstance(value, float) and math.isnan(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def split_examples(
    examples: Iterable[PreferenceExample],
    val_split: float,
    seed: int,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split preference examples by ID to avoid leakage between train and validation."""
    per_id: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        per_id.setdefault(example.id_value, []).append(example)

    ids = list(per_id.keys())
    random.Random(seed).shuffle(ids)

    if val_split <= 0.0 or len(ids) < 2:
        train_records = [item for per_examples in per_id.values() for item in per_examples]
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
    processor: AutoProcessor,
    image_root: Path,
) -> Dataset:
    """Turn preference examples into a Hugging Face dataset compatible with TRL's VLM DPO trainer."""
    records = [example.to_record(processor, image_root) for example in examples]
    dataset = Dataset.from_list(records)

    # Cast the images column so that examples yield PIL images on access.
    image_feature = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast_column("images", image_feature)

    return dataset


def resolve_image_path(image_root: Path, image_id: str) -> Path:
    candidate = Path(image_id)
    if candidate.suffix:
        return image_root / candidate.name
    return image_root / f"{image_id}.jpg"


def main() -> None:
    parser = TrlParser((DatasetArguments, DPOConfig, ModelConfig))
    cli_args = list(sys.argv[1:])
    user_supplied_precompute = any(arg.startswith("--precompute_ref_log_probs") for arg in cli_args)
    if "--bf16" not in cli_args and "--no_bf16" not in cli_args:
        cli_args.extend(["--bf16", "False"])
        LOGGER.info("Defaulting to --bf16 False (pass --bf16 True explicitly to override).")
    dataset_args, training_args, model_args = parser.parse_args_and_config(cli_args)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )

    dataset_args.image_root = dataset_args.image_root.expanduser()
    dataset_args.csv_path = dataset_args.csv_path.expanduser()
    training_args.output_dir = training_args.output_dir or "outputs/dpo-llava-work"
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    examples = collect_preference_examples(dataset_args.csv_path, dataset_args.prompt_fallback)
    train_examples, val_examples = split_examples(examples, dataset_args.val_split, dataset_args.dataset_seed)

    model_name = model_args.model_name_or_path or "./models/llava-v1.6-mistral-7b-hf"

    LOGGER.info("Loading processor from %s", model_name)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=dataset_args.local_files_only,
    )

    train_dataset = build_dataset(train_examples, processor, dataset_args.image_root)
    eval_dataset = build_dataset(val_examples, processor, dataset_args.image_root) if val_examples else None

    preferred_dtype = torch.bfloat16
    if model_args.dtype not in (None, "auto"):
        preferred_dtype = getattr(torch, model_args.dtype)

    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank: Optional[int] = int(local_rank_env) if local_rank_env is not None else None

    model_kwargs: dict[str, Any] = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": preferred_dtype,
        "trust_remote_code": True,
    }

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        if local_rank is not None:
            model_kwargs["device_map"] = {"": local_rank}
        else:
            model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs.pop("torch_dtype", None)  # dtype handled by quantization config
    else:
        if local_rank is not None:
            model_kwargs["device_map"] = {"": local_rank}
        else:
            model_kwargs.setdefault("device_map", "auto")

    LOGGER.info("Loading policy model from %s", model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        local_files_only=dataset_args.local_files_only,
        **model_kwargs,
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        LOGGER.info("Loading reference model for DPO loss.")
        ref_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            local_files_only=dataset_args.local_files_only,
            **model_kwargs,
        )
    else:
        ref_model = None

    training_args.remove_unused_columns = False
    if training_args.seed is None:
        training_args.seed = dataset_args.dataset_seed
    if eval_dataset is not None and training_args.eval_strategy == "no":
        training_args.eval_strategy = "steps"
    if eval_dataset is None:
        training_args.eval_strategy = "no"

    if training_args.precompute_ref_log_probs and not user_supplied_precompute:
        LOGGER.info(
            "Disabling precompute_ref_log_probs to avoid a lengthy reference-model warmup. "
            "Pass --precompute_ref_log_probs True explicitly if you prefer the original behaviour."
        )
        training_args.precompute_ref_log_probs = False

    # LLaVA expands a single <image> token into thousands of vision placeholders, so keep generous limits
    required_prompt_length = 4096
    if training_args.max_prompt_length is None or training_args.max_prompt_length < required_prompt_length:
        LOGGER.info("Raising max_prompt_length to %d to preserve all LLaVA image tokens.", required_prompt_length)
        training_args.max_prompt_length = required_prompt_length

    if training_args.max_length is None or training_args.max_length < training_args.max_prompt_length + 1024:
        training_args.max_length = training_args.max_prompt_length + 1024
        LOGGER.info("Setting max_length to %d so prompts and completions fit without truncation.", training_args.max_length)

    trainer = LlavaVisionDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
