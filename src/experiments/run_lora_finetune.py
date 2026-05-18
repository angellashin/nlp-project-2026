from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

from src.common.io import ensure_dir, iter_jsonl, write_json
from src.experiments.run_prompting import LABELS, format_chat, load_model


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def select_training_rows(
    variants_path: Path,
    condition: str,
    max_samples: Optional[int],
) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(variants_path):
        if row["condition"] != condition:
            continue
        if row["label"] not in LABELS:
            continue
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


class StanceSftDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        rendered_prompt = format_chat(self.tokenizer, row["prompt_text"])
        answer_text = row["label"] + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(rendered_prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False).input_ids
        available_prompt_tokens = max(self.max_length - len(answer_ids), 1)
        if len(prompt_ids) > available_prompt_tokens:
            prompt_ids = prompt_ids[-available_prompt_tokens:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class DataCollatorForCausalLabels:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train_lora(
    train_variants: Path,
    out_dir: Path,
    model_name: str,
    condition: str,
    dtype: str,
    device_map: str,
    max_train_samples: Optional[int],
    max_length: int,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
) -> Path:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("LoRA fine-tuning requires peft and transformers. Install requirements first.") from exc

    started = time.time()
    rows = select_training_rows(train_variants, condition, max_train_samples)
    if not rows:
        raise RuntimeError(f"No training rows found for condition={condition} in {train_variants}")
    print(f"[data] selected {len(rows)} {condition} rows from {train_variants}", flush=True)

    tokenizer, model = load_model(model_name, dtype, device_map)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = StanceSftDataset(rows, tokenizer, max_length=max_length)
    collator = DataCollatorForCausalLabels(tokenizer)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=dtype == "float16",
        bf16=dtype == "bfloat16",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    write_json(
        out_dir / "training_metadata.json",
        {
            "mode": "lora_sft",
            "model": model_name,
            "train_variants": str(train_variants),
            "condition": condition,
            "train_rows": len(rows),
            "max_train_samples": max_train_samples,
            "max_length": max_length,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "gradient_checkpointing": gradient_checkpointing,
            "seconds": round(time.time() - started, 3),
        },
    )
    print(f"[done] saved adapter to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a small LM on RumourEval stance variants.")
    parser.add_argument("--train-variants", default="data/variants/train_context_variants.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--condition", choices=["reply_only", "useful"], required=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    args = parser.parse_args()

    train_lora(
        Path(args.train_variants),
        ensure_dir(args.out_dir),
        args.model,
        args.condition,
        args.dtype,
        args.device_map,
        args.max_train_samples,
        args.max_length,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.target_modules,
        args.gradient_checkpointing,
        args.logging_steps,
        args.save_steps,
    )


if __name__ == "__main__":
    main()
