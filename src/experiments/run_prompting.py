from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Optional, Set

from src.common.io import ensure_dir, iter_jsonl, write_json, write_jsonl


LABELS = ["support", "deny", "query", "comment"]
LABEL_RE = re.compile(r"\b(support|deny|query|comment)\b", re.IGNORECASE)


STRICT_SUFFIX = "\n\nReturn only one of: support, deny, query, comment."


def parse_label(text: str) -> Optional[str]:
    match = LABEL_RE.search(text.strip())
    if not match:
        return None
    return match.group(1).lower()


def load_model(model_name: str, dtype: str, device_map: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "run_prompting.py requires torch and transformers. Install project "
            "dependencies in the GPU environment before running model inference."
        ) from exc

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[dtype]
    started = time.time()
    print(f"[load] Loading tokenizer for {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"[load] Loading model for {model_name} with dtype={dtype}, device_map={device_map}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[load] Model ready in {time.time() - started:.1f}s", flush=True)
    return tokenizer, model


def count_selected_variants(variants_path: Path, conditions: Optional[Set[str]], limit: Optional[int]) -> int:
    total = 0
    for row in iter_jsonl(variants_path):
        if conditions and row["condition"] not in conditions:
            continue
        if limit is not None and total >= limit:
            break
        total += 1
    return total


def format_chat(tokenizer: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful stance classifier for social media rumour discussions.",
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + messages[1]["content"] + "\nAnswer:"


def generate_one(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int) -> str:
    import torch

    rendered = format_chat(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_prompting(
    variants_path: Path,
    out_dir: Path,
    model_name: str,
    conditions: Optional[Set[str]],
    limit: Optional[int],
    dtype: str,
    device_map: str,
    max_new_tokens: int,
    prompt_version: str,
    log_every: int,
) -> Path:
    planned_total = count_selected_variants(variants_path, conditions, limit)
    print(
        f"[run] model={model_name} variants={variants_path} planned_predictions={planned_total}",
        flush=True,
    )
    tokenizer, model = load_model(model_name, dtype, device_map)
    started = time.time()
    predictions = []
    total = 0
    invalid_count = 0

    for row in iter_jsonl(variants_path):
        if conditions and row["condition"] not in conditions:
            continue
        if limit is not None and total >= limit:
            break
        total += 1

        raw = generate_one(tokenizer, model, row["prompt_text"], max_new_tokens)
        parsed = parse_label(raw)
        retried = False
        if parsed is None:
            retried = True
            raw_retry = generate_one(tokenizer, model, row["prompt_text"] + STRICT_SUFFIX, max_new_tokens)
            parsed = parse_label(raw_retry)
            raw = raw + "\n[retry]\n" + raw_retry

        invalid = parsed is None
        if invalid:
            invalid_count += 1
        metric_label = parsed if parsed in LABELS else "comment"

        predictions.append(
            {
                "example_id": row["example_id"],
                "split": row["split"],
                "platform": row.get("platform"),
                "event": row.get("event"),
                "model": model_name,
                "condition": row["condition"],
                "context_source": row.get("context_source"),
                "target_id": row["target_id"],
                "thread_id": row.get("thread_id"),
                "source_id": row.get("source_id"),
                "parent_id": row.get("parent_id"),
                "depth": row.get("depth"),
                "depth_bucket": row.get("depth_bucket"),
                "parent_available": row.get("parent_available"),
                "mixed_valid": row.get("mixed_valid"),
                "has_conflicting_reply": row.get("has_conflicting_reply"),
                "conflict_relation": row.get("conflict_relation"),
                "irrelevant_relation": row.get("irrelevant_relation"),
                "context_item_count": row.get("context_item_count"),
                "uses_gold_labels_for_stress_test": row.get("uses_gold_labels_for_stress_test"),
                "gold_label": row["label"],
                "predicted_label": parsed or "invalid",
                "metric_label": metric_label,
                "raw_output": raw,
                "invalid": invalid,
                "retried": retried,
                "prompt_version": prompt_version,
            }
        )
        if log_every > 0 and (total == 1 or total % log_every == 0 or total == planned_total):
            elapsed = time.time() - started
            rate = total / elapsed if elapsed else 0.0
            remaining = planned_total - total
            eta = remaining / rate if rate else 0.0
            print(
                "[progress] "
                f"{total}/{planned_total} predictions "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s invalid={invalid_count}",
                flush=True,
            )

    output = out_dir / "predictions.jsonl"
    write_jsonl(output, predictions)
    write_json(
        out_dir / "run_metadata.json",
        {
            "model": model_name,
            "variants": str(variants_path),
            "total_predictions": total,
            "invalid_predictions": invalid_count,
            "invalid_rate": invalid_count / total if total else 0.0,
            "seconds": round(time.time() - started, 3),
            "dtype": dtype,
            "device_map": device_map,
            "max_new_tokens": max_new_tokens,
            "prompt_version": prompt_version,
            "log_every": log_every,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference-only prompting for context variants.")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt-version", default="qwen_mvp_v2")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    conditions = set(args.conditions) if args.conditions else None
    output = run_prompting(
        Path(args.variants),
        out_dir,
        args.model,
        conditions,
        args.limit,
        args.dtype,
        args.device_map,
        args.max_new_tokens,
        args.prompt_version,
        args.log_every,
    )
    print(output)


if __name__ == "__main__":
    main()
