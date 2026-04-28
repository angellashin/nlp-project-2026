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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


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
) -> Path:
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
                "model": model_name,
                "condition": row["condition"],
                "target_id": row["target_id"],
                "gold_label": row["label"],
                "predicted_label": parsed or "invalid",
                "metric_label": metric_label,
                "raw_output": raw,
                "invalid": invalid,
                "retried": retried,
                "prompt_version": prompt_version,
            }
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
    parser.add_argument("--prompt-version", default="qwen_mvp_v1")
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
    )
    print(output)


if __name__ == "__main__":
    main()
