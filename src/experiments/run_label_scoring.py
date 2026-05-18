from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional, Set

from src.common.io import ensure_dir, iter_jsonl, write_json, write_jsonl
from src.experiments.run_prompting import LABELS, count_selected_variants, format_chat, load_model


def model_input_device(model: Any) -> Any:
    return next(model.parameters()).device


def maybe_load_adapter(model: Any, adapter_path: Optional[str]) -> Any:
    if not adapter_path:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Adapter scoring requires peft. Install it in the GPU environment.") from exc
    print(f"[load] Loading LoRA adapter from {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def label_token_ids(tokenizer: Any, label: str) -> list[int]:
    ids = tokenizer(label, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Tokenizer produced no ids for label {label!r}")
    return ids


def score_label_batch(
    tokenizer: Any,
    model: Any,
    prompt: str,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    import torch

    rendered = format_chat(tokenizer, prompt)
    base_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if not base_ids:
        raise ValueError("Rendered prompt produced no token ids.")

    encoded_labels = {label: label_token_ids(tokenizer, label) for label in labels}
    full_sequences = [
        torch.tensor(base_ids + encoded_labels[label], dtype=torch.long)
        for label in labels
    ]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded = torch.nn.utils.rnn.pad_sequence(full_sequences, batch_first=True, padding_value=pad_id)
    attention_mask = torch.zeros_like(padded)
    for i, seq in enumerate(full_sequences):
        attention_mask[i, : len(seq)] = 1

    device = model_input_device(model)
    padded = padded.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=padded, attention_mask=attention_mask).logits

    scores: dict[str, dict[str, float]] = {}
    base_len = len(base_ids)
    for i, label in enumerate(labels):
        target_ids = torch.tensor(encoded_labels[label], dtype=torch.long, device=logits.device)
        start = base_len - 1
        end = start + len(target_ids)
        token_logits = logits[i, start:end, :]
        token_logprobs = torch.log_softmax(token_logits, dim=-1).gather(1, target_ids.unsqueeze(1)).squeeze(1)
        sum_logprob = float(token_logprobs.sum().item())
        mean_logprob = float(token_logprobs.mean().item())
        scores[label] = {
            "sum_logprob": sum_logprob,
            "mean_logprob": mean_logprob,
            "token_count": len(encoded_labels[label]),
        }
    return scores


def choose_label(scores: dict[str, dict[str, float]], selection: str) -> str:
    score_key = "mean_logprob" if selection == "mean" else "sum_logprob"
    return max(LABELS, key=lambda label: scores[label][score_key])


def normalized_probs(scores: dict[str, dict[str, float]], selection: str) -> dict[str, float]:
    score_key = "mean_logprob" if selection == "mean" else "sum_logprob"
    values = {label: scores[label][score_key] for label in LABELS}
    max_value = max(values.values())
    exp_values = {label: math.exp(value - max_value) for label, value in values.items()}
    total = sum(exp_values.values())
    return {label: value / total for label, value in exp_values.items()}


def prediction_row(
    variant: dict[str, Any],
    model_name: str,
    predicted_label: str,
    scores: dict[str, dict[str, float]],
    probabilities: dict[str, float],
    prompt_version: str,
    selection: str,
    adapter_path: Optional[str],
) -> dict[str, Any]:
    return {
        "example_id": variant["example_id"],
        "split": variant["split"],
        "platform": variant.get("platform"),
        "event": variant.get("event"),
        "model": model_name if not adapter_path else f"{model_name}+adapter:{Path(adapter_path).name}",
        "base_model": model_name,
        "adapter_path": adapter_path,
        "condition": variant["condition"],
        "context_source": variant.get("context_source"),
        "target_id": variant["target_id"],
        "thread_id": variant.get("thread_id"),
        "source_id": variant.get("source_id"),
        "parent_id": variant.get("parent_id"),
        "depth": variant.get("depth"),
        "depth_bucket": variant.get("depth_bucket"),
        "parent_available": variant.get("parent_available"),
        "mixed_valid": variant.get("mixed_valid"),
        "has_conflicting_reply": variant.get("has_conflicting_reply"),
        "conflict_relation": variant.get("conflict_relation"),
        "irrelevant_relation": variant.get("irrelevant_relation"),
        "context_item_count": variant.get("context_item_count"),
        "uses_gold_labels_for_stress_test": variant.get("uses_gold_labels_for_stress_test"),
        "gold_label": variant["label"],
        "predicted_label": predicted_label,
        "metric_label": predicted_label,
        "raw_output": f"label_scoring:{predicted_label}",
        "invalid": False,
        "retried": False,
        "prompt_version": prompt_version,
        "scoring_selection": selection,
        "label_scores": scores,
        "label_probabilities": probabilities,
    }


def run_label_scoring(
    variants_path: Path,
    out_dir: Path,
    model_name: str,
    adapter_path: Optional[str],
    conditions: Optional[Set[str]],
    limit: Optional[int],
    dtype: str,
    device_map: str,
    prompt_version: str,
    selection: str,
    log_every: int,
) -> Path:
    planned_total = count_selected_variants(variants_path, conditions, limit)
    print(
        f"[run] label scoring model={model_name} adapter={adapter_path} variants={variants_path} planned={planned_total}",
        flush=True,
    )
    tokenizer, model = load_model(model_name, dtype, device_map)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = maybe_load_adapter(model, adapter_path)

    started = time.time()
    predictions = []
    total = 0
    for variant in iter_jsonl(variants_path):
        if conditions and variant["condition"] not in conditions:
            continue
        if limit is not None and total >= limit:
            break
        total += 1
        scores = score_label_batch(tokenizer, model, variant["prompt_text"], LABELS)
        predicted = choose_label(scores, selection)
        probabilities = normalized_probs(scores, selection)
        predictions.append(
            prediction_row(
                variant,
                model_name,
                predicted,
                scores,
                probabilities,
                prompt_version,
                selection,
                adapter_path,
            )
        )
        if log_every > 0 and (total == 1 or total % log_every == 0 or total == planned_total):
            elapsed = time.time() - started
            rate = total / elapsed if elapsed else 0.0
            eta = (planned_total - total) / rate if rate else 0.0
            print(f"[progress] {total}/{planned_total} scored elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    output = out_dir / "predictions.jsonl"
    write_jsonl(output, predictions)
    write_json(
        out_dir / "run_metadata.json",
        {
            "mode": "label_scoring",
            "model": model_name,
            "adapter_path": adapter_path,
            "variants": str(variants_path),
            "total_predictions": total,
            "invalid_predictions": 0,
            "invalid_rate": 0.0,
            "seconds": round(time.time() - started, 3),
            "dtype": dtype,
            "device_map": device_map,
            "prompt_version": prompt_version,
            "scoring_selection": selection,
            "log_every": log_every,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forced-choice label scoring for RumourEval variants.")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt-version", default="qwen_mvp_v3")
    parser.add_argument("--selection", choices=["mean", "sum"], default="mean")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    output = run_label_scoring(
        Path(args.variants),
        ensure_dir(args.out_dir),
        args.model,
        args.adapter,
        set(args.conditions) if args.conditions else None,
        args.limit,
        args.dtype,
        args.device_map,
        args.prompt_version,
        args.selection,
        args.log_every,
    )
    print(output)


if __name__ == "__main__":
    main()
