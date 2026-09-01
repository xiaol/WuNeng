from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from rnn_state_tuning import load_adapter
from rnn_state_tuning.data import RightPaddingCollator, StateTuningDataset
from rnn_state_tuning.modeling import input_device, load_qwen35_model, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a Qwen3.5 state/PLE adapter")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Gold JSONL evaluation data")
    parser.add_argument("--adapter", help="Adapter directory; omit for the frozen base model")
    parser.add_argument("--output", help="Optional JSON result path")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-limit", type=int, help="Score generation on the first N evaluation rows")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    return parser.parse_args()


def _load_records(path: str | Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def _boundaries(text: str) -> list[int] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    boundaries = value.get("boundaries") if isinstance(value, dict) else None
    if not isinstance(boundaries, list) or not all(isinstance(item, int) for item in boundaries):
        return None
    return sorted(set(boundaries))


def _target_boundaries(record: dict[str, Any]) -> list[int]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Boundary benchmark records must end with an assistant message")
    boundaries = _boundaries(messages[-1]["content"])
    if boundaries is None:
        raise ValueError("Gold assistant response is not valid boundary JSON")
    return boundaries


def evaluate_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    data_path: str | Path,
    max_length: int,
    limit: int | None,
) -> tuple[float, int]:
    full_dataset = StateTuningDataset(data_path, tokenizer, max_length)
    dataset = Subset(full_dataset, range(min(limit, len(full_dataset)))) if limit is not None else full_dataset
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=RightPaddingCollator(tokenizer.pad_token_id),
    )
    device = input_device(model)
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            output = model(**batch, use_cache=False)
            token_count = batch["labels"][:, 1:].ne(-100).sum().item()
            total_loss += output.loss.float().item() * token_count
            total_tokens += token_count
    return total_loss / total_tokens, total_tokens


def evaluate_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
    max_new_tokens: int,
) -> dict[str, int | float]:
    device = input_device(model)
    exact = 0
    invalid = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    predictions = []

    with torch.inference_mode():
        for record in records:
            messages = record["messages"]
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(prompt_ids, Mapping):
                prompt_ids = prompt_ids["input_ids"]
            if hasattr(prompt_ids, "ids"):
                prompt_ids = prompt_ids.ids
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids = prompt_ids.tolist()
            prompt_ids = prompt_ids[-max_length:]
            inputs = {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
            }
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            predicted = _boundaries(generated)
            target = _target_boundaries(record)
            is_valid = predicted is not None
            if predicted is None:
                invalid += 1
                predicted = []
            predictions.append(
                {
                    "generated": generated,
                    "predicted": predicted,
                    "target": target,
                    "valid_json": is_valid,
                }
            )
            exact += int(is_valid and predicted == target)
            predicted_set = set(predicted)
            target_set = set(target)
            true_positive += len(predicted_set & target_set)
            false_positive += len(predicted_set - target_set)
            false_negative += len(target_set - predicted_set)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "examples": len(records),
        "exact_match": exact / len(records),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "invalid_json": invalid,
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_qwen35_model(
        args.model,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    adapter_config = load_adapter(model, args.adapter) if args.adapter else None
    model.eval()

    records = _load_records(args.data, args.limit)
    loss, supervised_tokens = evaluate_loss(model, tokenizer, args.data, args.max_length, args.limit)
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "method": adapter_config.get("method") if adapter_config else "base",
        "data": str(Path(args.data).resolve()),
        "max_length": args.max_length,
        "response_loss": loss,
        "response_perplexity": math.exp(min(loss, 20.0)),
        "supervised_tokens": supervised_tokens,
        **evaluate_generation(
            model,
            tokenizer,
            records[: args.generation_limit] if args.generation_limit is not None else records,
            args.max_length,
            args.max_new_tokens,
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
