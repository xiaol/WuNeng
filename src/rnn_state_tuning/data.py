from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def _chat_token_ids(tokenizer, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "ids"):
        token_ids = token_ids.ids
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return token_ids


def encode_record(tokenizer, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    if "messages" in record:
        messages = record["messages"]
    elif "prompt" in record and "response" in record:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    elif "text" in record:
        input_ids = tokenizer(record["text"], add_special_tokens=True, truncation=True, max_length=max_length)[
            "input_ids"
        ]
        return {"input_ids": input_ids, "labels": list(input_ids)}
    else:
        raise ValueError("Each JSONL record needs messages, prompt/response, or text")

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    input_ids = _chat_token_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(input_ids)

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start = (
            len(_chat_token_ids(tokenizer, messages[:index], add_generation_prompt=True))
            if index > 0
            else 0
        )
        end = len(_chat_token_ids(tokenizer, messages[: index + 1], add_generation_prompt=False))
        start = min(start, len(input_ids))
        end = min(end, len(input_ids))
        labels[start:end] = input_ids[start:end]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    if all(label == -100 for label in labels):
        raise ValueError("Record has no assistant tokens inside max_length")
    return {"input_ids": input_ids, "labels": labels}


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


class StateTuningDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int,
        *,
        max_examples: int | None = None,
        sample_seed: int = 42,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets: list[tuple[int, int]] = []
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if line.strip():
                    self.offsets.append((offset, line_number))
        if not self.offsets:
            raise ValueError(f"Dataset is empty: {path}")
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            if max_examples < len(self.offsets):
                self.offsets = random.Random(sample_seed).sample(self.offsets, max_examples)

        self[0]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        offset, line_number = self.offsets[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from error
        try:
            return encode_record(self.tokenizer, record, self.max_length)
        except ValueError as error:
            raise ValueError(f"Cannot encode {self.path}:{line_number}: {error}") from error


class RightPaddingCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids = []
        labels = []
        attention_mask = []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            labels.append(example["labels"] + [-100] * padding)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
