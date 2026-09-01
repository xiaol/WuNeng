from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        return DTYPES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose from auto, {', '.join(DTYPES)}") from error


def load_qwen35_model(
    model_name_or_path: str | Path,
    *,
    dtype: torch.dtype,
    device_map: str | dict | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
) -> nn.Module:
    model_name_or_path = str(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    kwargs = {
        "config": config,
        "dtype": dtype,
        "device_map": device_map,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if config.model_type == "qwen3_5":
        return Qwen3_5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    if config.model_type == "qwen3_5_text":
        return Qwen3_5ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    raise ValueError(f"Expected a Qwen3.5 checkpoint, found model type {config.model_type!r}")


def input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device
