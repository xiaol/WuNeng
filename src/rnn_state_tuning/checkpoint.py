from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .adapter import (
    iter_ple_branches,
    iter_ple_embeddings,
    iter_state_modules,
    normalize_method,
    prepare_model_for_tuning,
)


CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"


def _adapter_paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return path.with_name(CONFIG_NAME), path
    return path / CONFIG_NAME, path / WEIGHTS_NAME


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    weights = {
        f"layers.{module.layer_idx}.recurrent_state": module.initial_state.detach().cpu().contiguous()
        for module in iter_state_modules(model)
    }
    ple_modules = list(iter_ple_embeddings(model))
    if len(ple_modules) > 1:
        raise ValueError("Expected one shared PLE module")
    if ple_modules:
        ple = ple_modules[0]
        weights.update(
            {
                "ple.token_embeddings.weight": ple.token_embeddings.detach().cpu().contiguous(),
                "ple.model_projection.weight": ple.model_projection.weight.detach().cpu().contiguous(),
                "ple.projection_norm.weight": ple.projection_norm.weight.detach().cpu().contiguous(),
            }
        )
    for branch in iter_ple_branches(model):
        prefix = f"ple.layers.{branch.layer_idx}"
        weights.update(
            {
                f"{prefix}.gate.weight": branch.gate.weight.detach().cpu().contiguous(),
                f"{prefix}.projection.weight": branch.projection.weight.detach().cpu().contiguous(),
                f"{prefix}.norm.weight": branch.norm.weight.detach().cpu().contiguous(),
            }
        )
    return weights


def _checkpoint_method(has_state: bool, has_ple: bool) -> str:
    if has_state and has_ple:
        return "direct_recurrent_state+per_layer_embeddings"
    if has_ple:
        return "per_layer_embeddings"
    return "direct_recurrent_state"


def save_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    base_model: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    config_path, weights_path = _adapter_paths(path)
    weights = adapter_state_dict(model)
    if not weights:
        raise ValueError("No state-tuning or PLE modules are attached to the model")

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, weights_path, metadata={"format": "rnn-state-tuning", "version": "2"})

    state_modules = list(iter_state_modules(model))
    ple_modules = list(iter_ple_embeddings(model))
    ple_branches = list(iter_ple_branches(model))
    config = {
        "format_version": 2,
        "backend": "qwen3_5",
        "method": _checkpoint_method(bool(state_modules), bool(ple_modules)),
        "base_model": base_model,
        "layers": [
            {"layer_idx": module.layer_idx, "shape": list(module.state_shape)} for module in state_modules
        ],
        "parameter_count": sum(tensor.numel() for tensor in weights.values()),
    }
    if state_modules:
        config["state_dtype"] = str(state_modules[0].initial_state.dtype).removeprefix("torch.")
    if ple_modules:
        ple = ple_modules[0]
        config["ple"] = {
            "dim": ple.ple_dim,
            "dtype": str(ple.token_embeddings.dtype).removeprefix("torch."),
            "layers": [branch.layer_idx for branch in ple_branches],
            "norm_position": ple_branches[0].norm_position,
            "vocab_size": ple.vocab_size,
        }
    if extra_metadata:
        config["metadata"] = extra_metadata
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return weights_path


def load_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    prepare: bool = True,
) -> dict[str, Any]:
    config_path, weights_path = _adapter_paths(path)
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if config and prepare:
        method = normalize_method(config.get("method", "direct_recurrent_state"))
        ple_config = config.get("ple", {})
        ple_dtype_name = ple_config.get("dtype")
        ple_dtype = getattr(torch, ple_dtype_name) if ple_dtype_name else None
        prepare_model_for_tuning(
            model,
            method=method,
            backend=config.get("backend", "auto"),
            state_dtype=getattr(torch, config.get("state_dtype", "float32")),
            ple_dim=ple_config.get("dim", 256),
            ple_dtype=ple_dtype,
            ple_norm_position=ple_config.get("norm_position", "post"),
        )
    weights = load_file(weights_path, device="cpu")

    expected = set()
    with torch.no_grad():
        for module in iter_state_modules(model):
            key = f"layers.{module.layer_idx}.recurrent_state"
            expected.add(key)
            if key not in weights:
                if strict:
                    raise KeyError(f"Adapter is missing {key}")
                continue
            value = weights[key]
            if tuple(value.shape) != module.state_shape:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint has {tuple(value.shape)}, "
                    f"model expects {module.state_shape}"
                )
            module.initial_state.copy_(value)

        ple_modules = list(iter_ple_embeddings(model))
        if len(ple_modules) > 1:
            raise ValueError("Expected one shared PLE module")
        if ple_modules:
            ple = ple_modules[0]
            ple_tensors = {
                "ple.token_embeddings.weight": ple.token_embeddings,
                "ple.model_projection.weight": ple.model_projection.weight,
                "ple.projection_norm.weight": ple.projection_norm.weight,
            }
            for key, parameter in ple_tensors.items():
                expected.add(key)
                if key not in weights:
                    if strict:
                        raise KeyError(f"Adapter is missing {key}")
                    continue
                value = weights[key]
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: checkpoint has {tuple(value.shape)}, "
                        f"model expects {tuple(parameter.shape)}"
                    )
                parameter.copy_(value)

        for branch in iter_ple_branches(model):
            prefix = f"ple.layers.{branch.layer_idx}"
            branch_tensors = {
                f"{prefix}.gate.weight": branch.gate.weight,
                f"{prefix}.projection.weight": branch.projection.weight,
                f"{prefix}.norm.weight": branch.norm.weight,
            }
            for key, parameter in branch_tensors.items():
                expected.add(key)
                if key not in weights:
                    if strict:
                        raise KeyError(f"Adapter is missing {key}")
                    continue
                value = weights[key]
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: checkpoint has {tuple(value.shape)}, "
                        f"model expects {tuple(parameter.shape)}"
                    )
                parameter.copy_(value)

    unexpected = set(weights) - expected
    if strict and unexpected:
        raise KeyError(f"Adapter contains unexpected tensors: {sorted(unexpected)}")
    return config
