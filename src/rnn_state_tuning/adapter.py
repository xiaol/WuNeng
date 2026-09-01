from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import nn

from .backends.qwen3_5 import (
    Qwen35PLEBranch,
    Qwen35PerLayerEmbeddings,
    Qwen35StateTuningGatedDeltaNet,
    attach_qwen35_ple,
    attach_qwen35_state_tuning,
)


TuningMethod = Literal["state", "ple", "state+ple"]


@dataclass(frozen=True)
class StateTuningInfo:
    backend: str
    method: TuningMethod
    layer_count: int
    parameter_count: int
    state_dtype: torch.dtype
    state_layer_count: int
    ple_layer_count: int
    ple_dim: int | None
    ple_dtype: torch.dtype | None
    ple_norm_position: str | None


def _detect_backend(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", None)
    if model_type in {"qwen3_5", "qwen3_5_text"} or text_model_type == "qwen3_5_text":
        return "qwen3_5"
    raise ValueError(
        f"No state-tuning backend is registered for model type {model_type!r}. "
        "Currently supported: qwen3_5."
    )


def iter_state_modules(model: nn.Module) -> Iterator[Qwen35StateTuningGatedDeltaNet]:
    for module in model.modules():
        if isinstance(module, Qwen35StateTuningGatedDeltaNet):
            yield module


def state_tuning_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in iter_state_modules(model):
        yield module.initial_state


def iter_ple_embeddings(model: nn.Module) -> Iterator[Qwen35PerLayerEmbeddings]:
    for module in model.modules():
        if isinstance(module, Qwen35PerLayerEmbeddings):
            yield module


def iter_ple_branches(model: nn.Module) -> Iterator[Qwen35PLEBranch]:
    for module in model.modules():
        if isinstance(module, Qwen35PLEBranch):
            yield module


def ple_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in iter_ple_embeddings(model):
        yield from module.parameters()
    for module in iter_ple_branches(model):
        yield from module.parameters()


def tuning_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    yield from (parameter for parameter in model.parameters() if parameter.requires_grad)


def set_trainable_components(model: nn.Module, components: str) -> int:
    components = normalize_method(components)
    train_state = components in {"state", "state+ple"}
    train_ple = components in {"ple", "state+ple"}

    for module in iter_state_modules(model):
        module.initial_state.requires_grad_(train_state)
    for module in iter_ple_embeddings(model):
        for parameter in module.parameters():
            parameter.requires_grad_(train_ple)
    for module in iter_ple_branches(model):
        for parameter in module.parameters():
            parameter.requires_grad_(train_ple)
    return sum(parameter.numel() for parameter in tuning_parameters(model))


def normalize_method(method: str) -> TuningMethod:
    aliases = {
        "state": "state",
        "direct_recurrent_state": "state",
        "ple": "ple",
        "per_layer_embeddings": "ple",
        "state+ple": "state+ple",
        "direct_recurrent_state+per_layer_embeddings": "state+ple",
    }
    try:
        return aliases[method]
    except KeyError as error:
        raise ValueError("Unknown tuning method; choose from state, ple, state+ple") from error


def prepare_model_for_tuning(
    model: nn.Module,
    *,
    method: str = "state",
    backend: str = "auto",
    state_dtype: torch.dtype = torch.float32,
    ple_dim: int = 256,
    ple_dtype: torch.dtype | None = None,
    ple_norm_position: str = "post",
) -> StateTuningInfo:
    method = normalize_method(method)
    if backend == "auto":
        backend = _detect_backend(model)
    if backend != "qwen3_5":
        raise ValueError(f"Unknown state-tuning backend: {backend}")

    has_state = any(iter_state_modules(model))
    has_ple = any(iter_ple_embeddings(model))
    if has_state and method == "ple":
        raise ValueError("The model already has state tuning attached; use method='state+ple'")
    if has_ple and method == "state":
        raise ValueError("The model already has PLE attached; use method='state+ple'")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    state_modules = []
    if method in {"state", "state+ple"}:
        state_modules = attach_qwen35_state_tuning(model, state_dtype=state_dtype)

    ple = None
    ple_branches = []
    if method in {"ple", "state+ple"}:
        ple, ple_branches = attach_qwen35_ple(
            model,
            ple_dim=ple_dim,
            ple_dtype=ple_dtype,
            norm_position=ple_norm_position,
        )
    set_trainable_components(model, method)

    if hasattr(model, "config"):
        model.config.use_cache = False
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = False

    parameters = list(tuning_parameters(model))
    return StateTuningInfo(
        backend=backend,
        method=method,
        layer_count=max(len(state_modules), len(ple_branches)),
        parameter_count=sum(parameter.numel() for parameter in parameters),
        state_dtype=state_dtype,
        state_layer_count=len(state_modules),
        ple_layer_count=len(ple_branches),
        ple_dim=ple.ple_dim if ple is not None else None,
        ple_dtype=ple.token_embeddings.dtype if ple is not None else None,
        ple_norm_position=ple_branches[0].norm_position if ple_branches else None,
    )


def prepare_model_for_state_tuning(
    model: nn.Module,
    *,
    backend: str = "auto",
    state_dtype: torch.dtype = torch.float32,
) -> StateTuningInfo:
    return prepare_model_for_tuning(
        model,
        method="state",
        backend=backend,
        state_dtype=state_dtype,
    )
