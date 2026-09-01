from __future__ import annotations

import math
from typing import Any
import weakref

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


class Qwen35StateTuningGatedDeltaNet(nn.Module):
    def __init__(self, base_layer: Qwen3_5GatedDeltaNet, state_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_layer = base_layer
        self.layer_idx = base_layer.layer_idx
        self.initial_state = nn.Parameter(
            torch.zeros(
                base_layer.num_v_heads,
                base_layer.head_k_dim,
                base_layer.head_v_dim,
                dtype=state_dtype,
                device=base_layer.in_proj_qkv.weight.device,
            )
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return tuple(self.initial_state.shape)

    def _expanded_initial_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.initial_state.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0).expand(
            hidden_states.shape[0], -1, -1, -1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        layer = self.base_layer
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )

        mixed_qkv = layer.in_proj_qkv(hidden_states).transpose(1, 2)
        z = layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, layer.head_v_dim)
        beta = layer.in_proj_b(hidden_states).sigmoid()
        a = layer.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                layer.conv1d.weight.squeeze(1),
                layer.conv1d.bias,
                layer.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv,
                    self.layer_idx,
                    conv_kernel_size=layer.conv_kernel_size,
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv1d.weight.squeeze(1),
                layer.conv1d.bias,
                activation=layer.activation,
                **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [layer.key_dim, layer.key_dim, layer.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)

        g = -layer.A_log.float().exp() * torch.nn.functional.softplus(a.float() + layer.dt_bias)
        if layer.num_v_heads // layer.num_k_heads > 1:
            repeats = layer.num_v_heads // layer.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        if use_precomputed_states:
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]
        else:
            recurrent_state = self._expanded_initial_state(hidden_states)

        cu_seqlens = kwargs.pop("cu_seq_lens_q", None)
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
        z = z.reshape(-1, layer.head_v_dim)
        core_attn_out = layer.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return layer.out_proj(core_attn_out)


class Qwen35PLERMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(hidden_states.dtype)


class Qwen35PerLayerEmbeddings(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        ple_dim: int,
        padding_idx: int | None,
        rms_norm_eps: float,
        initializer_range: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.padding_idx = padding_idx
        self.embedding_scale = math.sqrt(ple_dim)
        self.model_projection_scale = hidden_size**-0.5
        self.input_scale = 2.0**-0.5

        self.token_embeddings = nn.Parameter(
            torch.empty(vocab_size, num_layers * ple_dim, device=device, dtype=dtype)
        )
        self.model_projection = nn.Linear(
            hidden_size,
            num_layers * ple_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.projection_norm = Qwen35PLERMSNorm(
            ple_dim,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.current_inputs: torch.Tensor | None = None
        nn.init.normal_(self.token_embeddings, mean=0.0, std=initializer_range)
        nn.init.normal_(self.model_projection.weight, mean=0.0, std=initializer_range)

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        token_inputs = F.embedding(
            input_ids.to(self.token_embeddings.device),
            self.token_embeddings,
            padding_idx=self.padding_idx,
        )
        token_inputs = token_inputs * self.embedding_scale
        token_inputs = token_inputs.reshape(*input_ids.shape, self.num_layers, self.ple_dim)

        context_inputs = self.project_context(inputs_embeds)
        return (token_inputs + context_inputs) * self.input_scale

    def project_context(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        projection_inputs = inputs_embeds.to(
            device=self.model_projection.weight.device,
            dtype=self.model_projection.weight.dtype,
        )
        context_inputs = self.model_projection(projection_inputs) * self.model_projection_scale
        context_inputs = context_inputs.reshape(*inputs_embeds.shape[:-1], self.num_layers, self.ple_dim)
        return self.projection_norm(context_inputs)

    def capture_model_inputs(
        self,
        _model: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is not None:
            embedding = self._embedding_ref()
            if embedding is None:
                raise RuntimeError("The frozen Qwen token embedding was removed")
            embedding_input_ids = input_ids.to(embedding.weight.device)
            self.current_inputs = self(embedding_input_ids, embedding(embedding_input_ids))
        elif inputs_embeds is not None:
            self.current_inputs = self.project_context(inputs_embeds)
        else:
            self.current_inputs = None

    def layer_input(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.current_inputs is None:
            raise RuntimeError("Qwen PLE requires input_ids; no per-layer inputs were captured")
        per_layer_input = self.current_inputs[:, :, layer_idx, :]
        if per_layer_input.shape[:2] != hidden_states.shape[:2]:
            raise RuntimeError(
                "Qwen PLE input shape does not match the decoder hidden states: "
                f"{tuple(per_layer_input.shape[:2])} != {tuple(hidden_states.shape[:2])}"
            )
        return per_layer_input.to(device=hidden_states.device, dtype=hidden_states.dtype)


class Qwen35PLEBranch(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        hidden_size: int,
        ple_dim: int,
        hidden_activation: str,
        rms_norm_eps: float,
        initializer_range: float,
        norm_position: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if norm_position not in {"post", "pre"}:
            raise ValueError("PLE norm_position must be post or pre")
        self.layer_idx = layer_idx
        self.norm_position = norm_position
        self.gate = nn.Linear(hidden_size, ple_dim, bias=False, device=device, dtype=dtype)
        self.projection = nn.Linear(ple_dim, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = Qwen35PLERMSNorm(
            hidden_size if norm_position == "post" else ple_dim,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.activation = ACT2FN[hidden_activation]
        self.record_rms = False
        self.last_rms: dict[str, torch.Tensor] = {}
        nn.init.normal_(self.gate.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.to(device=self.gate.weight.device, dtype=self.gate.weight.dtype)
        per_layer_input = per_layer_input.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = self.activation(self.gate(hidden_states))
        hidden_states = hidden_states * per_layer_input
        if self.record_rms:
            self.last_rms = {
                "residual": residual.detach().float().square().mean().sqrt(),
                "feature": hidden_states.detach().float().square().mean().sqrt(),
            }
        if self.norm_position == "pre":
            hidden_states = self.norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        if self.record_rms:
            self.last_rms["projection"] = hidden_states.detach().float().square().mean().sqrt()
        if self.norm_position == "post":
            hidden_states = self.norm(hidden_states)
        if self.record_rms:
            self.last_rms["delta"] = hidden_states.detach().float().square().mean().sqrt()
        return residual + hidden_states.to(device=residual.device, dtype=residual.dtype)


class Qwen35PLEDecoderLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        branch: Qwen35PLEBranch,
        ple: Qwen35PerLayerEmbeddings,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.ple_branch = branch
        self.layer_idx = branch.layer_idx
        object.__setattr__(self, "_ple_ref", weakref.ref(ple))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden_states = self.base_layer(*args, **kwargs)
        ple = self._ple_ref()
        if ple is None:
            raise RuntimeError("The shared Qwen PLE module was removed")
        return self.ple_branch(hidden_states, ple.layer_input(self.layer_idx, hidden_states))


def _text_model(model: nn.Module) -> nn.Module:
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        model,
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        embed_tokens = getattr(candidate, "embed_tokens", None)
        if isinstance(layers, nn.ModuleList) and isinstance(embed_tokens, nn.Module):
            return candidate
    raise ValueError("Could not locate the Qwen3.5 text model")


def _text_layers(model: nn.Module) -> nn.ModuleList:
    return _text_model(model).layers


def _base_decoder_layer(decoder_layer: nn.Module) -> nn.Module:
    if isinstance(decoder_layer, Qwen35PLEDecoderLayer):
        return decoder_layer.base_layer
    return decoder_layer


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def attach_qwen35_state_tuning(
    model: nn.Module,
    *,
    state_dtype: torch.dtype = torch.float32,
) -> list[Qwen35StateTuningGatedDeltaNet]:
    attached = []
    for decoder_layer in _text_layers(model):
        base_decoder_layer = _base_decoder_layer(decoder_layer)
        linear_attn = getattr(base_decoder_layer, "linear_attn", None)
        if isinstance(linear_attn, Qwen35StateTuningGatedDeltaNet):
            attached.append(linear_attn)
        elif isinstance(linear_attn, Qwen3_5GatedDeltaNet):
            wrapped = Qwen35StateTuningGatedDeltaNet(linear_attn, state_dtype=state_dtype)
            base_decoder_layer.linear_attn = wrapped
            attached.append(wrapped)
    if not attached:
        raise ValueError("The model has no Qwen3.5 Gated DeltaNet layers")
    return attached


def attach_qwen35_ple(
    model: nn.Module,
    *,
    ple_dim: int = 256,
    ple_dtype: torch.dtype | None = None,
    norm_position: str = "post",
) -> tuple[Qwen35PerLayerEmbeddings, list[Qwen35PLEBranch]]:
    if ple_dim <= 0:
        raise ValueError("ple_dim must be positive")
    if norm_position not in {"post", "pre"}:
        raise ValueError("PLE norm_position must be post or pre")

    text_model = _text_model(model)
    existing = getattr(text_model, "rnn_state_tuning_ple", None)
    if isinstance(existing, Qwen35PerLayerEmbeddings):
        if existing.ple_dim != ple_dim:
            raise ValueError(f"Qwen PLE is already attached with ple_dim={existing.ple_dim}")
        branches = [layer.ple_branch for layer in text_model.layers if isinstance(layer, Qwen35PLEDecoderLayer)]
        if branches and branches[0].norm_position != norm_position:
            raise ValueError(f"Qwen PLE is already attached with norm_position={branches[0].norm_position}")
        return existing, branches

    config = text_model.config
    embedding_weight = text_model.embed_tokens.weight
    device = embedding_weight.device
    dtype = ple_dtype or embedding_weight.dtype
    num_layers = len(text_model.layers)
    ple = Qwen35PerLayerEmbeddings(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=num_layers,
        ple_dim=ple_dim,
        padding_idx=getattr(config, "pad_token_id", None),
        rms_norm_eps=config.rms_norm_eps,
        initializer_range=getattr(config, "initializer_range", 0.02),
        device=device,
        dtype=dtype,
    )
    text_model.rnn_state_tuning_ple = ple
    object.__setattr__(ple, "_embedding_ref", weakref.ref(text_model.embed_tokens))
    ple._model_hook_handle = model.register_forward_pre_hook(ple.capture_model_inputs, with_kwargs=True)

    branches = []
    for layer_idx, decoder_layer in enumerate(text_model.layers):
        base_decoder_layer = _base_decoder_layer(decoder_layer)
        layer_device = _module_device(base_decoder_layer)
        branch = Qwen35PLEBranch(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            ple_dim=ple_dim,
            hidden_activation=getattr(config, "hidden_act", "silu"),
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=getattr(config, "initializer_range", 0.02),
            norm_position=norm_position,
            device=layer_device,
            dtype=dtype,
        )
        text_model.layers[layer_idx] = Qwen35PLEDecoderLayer(base_decoder_layer, branch, ple)
        branches.append(branch)
    return ple, branches
