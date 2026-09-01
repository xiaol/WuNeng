import copy

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from rnn_state_tuning import (
    load_adapter,
    ple_parameters,
    prepare_model_for_tuning,
    save_adapter,
    set_trainable_components,
    state_tuning_parameters,
    tuning_parameters,
)


@pytest.mark.parametrize(("norm_position", "parameter_count"), [("post", 2632), ("pre", 2584)])
def test_zero_initialized_ple_preserves_logits(tiny_qwen35, norm_position, parameter_count):
    input_ids = torch.randint(3, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        baseline = tiny_qwen35(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    info = prepare_model_for_tuning(
        tiny_qwen35,
        method="ple",
        ple_dim=8,
        ple_norm_position=norm_position,
    )
    with torch.no_grad():
        tuned = tiny_qwen35(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    assert info.method == "ple"
    assert info.state_layer_count == 0
    assert info.ple_layer_count == 2
    assert info.parameter_count == parameter_count
    assert info.ple_norm_position == norm_position
    torch.testing.assert_close(tuned, baseline, rtol=0, atol=0)
    assert [parameter for parameter in tiny_qwen35.parameters() if parameter.requires_grad] == list(
        tuning_parameters(tiny_qwen35)
    )


def test_ple_output_projections_get_finite_gradients(tiny_qwen35):
    prepare_model_for_tuning(tiny_qwen35, method="ple", ple_dim=8)
    input_ids = torch.randint(3, 64, (2, 7))
    outputs = tiny_qwen35(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        use_cache=False,
    )
    outputs.loss.backward()

    gradients = [parameter.grad for parameter in ple_parameters(tiny_qwen35)]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    projection_gradients = [
        module.projection.weight.grad
        for module in tiny_qwen35.modules()
        if module.__class__.__name__ == "Qwen35PLEBranch"
    ]
    assert sum(gradient.abs().sum() for gradient in projection_gradients) > 0


@pytest.mark.parametrize("norm_position", ["post", "pre"])
def test_ple_records_per_layer_rms(norm_position, tiny_qwen35):
    prepare_model_for_tuning(
        tiny_qwen35,
        method="ple",
        ple_dim=8,
        ple_norm_position=norm_position,
    )
    branches = [
        module for module in tiny_qwen35.modules() if module.__class__.__name__ == "Qwen35PLEBranch"
    ]
    for branch in branches:
        branch.record_rms = True

    input_ids = torch.randint(3, 64, (2, 7))
    tiny_qwen35(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)

    for branch in branches:
        assert set(branch.last_rms) == {"residual", "feature", "projection", "delta"}
        assert all(torch.isfinite(value) for value in branch.last_rms.values())
        assert branch.last_rms["projection"] == 0
        assert branch.last_rms["delta"] == 0


def test_ple_gradient_checkpointing_recomputes_with_current_inputs(tiny_qwen35):
    prepare_model_for_tuning(tiny_qwen35, method="ple", ple_dim=8)
    tiny_qwen35.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    input_ids = torch.randint(3, 64, (2, 7))
    loss = tiny_qwen35(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        use_cache=False,
    ).loss
    loss.backward()

    assert all(parameter.grad is not None for parameter in ple_parameters(tiny_qwen35))


def test_state_and_ple_both_get_gradients(tiny_qwen35):
    prepare_model_for_tuning(tiny_qwen35, method="state+ple", ple_dim=8)
    input_ids = torch.randint(3, 64, (2, 7))
    outputs = tiny_qwen35(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        use_cache=False,
    )
    outputs.loss.backward()

    state_gradient = sum(parameter.grad.abs().sum() for parameter in state_tuning_parameters(tiny_qwen35))
    ple_gradient = sum(
        module.projection.weight.grad.abs().sum()
        for module in tiny_qwen35.modules()
        if module.__class__.__name__ == "Qwen35PLEBranch"
    )
    assert state_gradient > 0
    assert ple_gradient > 0


def test_ple_generation_updates_inputs_for_cached_tokens(tiny_qwen35):
    prepare_model_for_tuning(tiny_qwen35, method="ple", ple_dim=8)
    tiny_qwen35.config.use_cache = True
    input_ids = torch.randint(3, 64, (1, 5))

    with torch.no_grad():
        generated = tiny_qwen35.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)

    assert generated.shape == (1, 7)
    ple = next(module for module in tiny_qwen35.modules() if module.__class__.__name__ == "Qwen35PerLayerEmbeddings")
    assert ple.current_inputs.shape == (1, 1, 2, 8)


def test_state_and_ple_checkpoint_roundtrip_auto_attaches(tmp_path, tiny_qwen35):
    clean_model = copy.deepcopy(tiny_qwen35)
    prepare_model_for_tuning(tiny_qwen35, method="state+ple", ple_dim=8, ple_norm_position="pre")
    with torch.no_grad():
        for parameter in tuning_parameters(tiny_qwen35):
            parameter.normal_()
    expected = [parameter.detach().clone() for parameter in tuning_parameters(tiny_qwen35)]

    save_adapter(tiny_qwen35, tmp_path, base_model="tiny")
    config = load_adapter(clean_model, tmp_path)

    assert config["method"] == "direct_recurrent_state+per_layer_embeddings"
    assert config["ple"]["dim"] == 8
    assert config["ple"]["norm_position"] == "pre"
    actual = list(tuning_parameters(clean_model))
    assert len(actual) == len(expected)
    for actual_parameter, expected_parameter in zip(actual, expected):
        torch.testing.assert_close(actual_parameter, expected_parameter)


def test_sequential_ple_then_state_loading(tmp_path, tiny_qwen35):
    ple_model = copy.deepcopy(tiny_qwen35)
    prepare_model_for_tuning(ple_model, method="ple", ple_dim=8)
    with torch.no_grad():
        for parameter in ple_parameters(ple_model):
            parameter.normal_()
    expected_ple = [parameter.detach().clone() for parameter in ple_parameters(ple_model)]
    save_adapter(ple_model, tmp_path, base_model="tiny")

    prepare_model_for_tuning(tiny_qwen35, method="state+ple", ple_dim=8)
    load_adapter(tiny_qwen35, tmp_path, strict=False, prepare=False)
    set_trainable_components(tiny_qwen35, "state")

    for actual, expected in zip(ple_parameters(tiny_qwen35), expected_ple):
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad
    assert all(parameter.requires_grad for parameter in state_tuning_parameters(tiny_qwen35))
    assert all(torch.count_nonzero(parameter) == 0 for parameter in state_tuning_parameters(tiny_qwen35))


def test_explicit_shuffle_generator_is_independent_of_ple_initialization(tiny_qwen35):
    orders = []
    for method in ("state", "state+ple"):
        set_seed(42)
        model = copy.deepcopy(tiny_qwen35)
        prepare_model_for_tuning(model, method=method, ple_dim=8)
        generator = torch.Generator().manual_seed(42)
        order = [
            int(value)
            for batch in DataLoader(range(20), batch_size=1, shuffle=True, generator=generator)
            for value in batch
        ]
        orders.append(order)

    assert orders[0] == orders[1]
