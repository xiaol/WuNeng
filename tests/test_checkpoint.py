import json

import torch

from rnn_state_tuning import load_adapter, prepare_model_for_state_tuning, save_adapter, state_tuning_parameters


def test_adapter_roundtrip(tmp_path, tiny_qwen35):
    prepare_model_for_state_tuning(tiny_qwen35)
    expected = []
    with torch.no_grad():
        for parameter in state_tuning_parameters(tiny_qwen35):
            parameter.normal_()
            expected.append(parameter.clone())

    save_adapter(tiny_qwen35, tmp_path, base_model="tiny")
    with torch.no_grad():
        for parameter in state_tuning_parameters(tiny_qwen35):
            parameter.zero_()
    config = load_adapter(tiny_qwen35, tmp_path)

    assert config["method"] == "direct_recurrent_state"
    assert json.loads((tmp_path / "adapter_config.json").read_text())["base_model"] == "tiny"
    for actual, saved in zip(state_tuning_parameters(tiny_qwen35), expected):
        torch.testing.assert_close(actual, saved)
