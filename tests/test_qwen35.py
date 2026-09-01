import torch

from rnn_state_tuning import prepare_model_for_state_tuning, state_tuning_parameters


def test_zero_state_preserves_logits_and_only_state_trains(tiny_qwen35):
    input_ids = torch.randint(3, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        baseline = tiny_qwen35(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    info = prepare_model_for_state_tuning(tiny_qwen35)
    with torch.no_grad():
        tuned = tiny_qwen35(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    assert info.layer_count == 1
    assert info.parameter_count == 4 * 8 * 8
    torch.testing.assert_close(tuned, baseline, rtol=0, atol=0)
    assert [parameter for parameter in tiny_qwen35.parameters() if parameter.requires_grad] == list(
        state_tuning_parameters(tiny_qwen35)
    )


def test_initial_state_gets_finite_gradient(tiny_qwen35):
    prepare_model_for_state_tuning(tiny_qwen35)
    input_ids = torch.randint(3, 64, (2, 7))
    outputs = tiny_qwen35(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        use_cache=False,
    )
    outputs.loss.backward()

    gradients = [parameter.grad for parameter in state_tuning_parameters(tiny_qwen35)]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_prefill_writes_seeded_state_to_generation_cache(tiny_qwen35):
    prepare_model_for_state_tuning(tiny_qwen35)
    tiny_qwen35.config.use_cache = True
    input_ids = torch.randint(3, 64, (1, 5))
    with torch.no_grad():
        outputs = tiny_qwen35(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)

    recurrent_state = outputs.past_key_values.layers[0].recurrent_states[0]
    assert recurrent_state.shape == (1, 4, 8, 8)
    assert torch.isfinite(recurrent_state).all()
