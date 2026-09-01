import pytest
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@pytest.fixture
def tiny_qwen35():
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 0, 0]},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen3_5ForCausalLM(config)
