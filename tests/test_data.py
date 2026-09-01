import torch
import json

from rnn_state_tuning.data import RightPaddingCollator, encode_record


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(character) for character in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) for character in text[: kwargs["max_length"]]]}


class EncodingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": super().apply_chat_template(messages, tokenize, add_generation_prompt)}


def test_messages_mask_non_assistant_tokens():
    tokenizer = FakeTokenizer()
    record = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    encoded = encode_record(tokenizer, record, max_length=200)
    first_label = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
    expected_start = len(tokenizer.apply_chat_template(record["messages"][:1], True, True))
    assert first_label == expected_start


def test_chat_batch_encoding_is_normalized_to_input_ids():
    tokenizer = EncodingTokenizer()
    encoded = encode_record(
        tokenizer,
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        max_length=200,
    )
    assert isinstance(encoded["input_ids"], list)
    assert any(label != -100 for label in encoded["labels"])


def test_long_chat_keeps_assistant_tail():
    tokenizer = FakeTokenizer()
    encoded = encode_record(
        tokenizer,
        {
            "messages": [
                {"role": "user", "content": "q" * 100},
                {"role": "assistant", "content": "answer"},
            ]
        },
        max_length=30,
    )
    assert len(encoded["input_ids"]) == 30
    assert any(label != -100 for label in encoded["labels"])


def test_collator_right_pads_inputs_and_masks_labels():
    collator = RightPaddingCollator(pad_token_id=0, pad_to_multiple_of=None)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4], "labels": [4]},
        ]
    )
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3], [4, 0, 0]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, 2, 3], [4, -100, -100]]))


def test_jsonl_dataset_uses_deterministic_lazy_subset(tmp_path):
    from rnn_state_tuning.data import StateTuningDataset

    path = tmp_path / "data.jsonl"
    rows = [{"text": f"row-{index}"} for index in range(10)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    tokenizer = FakeTokenizer()
    first = StateTuningDataset(path, tokenizer, 100, max_examples=4, sample_seed=7)
    second = StateTuningDataset(path, tokenizer, 100, max_examples=4, sample_seed=7)
    assert len(first) == 4
    assert first.offsets == second.offsets
