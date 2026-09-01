from rnn_state_tuning.cli.benchmark import _boundaries


def test_boundary_parser_accepts_wrapped_json():
    assert _boundaries('```json\n{"boundaries": [4, 1, 4]}\n```') == [1, 4]


def test_boundary_parser_rejects_invalid_outputs():
    assert _boundaries("not json") is None
    assert _boundaries('{"boundaries": ["1"]}') is None
