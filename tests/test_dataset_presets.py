from rnn_state_tuning.dataset_presets import DATASET_PRESETS, NOVEL_AGENT_REPO, NOVEL_AGENT_REVISION


def test_novel_agent_presets_are_revision_pinned():
    assert {"novel-continuation", "novel-instruction", "novel-scene"} <= set(DATASET_PRESETS)
    assert all(preset.repo_id == NOVEL_AGENT_REPO for preset in DATASET_PRESETS.values())
    assert len(NOVEL_AGENT_REVISION) == 40
    assert all(preset.filename.endswith(".jsonl") for preset in DATASET_PRESETS.values())
