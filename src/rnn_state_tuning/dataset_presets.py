from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


NOVEL_AGENT_REPO = "mikuhhn1239/novel-agent-sft-dataset"
NOVEL_AGENT_REVISION = "5d3040d21f51b3ce90b9396b058e552c47f43cd5"


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    repo_id: str
    revision: str
    filename: str
    purpose: str


DATASET_PRESETS = {
    preset.name: preset
    for preset in (
        DatasetPreset(
            name="novel-continuation",
            repo_id=NOVEL_AGENT_REPO,
            revision=NOVEL_AGENT_REVISION,
            filename="training/base-sft/continuation.jsonl",
            purpose="72K Chinese novel context-to-continuation examples",
        ),
        DatasetPreset(
            name="novel-instruction",
            repo_id=NOVEL_AGENT_REPO,
            revision=NOVEL_AGENT_REVISION,
            filename="training/base-sft/instruction.jsonl",
            purpose="72K Chinese topic-to-complete-text examples",
        ),
        DatasetPreset(
            name="novel-attribution",
            repo_id=NOVEL_AGENT_REPO,
            revision=NOVEL_AGENT_REVISION,
            filename="training/v3.2-attribution-best-candidate/train.jsonl",
            purpose="speaker attribution with JSON answers",
        ),
        DatasetPreset(
            name="novel-narrative",
            repo_id=NOVEL_AGENT_REPO,
            revision=NOVEL_AGENT_REVISION,
            filename="training/v3.2-narrative-type-classification/train.jsonl",
            purpose="narrative-unit classification with JSON answers",
        ),
        DatasetPreset(
            name="novel-scene",
            repo_id=NOVEL_AGENT_REPO,
            revision=NOVEL_AGENT_REVISION,
            filename="training/v4-scene-boundary-detection/train.jsonl",
            purpose="scene-boundary detection with JSON answers",
        ),
    )
}


def get_dataset_preset(name: str) -> DatasetPreset:
    try:
        return DATASET_PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset preset {name!r}") from error


def download_dataset_preset(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    endpoint: str | None = None,
) -> tuple[Path, DatasetPreset]:
    preset = get_dataset_preset(name)
    path = hf_hub_download(
        repo_id=preset.repo_id,
        filename=preset.filename,
        repo_type="dataset",
        revision=preset.revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        endpoint=endpoint,
    )
    return Path(path), preset
