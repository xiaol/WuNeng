from __future__ import annotations

import argparse
import os

from rnn_state_tuning.dataset_presets import DATASET_PRESETS, download_dataset_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pinned RNN-StateTuning dataset preset")
    parser.add_argument("preset", choices=sorted(DATASET_PRESETS))
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path, preset = download_dataset_preset(
        args.preset,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        endpoint=args.hf_endpoint,
    )
    print(f"preset={preset.name}")
    print(f"repo={preset.repo_id}")
    print(f"revision={preset.revision}")
    print(f"file={preset.filename}")
    print(f"path={path}")


if __name__ == "__main__":
    main()
