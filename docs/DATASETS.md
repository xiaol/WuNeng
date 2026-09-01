# Dataset strategy

## Source

The Novel Agent presets are pinned to:

- Repository: `mikuhhn1239/novel-agent-sft-dataset`
- Revision: `5d3040d21f51b3ce90b9396b058e552c47f43cd5`
- Upstream project: `lin1753/novel-agent`

This is the same publisher dataset used by the neighboring multi-state RWKV research project. The state-tuning project consumes the original system/user/assistant rows, not that project's derived online-memory episodes.

## Recommended sequence

1. Train `novel-scene` as a short optimization and formatting sanity check.
2. Compare zero-state and tuned-state loss on a held-out scene split.
3. Train `novel-continuation` on a deterministic 5K sample.
4. Expand to all 72K continuation rows only if the pilot improves held-out continuation loss and generation quality.
5. Train attribution and narrative adapters separately if those capabilities are desired.

The scene training file was locally verified against the upstream revision:

- Rows: 1,804
- Bytes: 3,563,946
- SHA-256: `785fe54c0a4e5c64e33f64f9bc88d64719576407c21eb0d520f9dec5a59b8e22`
- Qwen3.5 token length at a 1,024 cap: mean 470.9, maximum 1,024
- Rows requiring prompt-tail truncation: 11
- Rows retaining assistant supervision: 1,804 of 1,804

## Why not the derived 41K episodes?

Those episodes split novel completions into write/read interactions for trainable online memory. Direct initial-state tuning has no per-example write phase: it learns one shared task prior and then follows Qwen's normal recurrence. Training it on the original chat rows is the faithful objective.

The multi-state report also records that its first 41,266-episode run reduced training loss but regressed the structured-task transfer benchmark. That result does not disqualify this different adapter, but it argues for a held-out pilot before full-corpus training.
