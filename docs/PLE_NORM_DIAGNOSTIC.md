# PLE RMSNorm Placement Diagnostic

This diagnostic isolates native Gemma post-projection RMSNorm from a retrofit-friendly pre-projection RMSNorm. Both Qwen3.5-9B PLE-only runs use `ple_dim=32`, FP32 PLE parameters and optimizer states, seed `42`, the explicitly seeded shuffled dataloader, 1,804 `novel-scene` rows, effective global batch `16`, learning rate `1e-4`, and 25 updates. The only topology difference is `--ple-norm-position post` versus `pre`.

## Signal behavior

| Topology | Step | Mean projection RMS | Mean final delta RMS | Mean amplification | Maximum delta RMS |
|---|---:|---:|---:|---:|---:|
| Native post | 2 | 0.000102 | 0.097540 | 979x | 0.349115 |
| Native post | 5 | 0.000343 | 0.253127 | 839x | 0.710404 |
| Native post | 10 | 0.000753 | 0.356322 | 668x | 0.928283 |
| Retrofit pre | 2 | 0.000540 | 0.000540 | 1x | 0.000607 |
| Retrofit pre | 5 | 0.001579 | 0.001579 | 1x | 0.001874 |
| Retrofit pre | 10 | 0.002485 | 0.002485 | 1x | 0.003099 |

The native post topology amplifies very small zero-initialized projection outputs by hundreds of times. Moving RMSNorm before the projection makes the residual grow linearly with the projection.

## Training

| Topology | Step 5 loss | Step 10 loss | Step 15 loss | Step 20 loss | Step 25 loss |
|---|---:|---:|---:|---:|---:|
| Native post | 2.892240 | 0.851280 | 0.517547 | 0.331511 | 0.276948 |
| Retrofit pre | 1.115543 | 0.321321 | 0.296795 | 0.247946 | 0.194719 |

## Held-out benchmark

Generation metrics use all 149 official test rows rather than the earlier first-20 subset.

| Topology | Response loss | Perplexity | Exact match | Boundary F1 | Unique predictions | Invalid JSON |
|---|---:|---:|---:|---:|---:|---:|
| Native post | 0.322644 | 1.380774 | 14.09% | 0.203947 | 1 | 0 |
| Retrofit pre | 0.236931 | 1.267354 | 26.17% | 0.308901 | 44 | 2 |

The native topology predicts `[1]` for every test row. The retrofit topology improves loss, exact match, boundary F1, and output diversity after only 25 updates. This supports pre-projection RMSNorm for zero-initialized retrofit PLE; it does not claim that native Gemma PLE is defective in its original full-pretraining regime.

Artifacts:

- Native adapter: `outputs/diagnostics/qwen35-ple32-fp32-post-novel-scene-s25/`
- Retrofit adapter: `outputs/diagnostics/qwen35-ple32-fp32-pre-novel-scene-s25/`
- Native RMS log: `outputs/diagnostics/logs/ple32-fp32-post-s25.log`
- Retrofit RMS log: `outputs/diagnostics/logs/ple32-fp32-pre-s25.log`
- Native benchmark: `outputs/benchmarks/ple32-fp32-post-s25-scene-all149.json`
- Retrofit benchmark: `outputs/benchmarks/ple32-fp32-pre-s25-scene-all149.json`
- Machine-readable comparison: `outputs/benchmarks/ple-norm-position-diagnostic.json`
