# PLE and State-Tuning Benchmark

All adaptation arms were trained after the external GPU job released the four A100s. The runs use the same Qwen3.5-9B checkpoint, Novel Agent `novel-scene` training file, seed `42`, 1,804 training rows, 512-token cap, effective global batch `16`, AdamW learning rate `1e-4`, and 113 optimizer updates per stage. PLE uses faithful Gemma-style `ple_dim=256`; state tuning changes only Qwen GDN initial states. The sequential arm first trains PLE, then reloads and freezes it while training fresh zero-initialized states.

Evaluation uses the official 149-row scene-boundary test file. Response loss and perplexity use all 149 rows. Greedy generation uses a fixed first-20-row subset with `enable_thinking=False`, 512-token prompt truncation from the right (preserving the assistant prompt), and 64 new tokens. A prediction is valid only when it contains JSON with an integer `boundaries` list.

These original runs predate the explicit dataloader generator. Although each used seed `42`, PLE initialization consumed the global RNG before dataloader iteration, so adapter methods did not receive identical shuffled example orders. The later controlled PLE norm-position diagnostic fixes this protocol issue.

| Model | Trainable parameters | Response loss | Perplexity | Exact match (20) | Boundary F1 (20) | Valid JSON |
|---|---:|---:|---:|---:|---:|---:|
| Frozen base — Qwen3.5-9B (about 9B parameters, all frozen) | 0 | 1.21704 | 3.37718 | 0% | 0.00000 | 0% |
| PLE `ple_dim=256` | 2,135,032,064 | 0.31828 | 1.37476 | 20% | 0.34146 | 100% |
| GDN state-only | 12,582,912 | 0.22401 | 1.25108 | 40% | 0.54054 | 100% |
| Joint PLE + GDN state | 2,147,614,976 | 0.31658 | 1.37243 | 25% | 0.34146 | 100% |
| Sequential PLE → GDN state | 2,135,032,064 → 12,582,912 | 0.31631 | 1.37206 | 25% | 0.35000 | 100% |

On this short structured-task pilot, state-only is better than PLE, joint training, and sequential training on every reported quality metric while using about 170 times fewer trainable parameters than PLE. Sequential training is slightly better than joint training on response loss and boundary F1, but neither combined method surpasses state-only. This is one seed and one held-out task; it supports using state tuning as the next default, not a broad claim that PLE is inferior for every task.

Artifacts:

- PLE adapter: `outputs/qwen35-ple256-novel-scene-e1/`
- State adapter: `outputs/qwen35-state-novel-scene-e1/`
- Joint adapter: `outputs/qwen35-state-ple256-novel-scene-e1/`
- Sequential adapter: `outputs/qwen35-ple256-then-state-novel-scene-e1/`
- Machine-readable summary: `outputs/benchmarks/comparison.json`
- Per-run benchmark JSON: `outputs/benchmarks/base-scene-e1.json`, `outputs/benchmarks/ple256-scene-e1.json`, `outputs/benchmarks/state-scene-e1.json`, `outputs/benchmarks/state-ple256-scene-e1.json`, and `outputs/benchmarks/ple256-then-state-scene-e1.json`

## Corrected pre-norm rerun

The original PLE and sequential rows above used the native post-projection RMSNorm topology. Because zero-initialized PLE projections can be amplified by that norm, a matched rerun uses the retrofit `pre` topology (RMSNorm before the zero-initialized down projection) and BF16 PLE weights so the full `ple_dim=256` adapter fits the four-GPU setup. The same 1,804 examples, 113 updates, seed, optimizer, and evaluation protocol are used.

| Model | Trainable parameters | Response loss | Perplexity | Exact match (20) | Boundary F1 (20) | Valid JSON |
|---|---:|---:|---:|---:|---:|---:|
| PLE `ple_dim=256`, pre-norm, BF16 | 2,134,909,184 | 0.21197 | 1.23611 | 50% | 0.54054 | 100% |
| Sequential pre-norm PLE → GDN state | 2,134,909,184 → 12,582,912 | 0.21088 | 1.23477 | 50% | 0.44444 | 100% |

The sequential stage lowers response loss slightly, but does not improve exact match and has lower boundary F1 than PLE alone on this 20-row generation subset. These results are one seed on one held-out task; the topology and dtype are recorded in each adapter config.

Additional artifacts:

- Pre-norm PLE adapter: `outputs/qwen35-ple256-bf16-pre-novel-scene-e1/`
- Sequential pre-norm adapter: `outputs/qwen35-ple256-bf16-pre-then-state-novel-scene-e1/`
- Per-run benchmark JSON: `outputs/benchmarks/ple256-bf16-pre-scene-e1.json` and `outputs/benchmarks/ple256-bf16-pre-then-state-scene-e1.json`
