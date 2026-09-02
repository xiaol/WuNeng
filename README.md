# RNN-StateTuning

RNN-StateTuning adapts a frozen Qwen3.5 model with trainable recurrent initial states, genuine Gemma-style Per-Layer Embeddings (PLE), or both. The state backend targets the 24 Gated DeltaNet (GDN) layers in Qwen3.5-9B.

The implementation starts every sequence from a learned state and then leaves Qwen's pretrained recurrence unchanged:

\[
S_t = e^{g_t}S_{t-1} + k_t\left(\beta_t(v_t-k_t^T e^{g_t}S_{t-1})\right)^T,
\qquad S_{t=0}=S_0^{(l)}.
\]

Each Qwen3.5-9B GDN layer receives one state with shape `32 x 128 x 128`. Across 24 GDN layers this is 12,582,912 trainable parameters: 24 MiB when saved as BF16 or 48 MiB in the default FP32 training representation.

## Status

- Implemented: direct recurrent-state tuning for Qwen3.5 text and conditional-generation checkpoints.
- Implemented: Gemma 3n/4-style token-plus-context PLE for every Qwen3.5 decoder layer.
- Implemented: controlled `state`, `ple`, and `state+ple` training methods.
- Implemented: response-only chat loss, right-padded batching, distributed training, adapter-only Safetensors checkpoints, and cached generation.
- Implemented: revision-pinned Hugging Face presets for the Novel Agent SFT dataset.
- Planned: rank-factorized states, causal-convolution history tuning, and RWKV backends.

See [docs/PLE.md](docs/PLE.md) for the PLE equations, placement, open reference implementations, and memory requirements. The matched native-post versus retrofit-pre experiment is recorded in [docs/PLE_NORM_DIAGNOSTIC.md](docs/PLE_NORM_DIAGNOSTIC.md).

## Scene-boundary benchmark

The main pilot uses Qwen3.5-9B with all base-model parameters frozen. Adapters are trained on the 1,804-row Novel Agent scene-boundary training split. Response loss and greedy-generation metrics below use all 149 held-out test scenes; higher exact match and Boundary F1 are better, while lower loss and perplexity are better.

| Model | Parameters optimized in stage | Response loss ↓ | Perplexity ↓ | Exact match ↑ | Boundary F1 ↑ | Valid JSON |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Qwen3.5-9B (about 9B parameters, all frozen) | 0 | 1.21704 | 3.37718 | 0.00% | 0.00000 | 0% |
| State-only, `1e-4` | 12,582,912 | 0.22401 | 1.25108 | 34.23% | 0.33117 | 100% |
| Pre-norm BF16 PLE-only | 2,134,909,184 | 0.21197 | 1.23611 | 36.24% | 0.32857 | 100% |
| PLE → state, `1e-4` | 12,582,912; PLE frozen | 0.21088 | 1.23477 | **40.94%** | **0.36232** | 100% |
| PLE → state, `3e-4` | 12,582,912; PLE frozen | **0.20885** | **1.23225** | 38.26% | 0.35336 | 100% |
| PLE → state, `1e-3` | 12,582,912; PLE frozen | 0.22273 | 1.24948 | 34.90% | 0.30435 | 100% |
| PLE → state, `3e-3` | 12,582,912; PLE frozen | 0.21806 | 1.24367 | 37.58% | 0.30657 | 100% |

Sequential state tuning does improve the PLE-only adapter on the complete test set: the `1e-4` stage raises exact match from 36.24% to 40.94% and Boundary F1 from 0.32857 to 0.36232. Increasing the state learning rate further is not consistently beneficial. The earlier 20-row generation screen was noisy—its 50% exact-match figures should not be treated as the full benchmark.

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for the original post-norm runs, the PLE norm correction, learning-rate sweep, protocol, and artifact paths.

## Install

Use an environment with a Qwen3.5-capable Transformers release:

```bash
cd /root/x/RNN-StateTuning
pip install -e .
```

The current local checkpoint is `/root/x/Qwen3.5-9B`.

## Data

Training data is JSONL. Chat records calculate loss only on assistant spans:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

The shorter prompt/response form is equivalent:

```json
{"prompt":"Question","response":"Answer"}
```

Records containing only `text` use full language-model loss. Batches always use right padding because left padding would decay the learned recurrent state before the first real token.

### Novel Agent presets

The presets use `mikuhhn1239/novel-agent-sft-dataset` at immutable revision `5d3040d21f51b3ce90b9396b058e552c47f43cd5`. Only the selected JSONL file is downloaded.

| Preset | Purpose |
|---|---|
| `novel-scene` | Small, measurable scene-boundary pilot; recommended first |
| `novel-attribution` | Speaker attribution task adapter |
| `novel-narrative` | Narrative-unit classification task adapter |
| `novel-continuation` | 72K context-to-continuation rows; recommended for novel-writing adaptation |
| `novel-instruction` | 72K topic-to-complete-text rows |

Download or inspect a preset without starting training:

```bash
HF_ENDPOINT=https://hf-mirror.com rnn-state-dataset novel-scene \
  --cache-dir /root/x/.cache/huggingface
```

Train separate adapters for the structured tasks. Mixing their incompatible JSON schemas into one static initial state makes the experiment harder to interpret.

## Train Qwen3.5-9B

A single A100 40 GB can hold the frozen BF16 model. Four-GPU DDP gives a larger effective batch without sharding the adapter:

```bash
accelerate launch --num_processes 4 \
  -m rnn_state_tuning.cli.train \
  --model /root/x/Qwen3.5-9B \
  --train-data /path/to/train.jsonl \
  --output-dir outputs/qwen35-state \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 1024 \
  --learning-rate 1e-4 \
  --dtype bfloat16 \
  --mixed-precision bf16 \
  --local-files-only
```

For the recommended Novel Agent scene pilot:

```bash
HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 4 \
  -m rnn_state_tuning.cli.train \
  --model /root/x/Qwen3.5-9B \
  --dataset-preset novel-scene \
  --dataset-cache-dir /root/x/.cache/huggingface \
  --output-dir outputs/qwen35-novel-scene \
  --max-length 1024 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --dtype bfloat16 \
  --mixed-precision bf16 \
  --local-files-only
```

To train faithful PLE instead of recurrent states:

```bash
accelerate launch --num_processes 4 \
  -m rnn_state_tuning.cli.train \
  --model /root/x/Qwen3.5-9B \
  --method ple \
  --ple-dim 256 \
  --dataset-preset novel-scene \
  --output-dir outputs/qwen35-ple \
  --dtype bfloat16 \
  --mixed-precision bf16 \
  --local-files-only
```

Native Gemma ordering is the default (`--ple-norm-position post`). Use `--ple-norm-position pre` for the
zero-projection retrofit experiment, and `--ple-rms-log-steps 0,1,2,5,10` for early per-layer signal diagnostics.

Use `--method state+ple` to optimize both adapter families. Faithful `ple_dim=256` adds 2,135,032,064 parameters on Qwen3.5-9B, so replicated Adam is not expected to fit the same setup as state-only tuning; use optimizer and parameter sharding.

For a cheaper continuation pilot, sample 5,000 rows deterministically before committing to all 72K:

```bash
HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 4 \
  -m rnn_state_tuning.cli.train \
  --model /root/x/Qwen3.5-9B \
  --dataset-preset novel-continuation \
  --dataset-max-examples 5000 \
  --dataset-seed 42 \
  --dataset-cache-dir /root/x/.cache/huggingface \
  --output-dir outputs/qwen35-novel-continuation-5k \
  --max-length 1024 \
  --learning-rate 1e-4 \
  --local-files-only
```

Because published RWKV recipes disagree substantially on learning rate, run short pilots at `1e-5`, `1e-4`, and `1e-3` before a long job. Defaults are AdamW, zero weight decay, cosine decay, 3% warmup, gradient clipping at 1.0, and non-reentrant gradient checkpointing.

Only two files are saved:

```text
outputs/qwen35-state/
├── adapter_config.json
└── adapter_model.safetensors
```

The base Qwen weights are never copied into the adapter checkpoint.

## Generate

```bash
rnn-state-generate \
  --model /root/x/Qwen3.5-9B \
  --adapter outputs/qwen35-state \
  --prompt "Explain recurrent state tuning." \
  --local-files-only
```

During prefill, each GDN layer starts from its learned state. The resulting recurrent state is then stored in the normal Transformers cache and used for token-by-token decoding.

## Python API

```python
import torch
from transformers import Qwen3_5ForConditionalGeneration
from rnn_state_tuning import load_adapter, prepare_model_for_state_tuning

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "/root/x/Qwen3.5-9B",
    dtype=torch.bfloat16,
)
info = prepare_model_for_state_tuning(model)
load_adapter(model, "outputs/qwen35-state")
print(info.parameter_count)
```

## Design Guarantees

- Zero recurrent states and zero PLE output projections exactly preserve base-model logits before training.
- The same task state is expanded across a batch; samples never receive independent parameters.
- Every example resets to the learned initial state instead of carrying state across unrelated samples.
- Training uses `use_cache=False`, avoiding in-place cache writes in the autograd path.
- Generation uses the standard Qwen cache after the state-tuned prefill.
- Adapter loading validates every layer index and state shape.

## Tests

```bash
pytest -q
```

The tests cover zero-init equivalence, state and PLE gradients, gradient checkpointing, generation caches, response masking, right padding, and checkpoint round trips.
