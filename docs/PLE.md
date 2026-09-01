# Per-Layer Embeddings for Qwen3.5

This project implements the Per-Layer Embeddings (PLE) architecture used by the Gemma 3n family and the newer Gemma4 implementation. It is PLE itself, not a sparse FFN or a low-rank approximation.

## Open references

- Hugging Face Transformers provides open-source `Gemma3nTextModel` and `Gemma3nTextDecoderLayer` implementations in `transformers.models.gemma3n`.
- Hugging Face Transformers also provides the same explicit PLE pipeline in `transformers.models.gemma4`.
- `google/gemma-3n-E2B-it` and `google/gemma-3n-E4B-it` are open-weight checkpoints under the Gemma license and contain trained PLE tensors.

The Gemma checkpoints are architecture references and experiment baselines. Their PLE tensors cannot be copied directly into Qwen because the vocabulary, hidden size, layer count, and learned residual stream differ.

## Architecture

For token IDs `t` and frozen Qwen input embeddings `e`, a shared PLE module computes a token component and a contextual component:

```text
token = Embedding(vocab, layers * ple_dim)(t) * sqrt(ple_dim)
context = RMSNorm(Linear(hidden, layers * ple_dim)(e) / sqrt(hidden))
per_layer = (token + context) / sqrt(2)
```

Both packed tensors are reshaped to `[batch, sequence, layers, ple_dim]`. Decoder layer `l` receives its own slice and adds a gated residual after Qwen's normal mixer and MLP residuals:

```text
qwen_output = QwenDecoderLayer(hidden)
ple_output = RMSNorm(Linear(activation(Linear(qwen_output)) * per_layer[l]))
next_hidden = qwen_output + ple_output
```

Each PLE output projection starts at zero. Attaching an adapter therefore preserves the base logits exactly while still giving the output projections a first-step gradient.

`--ple-norm-position post` keeps the native Gemma ordering shown above. For retrofit experiments,
`--ple-norm-position pre` applies RMSNorm to the gated `ple_dim` features before the zero-initialized output
projection, so the added residual grows proportionally with that projection. Checkpoints record the selected topology.

Use `--ple-rms-log-steps 0,1,2,5,10` to print each layer's residual, gated-feature, projected, and final PLE RMS at selected completed-step counts. Training uses an explicitly seeded dataloader generator, so adapter initialization cannot change the shuffled example order.

See [PLE RMSNorm Placement Diagnostic](PLE_NORM_DIAGNOSTIC.md) for the matched FP32 experiment comparing both topologies.

## Interaction with Qwen state

Qwen's attention/GDN value computation and recurrent cache stay inside the frozen decoder layer. PLE is added only after that layer has completed both of its existing residual updates:

```text
input -> attention or GDN residual -> MLP residual -> PLE residual -> next layer
```

Consequently, an inter-layer residual stream does not make PLE difficult to implement and no value tensors need to be intercepted. In `state+ple` mode, the learned GDN initial state affects the normal Qwen path and PLE adapts its completed layer output. Cached generation recomputes PLE for the current token while the standard Qwen cache carries attention and recurrent state forward.

## Qwen3.5-9B size

With 32 decoder layers, hidden size 4096, vocabulary size 248,320, and `ple_dim=256`:

| Component | Parameters |
|---|---:|
| Packed token table | 2,034,237,440 |
| Context projection | 33,554,432 |
| Per-layer gates and projections | 67,108,864 |
| RMSNorm weights | 131,328 |
| Total PLE | 2,135,032,064 |

The PLE weights alone occupy about 3.98 GiB in BF16. Gradients and ordinary Adam moments increase this substantially, so use FSDP or ZeRO-style parameter and optimizer sharding for the 9B experiment. State-only tuning remains the inexpensive 12,582,912-parameter baseline.

## Methods

```text
--method state       train only GDN initial states
--method ple         train only PLE
--method state+ple   train both
```

All original Qwen attention, GDN, MLP, token-embedding, normalization, and LM-head weights remain frozen in every method.
