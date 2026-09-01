from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler, set_seed

from rnn_state_tuning import (
    iter_ple_branches,
    load_adapter,
    prepare_model_for_tuning,
    save_adapter,
    set_trainable_components,
    tuning_parameters,
)
from rnn_state_tuning.data import RightPaddingCollator, StateTuningDataset
from rnn_state_tuning.dataset_presets import DATASET_PRESETS, download_dataset_preset
from rnn_state_tuning.modeling import load_qwen35_model, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Qwen3.5 GDN states and/or Gemma-style PLE")
    parser.add_argument("--model", required=True, help="Qwen3.5 model ID or local checkpoint")
    parser.add_argument("--method", choices=["state", "ple", "state+ple"], default="state")
    parser.add_argument("--ple-dim", type=int, default=256)
    parser.add_argument(
        "--ple-norm-position",
        choices=["post", "pre"],
        default="post",
        help="post matches native Gemma; pre gives zero-init projections smooth residual scaling",
    )
    parser.add_argument(
        "--ple-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="PLE parameter dtype; auto follows the loaded model",
    )
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--train-data", help="Local JSONL training data")
    data_source.add_argument("--dataset-preset", choices=sorted(DATASET_PRESETS))
    parser.add_argument("--dataset-cache-dir")
    parser.add_argument("--dataset-max-examples", type=int)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-adapter")
    parser.add_argument(
        "--freeze-ple",
        action="store_true",
        help="For sequential PLE->state tuning, load --resume-adapter PLE and optimize only fresh states",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=["fused-adamw", "adamw"],
        default="fused-adamw",
        help="Fused AdamW avoids a full-size temporary that is prohibitive for faithful PLE",
    )
    parser.add_argument(
        "--manual-allreduce",
        action="store_true",
        help="Use in-place adapter gradient all-reduce instead of memory-heavy DDP buckets",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument(
        "--ple-rms-log-steps",
        default="",
        help="Comma-separated completed-step counts for per-layer PLE RMS diagnostics, such as 0,1,2,5,10",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    return parser.parse_args()


def _parse_step_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    steps = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(step < 0 for step in steps):
        raise ValueError("PLE RMS log steps must be non-negative")
    return steps


def _set_ple_rms_recording(model: torch.nn.Module, enabled: bool) -> None:
    for branch in iter_ple_branches(model):
        branch.record_rms = enabled


def _print_ple_rms(accelerator: Accelerator, model: torch.nn.Module, step: int) -> None:
    for branch in iter_ple_branches(model):
        values = " ".join(f"{name}_rms={value.item():.6e}" for name, value in branch.last_rms.items())
        accelerator.print(
            f"ple_rms step={step} layer={branch.layer_idx} norm_position={branch.norm_position} {values}"
        )


def main() -> None:
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_qwen35_model(
        args.model,
        dtype=resolve_dtype(args.dtype),
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    ple_dtype = None if args.ple_dtype == "auto" else resolve_dtype(args.ple_dtype)
    info = prepare_model_for_tuning(
        model,
        method=args.method,
        ple_dim=args.ple_dim,
        ple_dtype=ple_dtype,
        ple_norm_position=args.ple_norm_position,
    )
    if args.freeze_ple:
        if args.method != "state+ple" or not args.resume_adapter:
            raise ValueError("--freeze-ple requires --method state+ple and --resume-adapter")
        load_adapter(model, args.resume_adapter, strict=False, prepare=False)
        set_trainable_components(model, "state")
    elif args.resume_adapter:
        load_adapter(model, args.resume_adapter)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset_preset = None
    train_data = args.train_data
    if args.dataset_preset:
        train_data, dataset_preset = download_dataset_preset(
            args.dataset_preset,
            cache_dir=args.dataset_cache_dir,
            local_files_only=args.local_files_only,
            endpoint=args.hf_endpoint,
        )
    dataset = StateTuningDataset(
        train_data,
        tokenizer,
        args.max_length,
        max_examples=args.dataset_max_examples,
        sample_seed=args.dataset_seed,
    )
    collator = RightPaddingCollator(tokenizer.pad_token_id)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=shuffle_generator,
    )

    if args.manual_allreduce:
        model.to(accelerator.device)
        dataloader = accelerator.prepare(dataloader)

    parameters = list(tuning_parameters(model))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
        fused=args.optimizer == "fused-adamw",
        foreach=False if args.optimizer == "adamw" else None,
    )
    local_batches = len(dataloader) if args.manual_allreduce else math.ceil(len(dataloader) / accelerator.num_processes)
    updates_per_epoch = math.ceil(local_batches / args.gradient_accumulation_steps)
    total_steps = args.max_steps or args.epochs * updates_per_epoch
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if not args.manual_allreduce:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    accelerator.print(
        f"backend={info.backend} method={info.method} state_layers={info.state_layer_count} "
        f"ple_layers={info.ple_layer_count} ple_dim={info.ple_dim} ple_norm_position={info.ple_norm_position} "
        f"attached_parameters={info.parameter_count:,} optimized_parameters={sum(p.numel() for p in parameters):,} "
        f"examples={len(dataset):,} updates={total_steps:,} "
        f"dataset={dataset_preset.name if dataset_preset else train_data}"
    )

    model.train()
    completed_steps = 0
    running_loss = 0.0
    running_microsteps = 0
    ple_rms_steps = _parse_step_set(args.ple_rms_log_steps)
    logged_ple_rms_steps: set[int] = set()
    optimizer.zero_grad()
    for epoch in range(args.epochs if args.max_steps is None else 10**9):
        for batch_index, batch in enumerate(dataloader):
            record_ple_rms = completed_steps in ple_rms_steps and completed_steps not in logged_ple_rms_steps
            _set_ple_rms_recording(model, record_ple_rms)
            if args.manual_allreduce:
                with accelerator.autocast():
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss
                accelerator.backward(loss)
                should_step = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                    dataloader
                )
            else:
                with accelerator.accumulate(model):
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss
                    accelerator.backward(loss)
                should_step = accelerator.sync_gradients

            if record_ple_rms:
                _print_ple_rms(accelerator, model, completed_steps)
                logged_ple_rms_steps.add(completed_steps)
                _set_ple_rms_recording(model, False)

            if should_step:
                if args.manual_allreduce and accelerator.num_processes > 1:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            torch.distributed.all_reduce(parameter.grad)
                            parameter.grad.div_(accelerator.num_processes)
                accelerator.clip_grad_norm_(parameters, args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float().item()
            running_microsteps += 1
            if not should_step:
                continue
            completed_steps += 1
            if completed_steps % args.log_steps == 0:
                accelerator.print(
                    f"step={completed_steps}/{total_steps} loss={running_loss / running_microsteps:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
                running_loss = 0.0
                running_microsteps = 0
            if args.save_steps and completed_steps % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_adapter(
                        accelerator.unwrap_model(model),
                        Path(args.output_dir) / f"checkpoint-{completed_steps}",
                        base_model=args.model,
                        extra_metadata={
                            "step": completed_steps,
                            "training_stage": "ple_then_state" if args.freeze_ple else "joint",
                            "source_adapter": args.resume_adapter,
                            "dataset_preset": dataset_preset.name if dataset_preset else None,
                            "dataset_revision": dataset_preset.revision if dataset_preset else None,
                        },
                    )
            if completed_steps >= total_steps:
                break
        if completed_steps >= total_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weights_path = save_adapter(
            accelerator.unwrap_model(model),
            args.output_dir,
            base_model=args.model,
            extra_metadata={
                "step": completed_steps,
                "training_stage": "ple_then_state" if args.freeze_ple else "joint",
                "source_adapter": args.resume_adapter,
                "dataset_preset": dataset_preset.name if dataset_preset else None,
                "dataset_revision": dataset_preset.revision if dataset_preset else None,
            },
        )
        accelerator.print(f"saved={weights_path}")


if __name__ == "__main__":
    main()
