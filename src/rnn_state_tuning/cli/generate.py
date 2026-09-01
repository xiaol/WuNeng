from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from rnn_state_tuning import load_adapter
from rnn_state_tuning.modeling import input_device, load_qwen35_model, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a state-tuned Qwen3.5 adapter")
    parser.add_argument("--model", required=True, help="Qwen3.5 model ID or local checkpoint")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = load_qwen35_model(
        args.model,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    load_adapter(model, args.adapter)
    model.eval()

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = input_device(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
