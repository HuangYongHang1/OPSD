#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def load_model_class(model_loader: str):
    if model_loader == "causal_lm":
        return AutoModelForCausalLM
    if model_loader == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    raise ValueError(f"Unknown model_loader: {model_loader}")


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype: {dtype}")
    return mapping[dtype]


def main():
    parser = argparse.ArgumentParser(description="Generate with Qwen3.5 using native Transformers/Torch.")
    parser.add_argument("--model", default="/Users/hyh/Desktop/Qwen3.5_2B", help="Base model path.")
    parser.add_argument("--checkpoint_dir", default=None, help="Optional LoRA checkpoint directory.")
    parser.add_argument("--model_loader", default="image_text_to_text", choices=["causal_lm", "image_text_to_text"])
    parser.add_argument("--prompt", default="Problem: What is 17 + 28? Please reason step by step, and put your final answer within \\boxed{}.")
    parser.add_argument("--enable_thinking", default="False", help="True/False passed to the chat template.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = load_model_class(args.model_loader)
    model = model_cls.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_path))

    model.eval()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=str_to_bool(args.enable_thinking),
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)

    print("\n===== Prompt =====")
    print(args.prompt)
    print("\n===== Completion =====")
    print(completion)


if __name__ == "__main__":
    main()
