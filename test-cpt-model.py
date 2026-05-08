# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.51",
#     "peft>=0.13",
#     "accelerate>=1.0",
#     "bitsandbytes>=0.44",
#     "huggingface_hub",
#     "hf_transfer",
# ]
# ///
"""
Test the qwen3-4b-latin LoRA adapter by loading it on top of the base model
and generating completions for a handful of Latin prompts.

Local:
    uv run test-cpt-model.py --adapter MenemAI/qwen3-4b-latin

On HF Jobs:
    hf jobs uv run test-cpt-model.py \
        --flavor a10g-small --secrets HF_TOKEN --timeout 30m \
        -- --adapter MenemAI/qwen3-4b-latin
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


DEFAULT_PROMPTS = [
    "In principio creavit Deus caelum et terram. ",
    "Gallia est omnis divisa in partes tres, ",
    "Veni, vidi, vici. Caesar dixit ",
    "Roma aeterna est. Cives Romani ",
    "Lingua Latina est lingua antiqua quae ",
    "Cogito, ergo sum. Hoc principium philosophiae ",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--adapter", default="MenemAI/qwen3-4b-latin")
    p.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        default=True,
        help="Load base in bf16 instead of 4-bit (uses more VRAM)",
    )
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Override the default prompts (pass multiple times for several)",
    )
    p.add_argument(
        "--compare-base",
        action="store_true",
        help="Also generate from the base model (no adapter) for side-by-side comparison",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("qwen3-4b-latin — test generation")
    print("=" * 70)
    print(f"  Base model: {args.base_model}")
    print(f"  Adapter:    {args.adapter}")
    print(f"  4-bit:      {args.load_in_4bit}")
    print(f"  Sampling:   T={args.temperature}, top_p={args.top_p}, rep_pen={args.repetition_penalty}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login

    if not torch.cuda.is_available():
        print("CUDA not available; running on CPU will be very slow.", file=sys.stderr)

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    torch.manual_seed(args.seed)

    print("Loading base model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_kwargs = {"quantization_config": bnb}
    else:
        base_kwargs = {"torch_dtype": torch.bfloat16}

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="sdpa",
        **base_kwargs,
    )
    base.eval()
    print(f"Base loaded in {time.time() - t0:.1f}s")

    print(f"Loading adapter from {args.adapter}...")
    t0 = time.time()
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print(f"Adapter loaded in {time.time() - t0:.1f}s")

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    def generate(m, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):]

    for i, prompt in enumerate(prompts, 1):
        print("\n" + "-" * 70)
        print(f"[{i}/{len(prompts)}] Prompt:")
        print(f"  {prompt!r}")

        if args.compare_base:
            torch.manual_seed(args.seed + i)
            with model.disable_adapter():
                base_completion = generate(model, prompt)
            print("\n--- Base (no adapter) ---")
            print(base_completion)

        torch.manual_seed(args.seed + i)
        adapter_completion = generate(model, prompt)
        print("\n--- With adapter ---")
        print(adapter_completion)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
