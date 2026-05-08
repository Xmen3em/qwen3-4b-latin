# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.51",
#     "peft>=0.13",
#     "trl>=0.12",
#     "accelerate>=1.0",
#     "datasets>=3.0",
#     "bitsandbytes>=0.44",
#     "huggingface_hub",
#     "hf_transfer",
#     "trackio",
#     "tensorboard",
#     "pyarrow",
# ]
# ///
"""
Continued pretraining on streaming HF datasets — vanilla version (no Unsloth).

Uses transformers + peft + trl directly. Slower than the Unsloth path but
numerically robust. Use this when cpt-streaming.py produces NaN gradients
or fails to learn.

CLI is intentionally identical to cpt-streaming.py:
    uv run cpt-vanilla.py \
        --dataset HuggingFaceFW/fineweb-2 \
        --dataset-name lat_Latn \
        --base-model Qwen/Qwen3-4B-Base \
        --max-steps 1000 \
        --output-repo your-username/qwen3-4b-latin

On HF Jobs:
    hf jobs uv run cpt-vanilla.py \
        --flavor a10g-large --secrets HF_TOKEN --timeout 4h \
        -- --dataset HuggingFaceFW/fineweb-2 \
           --dataset-name lat_Latn \
           --base-model Qwen/Qwen3-4B-Base \
           --max-steps 1000 \
           --output-repo your-username/qwen3-4b-latin
"""

import argparse
import logging
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining on streaming HF datasets (vanilla — no Unsloth)",
    )

    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    parser.add_argument("--dataset-name", default="lat_Latn")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-repo", required=True)

    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch_fused")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)

    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="QLoRA: load base in 4-bit via bitsandbytes (default: True)",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit; load base in bf16 (more VRAM)",
    )

    parser.add_argument("--prefetch-limit", type=int, default=1)
    parser.add_argument("--range-size-limit-mb", type=int, default=128)

    parser.add_argument("--trackio-space", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-local", default="cpt-output")

    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--merge-model",
        action="store_true",
        default=False,
        help="Merge LoRA into base before pushing (larger upload, easier to use)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Continued Pretraining (vanilla — transformers + peft + trl)")
    print("=" * 70)
    print(f"\n  Base model:      {args.base_model}")
    print(f"  Dataset:         {args.dataset} ({args.dataset_name})")
    print(f"  Steps:           {args.max_steps}")
    print(
        f"  Batch:           {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}"
    )
    print(f"  LR:              {args.learning_rate}")
    print(f"  LoRA rank:       {args.lora_r}")
    print(f"  Max seq length:  {args.max_seq_length}")
    print(f"  4-bit base:      {args.load_in_4bit}")
    print(f"  Output repo:     {args.output_repo}")
    print()

    check_cuda()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login
    import pyarrow
    import pyarrow.dataset

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")

    # 1. Load tokenizer + model
    print("\n[1/5] Loading model...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Streaming dataset
    print("\n[2/5] Loading streaming dataset...")
    start = time.time()

    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=args.prefetch_limit,
            range_size_limit=args.range_size_limit_mb << 20,
        ),
    )

    dataset = load_dataset(
        args.dataset,
        name=args.dataset_name,
        split="train",
        streaming=True,
        fragment_scan_options=fragment_scan_options,
    )

    if args.num_samples:
        dataset = dataset.take(args.num_samples)

    text_field = args.text_field
    eos = tokenizer.eos_token

    def format_text(example):
        return {"text": example[text_field] + eos}

    dataset = dataset.map(format_text)
    print(f"  Streaming from: {args.dataset} ({args.dataset_name})")
    print(f"  Dataset ready in {time.time() - start:.1f}s")

    # 3. Trainer
    print("\n[3/5] Configuring trainer...")

    run_name = args.run_name or f"cpt-vanilla-{args.max_steps}steps"
    logging_steps = max(1, args.max_steps // 20)
    save_steps = max(1, args.max_steps // 4)

    report_to = ["tensorboard"]
    if args.trackio_space:
        report_to.append("trackio")

    training_config = SFTConfig(
        output_dir=args.save_local,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        logging_steps=logging_steps,
        optim=args.optim,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=True,
        bf16=True,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=True,
        hub_model_id=args.output_repo,
        save_steps=save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_config,
    )

    # 4. Train
    print(f"\n[4/5] Training for {args.max_steps} steps...")
    start = time.time()
    train_result = trainer.train()
    train_time = time.time() - start
    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    if "train_loss" in train_result.metrics:
        print(f"  Final train loss: {train_result.metrics['train_loss']:.4f}")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    if args.merge_model:
        print(f"Merging and pushing to {args.output_repo}...")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)
        merged.push_to_hub(args.output_repo)
        tokenizer.push_to_hub(args.output_repo)
        print(f"Merged model: https://huggingface.co/{args.output_repo}")
    else:
        model.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)
        model.push_to_hub(args.output_repo)
        tokenizer.push_to_hub(args.output_repo)
        print(f"Adapter: https://huggingface.co/{args.output_repo}")

    from huggingface_hub import metadata_update

    metadata_update(
        args.output_repo,
        {"datasets": [args.dataset]},
        overwrite=True,
    )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
