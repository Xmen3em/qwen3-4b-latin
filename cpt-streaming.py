# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
#     "transformers==4.57.3",
#     "pyarrow",
# ]
# ///
"""
Continued pretraining on massive streaming datasets using Unsloth optimizations.

Trains a base LLM on raw text from Hugging Face datasets without downloading
them locally — data is streamed directly from the Hub. Ideal for resource-
constrained environments (Colab, Kaggle, HF Jobs) where disk space is limited.

Uses Unsloth for ~60% less VRAM and 2x faster training, plus PyArrow prefetching
for higher throughput when streaming Parquet shards.

Streaming-based training (recommended for huge corpora):
    uv run cpt-streaming.py \
        --dataset HuggingFaceFW/fineweb-2 \
        --dataset-name lat_Latn \
        --max-steps 1000 \
        --output-repo your-username/qwen3-latin

Run on HF Jobs (1000 steps on Qwen3-4B with FineWeb-2 Latin):
    hf jobs uv run cpt-streaming.py \
        --flavor a100-large --secrets HF_TOKEN --timeout 4h \
        -- --dataset HuggingFaceFW/fineweb-2 \
           --dataset-name lat_Latn \
           --base-model unsloth/Qwen3-4B-Base-unsloth-bnb-4bit \
           --max-steps 1000 \
           --output-repo your-username/qwen3-4b-latin

Quick smoke test (50 steps, 0.6B model):
    uv run cpt-streaming.py \
        --dataset HuggingFaceFW/fineweb-2 \
        --dataset-name lat_Latn \
        --max-steps 50 \
        --output-repo your-username/qwen3-test
"""

import argparse
import logging
import os
import sys
import time

# Force unbuffered output for HF Jobs logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda():
    """Check CUDA availability and exit if not available."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on a machine with a CUDA-capable GPU or use HF Jobs:")
        logger.error("  hf jobs uv run cpt-streaming.py --flavor a10g-small ...")
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining on streaming HF datasets with Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  uv run cpt-streaming.py \\
      --dataset HuggingFaceFW/fineweb-2 \\
      --dataset-name lat_Latn \\
      --max-steps 50 \\
      --output-repo username/qwen3-test

  # Full streaming run on Qwen3-4B
  uv run cpt-streaming.py \\
      --dataset HuggingFaceFW/fineweb-2 \\
      --dataset-name lat_Latn \\
      --base-model unsloth/Qwen3-4B-Base-unsloth-bnb-4bit \\
      --max-steps 1000 \\
      --output-repo username/qwen3-4b-latin

  # With Trackio monitoring
  uv run cpt-streaming.py \\
      --dataset HuggingFaceFW/fineweb-2 \\
      --dataset-name lat_Latn \\
      --max-steps 1000 \\
      --output-repo username/qwen3-latin \\
      --trackio-space username/trackio
        """,
    )

    # Model and data
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
        help="Base model (default: unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit)",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-2",
        help="HF dataset to stream (default: HuggingFaceFW/fineweb-2)",
    )
    parser.add_argument(
        "--dataset-name",
        default="lat_Latn",
        help="Dataset config/subset name, e.g. language code for FineWeb-2 (default: lat_Latn)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Column name containing raw text (default: text)",
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="HF Hub repo to push model to (e.g., 'username/qwen3-latin')",
    )

    # Training config
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Training steps (default: 1000). Required for streaming datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch-size * this",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5). For raw-text CPT on a low-resource language, keep this low. 5e-5 and above frequently diverge with Qwen3-4B + 4-bit + LoRA.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (default: 1.0). Critical for stability with 4-bit base + LoRA — without it, a single bad batch can produce NaN gradients that poison the rest of training.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps (default: 100). For 4-bit + LoRA + Unsloth on low-resource languages, generous warmup is essential to avoid early NaN gradients.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Optimizer (default: adamw_torch_fused). Use full-precision optimizer state — adamw_8bit can corrupt permanently after a single NaN gradient.",
    )

    # LoRA config
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16). Higher = more capacity but more VRAM",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16). Same as r per Unsloth recommendation",
    )

    # Quantization
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load base model in 4-bit (default: True). Recommended for Qwen-family unsloth-bnb-4bit checkpoints.",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading (use 16-bit instead)",
    )

    # Streaming / prefetch
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=1,
        help="PyArrow Parquet prefetch_limit (default: 1)",
    )
    parser.add_argument(
        "--range-size-limit-mb",
        type=int,
        default=128,
        help="PyArrow range_size_limit in MB (default: 128)",
    )

    # Logging
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="HF Space for Trackio dashboard (e.g., 'username/trackio')",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name for Trackio (default: auto-generated)",
    )
    parser.add_argument(
        "--save-local",
        default="cpt-output",
        help="Local directory to save model (default: cpt-output)",
    )

    # Data control
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit samples taken from the stream (default: None = unlimited)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility (default: 3407)",
    )
    parser.add_argument(
        "--merge-model",
        action="store_true",
        default=False,
        help="Merge LoRA weights into base model before uploading (larger file, easier to use)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    duration_str = f"{args.max_steps} steps"

    print("=" * 70)
    print("Continued Pretraining on Streaming Datasets with Unsloth")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Base model:      {args.base_model}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Dataset subset:  {args.dataset_name}")
    print(f"  Text field:      {args.text_field}")
    print(f"  Num samples:     {args.num_samples or 'unlimited (stream)'}")
    print(f"  Seed:            {args.seed}")
    print(f"  Training:        {duration_str}")
    print(
        f"  Batch size:      {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}"
    )
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  LoRA rank:       {args.lora_r}")
    print(f"  Max seq length:  {args.max_seq_length}")
    print(f"  Load in 4-bit:   {args.load_in_4bit}")
    print(f"  Output repo:     {args.output_repo}")
    print(f"  Trackio space:   {args.trackio_space or '(not configured)'}")
    print()

    # Check CUDA before heavy imports
    check_cuda()

    # Enable fast transfers
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Set Trackio space if provided
    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
        logger.info(
            f"Trackio dashboard: https://huggingface.co/spaces/{args.trackio_space}"
        )

    # Import heavy dependencies
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login
    import pyarrow
    import pyarrow.dataset

    # Login to Hub
    token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.warning("HF_TOKEN not set - model upload may fail")

    # 1. Load model
    print("\n[1/5] Loading model...")
    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    # Add LoRA adapters with Qwen-style target modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Load and prepare streaming dataset
    print("\n[2/5] Loading streaming dataset...")
    start = time.time()

    # PyArrow prefetching for higher throughput on Parquet shards
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
    print(f"  Streaming from: {args.dataset} ({args.dataset_name})")

    if args.num_samples:
        dataset = dataset.take(args.num_samples)
        print(f"  Limited stream to first {args.num_samples} samples")

    # Append EOS token to each text field — no chat template for raw-text CPT
    text_field = args.text_field
    eos_token = tokenizer.eos_token

    def format_text(example):
        return {"text": example[text_field] + eos_token}

    dataset = dataset.map(format_text)

    print(f"  Dataset ready in {time.time() - start:.1f}s")

    # 3. Configure trainer
    print("\n[3/5] Configuring trainer...")

    # Determine run name and logging steps
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"cpt-stream-{args.max_steps}steps"

    logging_steps = max(1, args.max_steps // 20)
    save_steps = max(1, args.max_steps // 4)

    # Determine reporting backend
    if args.trackio_space:
        report_to = ["tensorboard", "trackio"]
    else:
        report_to = ["tensorboard"]

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
        lr_scheduler_type="linear",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=True,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=True,
        hub_model_id=args.output_repo,
        save_steps=save_steps,
        save_total_limit=3,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_config,
    )

    # 4. Train
    print(f"\n[4/5] Training for {duration_str}...")
    start = time.time()

    train_result = trainer.train()

    train_time = time.time() - start
    total_steps = train_result.metrics.get("train_steps", args.max_steps)
    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    print(f"  Speed: {total_steps / train_time:.2f} steps/s")

    # Print training metrics
    train_loss = train_result.metrics.get("train_loss")
    if train_loss:
        print(f"  Final train loss: {train_loss:.4f}")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    if args.merge_model:
        print("Merging LoRA weights into base model...")
        print(f"\nPushing merged model to {args.output_repo}...")
        model.push_to_hub_merged(
            args.output_repo,
            tokenizer=tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model available at: https://huggingface.co/{args.output_repo}")
    else:
        model.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)
        print(f"Saved locally to {args.save_local}/")

        print(f"\nPushing adapter to {args.output_repo}...")
        model.push_to_hub(args.output_repo, tokenizer=tokenizer)
        print(f"Adapter available at: https://huggingface.co/{args.output_repo}")

    # Update model card metadata with dataset info
    from huggingface_hub import metadata_update

    metadata_update(
        args.output_repo,
        {"datasets": [args.dataset]},
        overwrite=True,
    )
    print(f"  Model card updated with dataset: {args.dataset} ({args.dataset_name})")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Continued Pretraining on Streaming Datasets with Unsloth")
        print("=" * 70)
        print("\nTrain a base LLM on massive HF datasets without downloading them.")
        print("\nFeatures:")
        print("  - Streams data directly from the Hub (streaming=True)")
        print("  - PyArrow prefetching for higher throughput")
        print("  - ~60% less VRAM with Unsloth + 4-bit base models")
        print("  - 2x faster training vs standard methods")
        print("  - Step-based training (epochs are undefined for streams)")
        print("  - Raw-text CPT (no chat template, no response masking)")
        print("\nStreaming run:")
        print("\n  uv run cpt-streaming.py \\")
        print("      --dataset HuggingFaceFW/fineweb-2 \\")
        print("      --dataset-name lat_Latn \\")
        print("      --max-steps 1000 \\")
        print("      --output-repo your-username/qwen3-latin")
        print("\nHF Jobs example:")
        print("\n  hf jobs uv run cpt-streaming.py \\")
        print("      --flavor a100-large --secrets HF_TOKEN --timeout 4h \\")
        print("      -- --dataset HuggingFaceFW/fineweb-2 \\")
        print("         --dataset-name lat_Latn \\")
        print("         --base-model unsloth/Qwen3-4B-Base-unsloth-bnb-4bit \\")
        print("         --max-steps 1000 \\")
        print("         --output-repo your-username/qwen3-4b-latin")
        print("\nFor full help: uv run cpt-streaming.py --help")
        print("=" * 70)
        sys.exit(0)

    main()
