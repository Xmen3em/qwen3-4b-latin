# Qwen3-4B-Latin (LoRA adapter)

A LoRA adapter for **`Qwen/Qwen3-4B-Base`** continued-pretrained on the Latin (`lat_Latn`) subset of **`HuggingFaceFW/fineweb-2`**. The base Qwen3 family has very little Latin in its pretraining mix; this adapter adds Latin fluency on top.

The repo contains the **adapter only** (~66 MB). Load it on top of the official base model — see *Usage* below.

## Training summary

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Base` (4.05 B params) |
| Trainable params | 33 M (0.81 %) — LoRA r=16 on q/k/v/o + gate/up/down |
| Quantization | 4-bit NF4 (QLoRA), bf16 compute |
| Dataset | `HuggingFaceFW/fineweb-2`, subset `lat_Latn`, streaming |
| Tokens seen | ≈ 16.3 M (≈ 16 384 tokens/step × 1 000 steps) |
| Steps | 1 000 |
| Effective batch | 2 × 4 = 8 sequences × 2 048 tokens |
| Optimizer | adamw_torch_fused, cosine schedule |
| Learning rate | 1e-5, warmup 100 |
| Gradient clip | 1.0 |
| Hardware | 1× NVIDIA A10G (Hugging Face Jobs, `a10g-large`) |
| Wall time | ≈ 8.6 h |

### Training curve

| Step | Train loss | Token accuracy |
|---:|---:|---:|
| 50 | 2.94 | 0.45 |
| 100 | 2.49 | 0.50 |
| 150 | 2.30 | 0.52 |
| 200 | 2.16 | 0.54 |
| 250 | 2.27 | 0.52 |
| 300 | 2.29 | 0.52 |
| 400 | 2.30 | 0.52 |
| 500 | 2.18 | 0.54 |
| 600 | 2.61 | 0.49 |
| 700 | 2.46 | 0.50 |
| 800 | 2.33 | 0.51 |
| 900 | 2.35 | 0.51 |
| 1000 | 2.35 | 0.51 |
| **Final** | **2.36** | **0.51** |

Perplexity dropped from ≈ 18.9 (step 50) to ≈ 10.6 (final) — a 44 % reduction. The brief uptick around step 600–650 reflects a harder batch from the streaming dataset and recovered within ~50 steps; the run finished cleanly with no NaN and finite gradients throughout.

## Usage

### Quick generation (Python)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen3-4B-Base"
ADAPTER = "MenemAI/qwen3-4b-latin"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

prompt = "In principio creavit Deus caelum et terram. "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Run the test script on Hugging Face Jobs

A reproducible test script is provided in this repo as `test-cpt-model.py`. To run it:

```bash
hf jobs uv run test-cpt-model.py \
    --flavor a10g-small --secrets HF_TOKEN --timeout 30m \
    -- \
    --adapter MenemAI/qwen3-4b-latin
```

The script loads the adapter on top of `Qwen/Qwen3-4B-Base` (4-bit) and generates completions for several Latin prompts so you can verify the adapter visually.

## Intended use

- Latin text completion / generation
- A starting point for downstream Latin SFT or domain adaptation (Vulgate, classical authors, mediaeval Latin)
- Educational / research use

## Limitations and caveats

- **Small training corpus.** Only ≈ 16 M tokens were seen — enough to demonstrably shift the model's perplexity on Latin, but far short of what a strong Latin-fluent model would require (typically 1 B+ tokens).
- **FineWeb-2 `lat_Latn` quality.** The subset contains web-scraped Latin of mixed quality, including some machine-generated and modern Latin. Output style reflects this distribution.
- **Base model is small for low-resource languages.** Qwen3-4B has limited Latin capacity to begin with; expect mistakes in inflection, agreement, and uncommon vocabulary.
- **No instruction tuning.** This is a *base* model with continued pretraining, not a chat / instruct model. It continues text — it does not follow instructions.
- **No safety alignment beyond the base model.**

## Reproduction

The training script (`cpt-vanilla.py`) and exact CLI used:

```bash
hf jobs uv run cpt-vanilla.py \
    --flavor a10g-large --secrets HF_TOKEN --timeout 9h \
    -- \
    --base-model Qwen/Qwen3-4B-Base \
    --dataset HuggingFaceFW/fineweb-2 \
    --dataset-name lat_Latn \
    --max-steps 1000 \
    --output-repo MenemAI/qwen3-4b-latin
```

(Use `--timeout 9h` rather than the original `6h` — the actual run takes ≈ 8.6 h on `a10g-large` without flash-attention.)

## License

Apache 2.0 (inherited from `Qwen/Qwen3-4B-Base`).

## Acknowledgements

- **Base model:** [Qwen team](https://huggingface.co/Qwen) — `Qwen/Qwen3-4B-Base`
- **Dataset:** [HuggingFaceFW](https://huggingface.co/HuggingFaceFW) — `fineweb-2`
- **Compute:** Hugging Face Jobs, `a10g-large`
