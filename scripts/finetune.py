"""Fine-tune base SLM on Alpaca BFSI dataset using LoRA. Saves adapters to models/adapters/v1.0."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, PROJECT_ROOT


def load_alpaca(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config()
    slm_cfg = cfg.get("slm", {})
    base_model = slm_cfg.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_path = PROJECT_ROOT / slm_cfg.get("adapter_path", "models/adapters/v1.0")
    dataset_path = PROJECT_ROOT / cfg.get("similarity", {}).get("dataset_path", "data/alpaca_bfsi.json")
    use_4bit = slm_cfg.get("use_4bit", False)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        print("Install: pip install transformers peft datasets")
        raise SystemExit(1) from e

    try:
        import torch
    except ImportError:
        print("PyTorch required for fine-tuning.")
        raise SystemExit(1)

    samples = load_alpaca(dataset_path)

    def format_prompt(instruction: str, input_text: str, output: str) -> str:
        inp = input_text.strip() if (input_text or "").strip() else "N/A"
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        )

    texts = [
        format_prompt(s["instruction"], s.get("input", ""), s["output"])
        for s in samples
    ]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        tokenizer = tokenizer_ref[0]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_ref = [tokenizer]
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    model_kwargs = {"trust_remote_code": True}
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("bitsandbytes not available; training in full precision")
            use_4bit = False
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    adapter_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(adapter_path),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("Adapters saved to", adapter_path)


if __name__ == "__main__":
    main()
