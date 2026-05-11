import os
import argparse
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from src.prepare_data import load_and_format_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="unsloth/Mixtral-8x7B-v0.1-bnb-4bit")
    parser.add_argument("--dataset_path", type=str, default="data/sample_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="/workspace/output")
    args = parser.parse_args()

    print(f"Initializing model: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )

    print("Applying QLoRA/QDoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_dora=True, # Requested by PRD
    )

    print("Loading dataset...")
    dataset = load_and_format_data(args.dataset_path)

    print("Configuring Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            max_steps=10, # Brief fine-tuning job for testing
            learning_rate=2e-4,
            fp16=not FastLanguageModel.is_bfloat16_supported(),
            bf16=FastLanguageModel.is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
