import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, messages, max_new_tokens=8192):
    """Generates a response using the chat template."""
    # Strip out the target assistant response so the model has to generate it
    input_messages = [m for m in messages if m["role"] != "assistant"]
    
    prompt = tokenizer.apply_chat_template(
        input_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False # Greedy decoding for reproducible evaluation (no temperature)
        )
    
    # Slice the output to only return the newly generated text
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="HF model ID (e.g., cyankiwi/Mistral-Small-4...)")
    parser.add_argument("--adapter_path", type=str, default="/app/output", help="Path to your trained QLoRA weights")
    parser.add_argument("--dataset_path", type=str, default="data/train/dataset.jsonl")
    parser.add_argument("--output_file", type=str, default="data/evaluation_results.jsonl")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load base model (Transformers will auto-detect AWQ or BNB from the config)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Load the dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # To save time on testing, let's just evaluate the first 20 samples
    dataset = dataset[:20] 
    results = []

    # Generate Base Model Baseline
    print("\nGenerating Base Model Responses")
    for i, entry in enumerate(dataset, 1):
        print(f"Processing {i}/{len(dataset)}...")
        base_text = generate_response(model, tokenizer, entry["messages"])
        
        # Save state
        results.append({
            "id": entry.get("id", i),
            "original_prompt": entry["messages"][1]["content"], # The standard German text
            "target_leichte_sprache": entry["messages"][2]["content"],
            "base_model_output": base_text
        })

    # Hot-Swap the adapter
    print(f"\nAttaching adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Generate Fine-Tuned Responses
    print("\nGenerating Fine-Tuned Responses")
    for i, entry in enumerate(dataset, 1):
        print(f"Processing {i}/{len(dataset)}...")
        tuned_text = generate_response(model, tokenizer, entry["messages"])
        results[i-1]["tuned_model_output"] = tuned_text

    # Save Results
    with open(args.output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"\nEvaluation complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()