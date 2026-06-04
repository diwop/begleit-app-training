import argparse
import json
import gc
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Hardcoded Challenger Variables
CHALLENGER_BASE = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
CHALLENGER_ADAPTER = "tschomacker/lora_adapter_llama_3.1_8B"

def generate_response(model, tokenizer, messages, max_new_tokens=4096):
    """Helper function to format, tokenize, and generate text safely."""
    # Format prompt using the model's native chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # dynamically send inputs to the exact GPU where the model starts
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Strip the input tokens to isolate just the new generated text
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline, Custom LoRA, and Challenger Models.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or HF ID of the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--seq_length", type=int, default=16384, help="Context sequence length threshold")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (kept for CLI compatibility)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Markdown file")
    args = parser.parse_args()

    # Hardware Check
    num_gpus = torch.cuda.device_count()
    print(f"\n[Hardware Check] Detected {num_gpus} GPU(s).")
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB VRAM)")

    # 1. Load the JSONL dataset and slice
    print(f"\nLoading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    dataset = dataset[:20]

    # Pre-process evaluation data
    evaluation_data = []
    for entry in dataset:
        messages = entry.get("messages", [])
        reference_sol = ""
        input_messages = []
        original_prompt = ""
        
        for msg in messages:
            if msg.get("role") == "assistant":
                reference_sol = msg.get("content", "")
            else:
                input_messages.append(msg)
                if msg.get("role") == "user":
                    original_prompt = msg.get("content", "")
                    
        evaluation_data.append({
            "prompt": original_prompt,
            "reference": reference_sol,
            "input_messages": input_messages
        })

    # ==========================================
    # STAGE 1: MIXTRAL BASELINE
    # ==========================================
    print(f"\n--- STAGE 1: Loading Baseline ({args.base_model}) ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )

    baseline_results = []
    for i, data in enumerate(evaluation_data):
        print(f"Generating Baseline {i+1}/{len(evaluation_data)}...")
        baseline_results.append(generate_response(model, tokenizer, data["input_messages"]))

    # ==========================================
    # STAGE 2: MIXTRAL TUNED (YOUR LORA)
    # ==========================================
    print(f"\n--- STAGE 2: Loading Tuned Adapter ({args.adapter_path}) ---")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    tuned_results = []
    for i, data in enumerate(evaluation_data):
        print(f"Generating Tuned {i+1}/{len(evaluation_data)}...")
        tuned_results.append(generate_response(model, tokenizer, data["input_messages"]))

    # ==========================================
    # VRAM WIPE
    # ==========================================
    print("\n[Wiping VRAM for Challenger Model...]")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # STAGE 3: CHALLENGER MODEL
    # ==========================================
    print(f"\n--- STAGE 3: Loading Challenger ({CHALLENGER_BASE} + {CHALLENGER_ADAPTER}) ---")
    tokenizer = AutoTokenizer.from_pretrained(CHALLENGER_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        CHALLENGER_BASE, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, CHALLENGER_ADAPTER)

    challenger_results = []
    for i, data in enumerate(evaluation_data):
        print(f"Generating Challenger {i+1}/{len(evaluation_data)}...")
        challenger_results.append(generate_response(model, tokenizer, data["input_messages"]))

    # Final Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # OUTPUT MARKDOWN REPORT
    # ==========================================
    print(f"\nWriting Markdown report to {args.output_file}...")
    report_lines = []
    
    for i in range(len(evaluation_data)):
        data = evaluation_data[i]
        entry_md = f"""## Entry {i + 1}

### Prompt
```text
{data['prompt']}
```

### Reference Solution
```text
{data['reference']}
```

### Baseline ({args.base_model})
```text
{baseline_results[i]}
```

### Tuned ({args.base_model} with adapter)
```text
{tuned_results[i]}
```

### Challenger ({CHALLENGER_BASE} with {CHALLENGER_ADAPTER})
```text
{challenger_results[i]}
```
"""

    report_lines.append(entry_md)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=== Evaluation pipeline successfully completed! ===")

if __name__ == "__main__":
    main()