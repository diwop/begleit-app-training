import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a dataset using vLLM with batching and Multi-LoRA support.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or HF ID of the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--seq_length", type=int, default=2048, help="Context sequence length threshold")
    parser.add_argument("--lora_rank", type=int, default=64, help="Maximum LoRA rank for vLLM")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Markdown file")
    args = parser.parse_args()

    # Load the JSONL dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    # Slice the dataset to the first 20 samples to save time
    dataset = dataset[:20]

    # Chat Templates & Token Limits
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    prompts = []
    reference_solutions = []
    categories = []
    original_prompts = []

    for entry in dataset:
        messages = entry.get("messages", [])
        
        # Save assistant content as reference solution, then remove it
        reference_solution = ""
        user_content = ""
        input_messages = []
        for msg in messages:
            if msg.get("role") == "assistant":
                reference_solution = msg.get("content", "")
            else:
                input_messages.append(msg)
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    
        reference_solutions.append(reference_solution)
        original_prompts.append(user_content)

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

        # Measure token length of this generated prompt
        tokenized_prompt = tokenizer(prompt)
        prompt_len = len(tokenized_prompt.input_ids)

        if prompt_len > args.seq_length:
            category = "Unseen (Oversized)"
        else:
            category = "Train/Test (Fits Context)"
        categories.append(category)

    # Calculate max model length dynamically
    max_prompt_len = max(len(tokenizer(p).input_ids) for p in prompts)
    max_model_len = max_prompt_len + 4096
    print(f"Max prompt token length: {max_prompt_len}")
    print(f"Dynamically setting max_model_len to {max_model_len}")

    # Initialize vLLM
    print("Initializing vLLM model engine...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        max_model_len=max_model_len,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096
    )

    # 5. Generate Batches
    print("Generating baseline (Base Model) outputs...")
    baseline_outputs = llm.generate(prompts, sampling_params)
    baseline_results = [out.outputs[0].text for out in baseline_outputs]

    print("Generating tuned (Model + Adapter) outputs...")
    tuned_outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("adapter", 1, args.adapter_path)
    )
    tuned_results = [out.outputs[0].text for out in tuned_outputs]

    # 6. Markdown Report
    print(f"Writing Markdown report to {args.output_file}...")
    report_lines = []
    for i in range(len(dataset)):
        entry_idx = i + 1
        category = categories[i]
        original_prompt = original_prompts[i]
        reference_sol = reference_solutions[i]
        baseline_res = baseline_results[i]
        tuned_res = tuned_results[i]

        entry_md = f"""## Entry {entry_idx} | Split: {category}

### Prompt
```text
{original_prompt}
```

### Reference Solution
```text
{reference_sol}
```

### Baseline Result
```text
{baseline_res}
```

### Tuned Result
```text
{tuned_res}
```
---
"""
        report_lines.append(entry_md)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Evaluation pipeline successfully completed!")

if __name__ == "__main__":
    main()