import argparse
import json
import gc
import torch
import textstat
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Hardcoded Challenger Variables (Swapped to AWQ for vLLM compatibility)
CHALLENGER_BASE = "casperhansen/llama-3.1-8b-instruct-awq"
CHALLENGER_ADAPTER = "tschomacker/lora_adapter_llama_3.1_8B"

def format_metric(value: float) -> str:
    """Formats a float to 1 optional decimal place (e.g., 4.0 -> 4, 4.5 -> 4.5)"""
    formatted = f"{value:.1f}"
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def get_metrics_str(text: str) -> str:
    """Calculates German textstat metrics and formats the header string."""
    if not text.strip():
        return "(FRE N/A, WSTF1 N/A)"
    fre = textstat.flesch_reading_ease(text)
    wstf = textstat.wiener_sachtextformel(text, 1)
    return f"(FRE {format_metric(fre)}, WSTF1 {format_metric(wstf)})"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline, Custom LoRA, and Challenger Models using vLLM.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or HF ID of the AWQ base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--seq_length", type=int, default=16384, help="Context sequence length threshold")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (kept for CLI compatibility)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Markdown file")
    args = parser.parse_args()

    # Configure Textstat for German
    textstat.set_lang("de")

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

        original_text = entry.get("original_text", original_prompt)            
                    
        evaluation_data.append({
            "original": original_text,
            "reference": reference_sol,
            "input_messages": input_messages
        })

    # Common Generation Config
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    # ========================================================
    # STAGE 1 & 2: BASELINE & TUNED ADAPTER (Shared Base Model)
    # ========================================================
    print(f"\n--- INITIALIZING vLLM: Loading AWQ Base Model ({args.base_model}) with LoRA Enabled ---")
    
    # Load base tokenizer to format prompts natively via chat template
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    formatted_prompts = [
        tokenizer.apply_chat_template(data["input_messages"], tokenize=False, add_generation_prompt=True)
        for data in evaluation_data
    ]

    # Spin up the vLLM Engine
    llm = LLM(
        model=args.base_model,
        tokenizer=args.adapter_path, # need to take the adapter's tokenizer config because the plain inference model has a different one
        quantization="awq",
        enable_lora=True,
        max_model_len=args.seq_length,
        max_loras=1,
        max_lora_rank=args.lora_rank
    )

    # Batch Generate Stage 1 (Baseline)
    print(f"\n🚀 Executing batched generation for Baseline ({len(formatted_prompts)} samples)...")
    baseline_outputs = llm.generate(formatted_prompts, sampling_params)
    baseline_results = [output.outputs[0].text.strip() for output in baseline_outputs]

    # Batch Generate Stage 2 (Tuned Adapter via dynamic LoRARequest)
    print(f"\n🚀 Executing batched generation for Tuned Adapter using dynamic mapping...")
    tuned_outputs = llm.generate(
        formatted_prompts, 
        sampling_params,
        lora_request=LoRARequest("tuned_adapter", 1, args.adapter_path)
    )
    tuned_results = [output.outputs[0].text.strip() for output in tuned_outputs]

    # VRAM WIPE
    print("\n[Wiping vLLM Engine Memory for Challenger Model...]")
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================
    # STAGE 3: CHALLENGER MODEL
    # ========================================================
    print(f"\n--- INITIALIZING vLLM: Loading Challenger AWQ Base ({CHALLENGER_BASE}) ---")
    
    chal_tokenizer = AutoTokenizer.from_pretrained(CHALLENGER_BASE)
    chal_formatted_prompts = [
        chal_tokenizer.apply_chat_template(data["input_messages"], tokenize=False, add_generation_prompt=True)
        for data in evaluation_data
    ]

    llm_challenger = LLM(
        model=CHALLENGER_BASE,
        quantization="awq",
        enable_lora=True,
        max_model_len=args.seq_length,
        max_loras=1,
        max_lora_rank=args.lora_rank
    )

    print(f"\n🚀 Executing batched generation for Challenger ({len(chal_formatted_prompts)} samples)...")
    challenger_outputs = llm_challenger.generate(
        chal_formatted_prompts,
        sampling_params,
        lora_request=LoRARequest("challenger_adapter", 2, CHALLENGER_ADAPTER)
    )
    challenger_results = [output.outputs[0].text.strip() for output in challenger_outputs]

    # Final Cleanup
    del llm_challenger
    del chal_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # OUTPUT MARKDOWN REPORT
    # ==========================================
    print(f"\nWriting Markdown report to {args.output_file}...")
    report_lines = []
    
    for i in range(len(evaluation_data)):
        data = evaluation_data[i]
        
        orig_text = data['original']
        ref_text = data['reference']
        base_text = baseline_results[i]
        tuned_text = tuned_results[i]
        chal_text = challenger_results[i]
        
        # Calculate dynamic metrics for the headers
        m_orig = get_metrics_str(orig_text)
        m_ref = get_metrics_str(ref_text)
        m_base = get_metrics_str(base_text)
        m_tuned = get_metrics_str(tuned_text)
        m_chal = get_metrics_str(chal_text)

        entry_md = f"""## Entry {i + 1}

### Original (*{m_orig}*)
```text
{data['original']}
```

### Reference Solution (*{m_ref}*)
```text
{data['reference']}
```

### {args.base_model} (*{m_base}*)
```text
{baseline_results[i]}
```

### fine-tuned {args.base_model} (*{m_tuned}*)
```text
{tuned_results[i]}
```

### {CHALLENGER_BASE} tuned with {CHALLENGER_ADAPTER} (*{m_chal}*)
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