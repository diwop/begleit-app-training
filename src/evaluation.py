import argparse
import json
import gc
import torch
import textstat
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Hardcoded Challenger Variables
CHALLENGER_BASE = "casperhansen/llama-3.1-8b-instruct-awq"
CHALLENGER_ADAPTER = "tschomacker/lora_adapter_llama_3.1_8B"

def format_metric(value: float) -> str:
    """Formats a float to 1 optional decimal place."""
    formatted = f"{value:.1f}"
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def get_metrics_str(text: str) -> str:
    """Calculates German textstat metrics and formats the header string."""
    if not text.strip():
        return "(FRE N/A, WSTF1 N/A)"
    fre = textstat.flesch_reading_ease(text)
    wstf = textstat.wiener_sachtextformel(text, 1)
    return f"(FRE {format_metric(fre)}, WSTF1 {format_metric(wstf)})"

def get_model_loading_kwargs(model_id: str) -> dict:
    """Dynamically determines the correct backend loading arguments based on the model name."""
    m_lower = model_id.lower()
    
    # Base settings shared by all implementations on Ampere/Hopper
    base_kwargs = {
        "device_map": "auto",
        "attn_implementation": "flash_attention_2"  # Forces high-speed math kernels
    }
    
    if "awq" in m_lower:
        print(f" -> Mapping {model_id} to AWQ FP16 layout...")
        base_kwargs["torch_dtype"] = torch.float16
        return base_kwargs
        
    elif "bnb" in m_lower or "4bit" in m_lower:
        print(f" -> Mapping {model_id} to BitsAndBytes 4-Bit NF4 layout...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_kwargs["quantization_config"] = bnb_config
        return base_kwargs
        
    else:
        # Handles native BF16 models (Gemma 4, Mistral Small 4)
        print(f" -> Mapping {model_id} to native unquantized BF16 layout...")
        base_kwargs["torch_dtype"] = torch.bfloat16
        return base_kwargs

def generate_batched(model, tokenizer, prompts, batch_size=4, desc="Generating"):
    """Executes high-throughput left-padded batch generation with a real-time progress bar."""
    results = []
    
    # Wrap the range iterator with tqdm
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        for j, out in enumerate(outputs):
            gen_tokens = out[inputs.input_ids.shape[1]:]
            results.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline, Custom LoRA, and Challenger Models using HF Transformers.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or HF ID of the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--seq_length", type=int, default=16384, help="Context sequence length threshold")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (kept for CLI compatibility)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Markdown file")
    args = parser.parse_args()

    textstat.set_lang("de")

    # Hardware Check
    num_gpus = torch.cuda.device_count()
    print(f"\n[Hardware Check] Detected {num_gpus} GPU(s).")
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB VRAM)")

    print(f"\nLoading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    dataset = dataset[:20]

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

    # ========================================================
    # STAGE 1: BASELINE MODEL EVALUATION
    # ========================================================
    print(f"\n--- INITIALIZING BASE MODEL CONFIGURATION ---")
    
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.padding_side = "left"  # Crucial for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_prompts = [
        tokenizer.apply_chat_template(data["input_messages"], tokenize=False, add_generation_prompt=True)
        for data in evaluation_data
    ]

    loading_kwargs = get_model_loading_kwargs(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **loading_kwargs)

    print(f"\n🚀 Executing native batch generation for Baseline...")
    baseline_results = generate_batched(base_model, tokenizer, formatted_prompts, batch_size=4, desc="Evaluating Baseline")

    # ========================================================
    # STAGE 2: TUNED ADAPTER
    # ========================================================
    print(f"\n--- ATTACHING ADAPTER VIA PEFT: {args.adapter_path} ---")
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    print(f"\n🚀 Executing native batch generation for Tuned Adapter...")
    tuned_results = generate_batched(adapter_model, tokenizer, formatted_prompts, batch_size=4, desc="Evaluating Tuned Adapter")

    print("\n[Wiping Memory for Challenger Model...]")
    del adapter_model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================
    # STAGE 3: CHALLENGER MODEL
    # ========================================================
    print(f"\n--- INITIALIZING CHALLENGER MODEL CONFIGURATION ---")
    
    chal_tokenizer = AutoTokenizer.from_pretrained(CHALLENGER_BASE)
    chal_tokenizer.padding_side = "left"
    if chal_tokenizer.pad_token is None:
        chal_tokenizer.pad_token = chal_tokenizer.eos_token

    chal_formatted_prompts = [
        chal_tokenizer.apply_chat_template(data["input_messages"], tokenize=False, add_generation_prompt=True)
        for data in evaluation_data
    ]

    chal_loading_kwargs = get_model_loading_kwargs(CHALLENGER_BASE)
    chal_base = AutoModelForCausalLM.from_pretrained(CHALLENGER_BASE, **chal_loading_kwargs)
    chal_model = PeftModel.from_pretrained(chal_base, CHALLENGER_ADAPTER)

    print(f"\n🚀 Executing native batch generation for Challenger...")
    challenger_results = generate_batched(chal_model, chal_tokenizer, chal_formatted_prompts, batch_size=4, desc="Evaluating Challenger")

    del chal_model
    del chal_base
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
        
        m_orig = get_metrics_str(orig_text)
        m_ref = get_metrics_str(ref_text)
        m_base = get_metrics_str(base_text)
        m_tuned = get_metrics_str(tuned_text)
        m_chal = get_metrics_str(chal_text)

        entry_md = f"""## Entry {i + 1}

### Original (*{m_orig}*)
```text
{orig_text}
```

### Reference Solution (*{m_ref}*)
```text
{data['ref_text']}
```

### {args.base_model} (*{m_base}*)
```text
{base_text}
```

### fine-tuned {args.base_model} (*{m_tuned}*)
```text
{tuned_text}
```

### {CHALLENGER_BASE} tuned with {CHALLENGER_ADAPTER} (*{m_chal}*)
```text
{chal_text}
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