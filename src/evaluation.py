import argparse
import json
import gc
import os
import torch
import textstat
import time
import subprocess
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AwqConfig
from peft import PeftModel

# Hardcoded Challenger Variables
CHALLENGER_BASE = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
CHALLENGER_ADAPTER = "tschomacker/lora_adapter_llama_3.1_8B"

def get_raw_metrics(text: str) -> tuple:
    """Calculates German textstat metrics and returns rounded raw floats."""
    if not text.strip():
        return 0.0, 0.0
    fre = round(textstat.flesch_reading_ease(text), 1)
    wstf = round(textstat.wiener_sachtextformel(text, 1), 1)
    return fre, wstf

def get_model_loading_kwargs(model_id: str) -> dict:
    """Dynamically determines backend loading arguments for quantization."""
    m_lower = model_id.lower()
    
    base_kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa", 
        "torch_dtype": torch.bfloat16  
    }
    
    # Intercept AWQ explicitly to stop the Marlin JIT compiler
    if "awq" in m_lower:
        print(f" -> AWQ model detected. Forcing GEMM backend to stop Marlin JIT...")
        base_kwargs["quantization_config"] = AwqConfig(
            bits=4,
            group_size=128,
            zero_point=True,
            version="gemm"  # <--- The Magic Bullet that bypasses Marlin
        )
        return base_kwargs
        
    # Handle standard pre-quantized BitsAndBytes models
    elif "bnb" in m_lower or "4bit" in m_lower:
        print(f" -> BNB pre-quantized model detected. Bypassing custom config...")
        return base_kwargs
        
    # Handle unquantized models (needs on-the-fly shrinking)
    else:
        print(f" -> Unquantized model detected. Shrinking to 4-Bit NF4 on the fly...")
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return base_kwargs

def calculate_optimal_batch_size(model, current_batch_max_tokens, safety_factor=0.7) -> int:
    """Dynamically calculates the optimal batch size based on remaining free VRAM."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 4
        
    free_vram, _ = torch.cuda.mem_get_info(device)
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    bytes_per_element = 2
    kv_cache_per_seq = 2 * num_layers * num_kv_heads * head_dim * current_batch_max_tokens * bytes_per_element
    
    usable_vram = free_vram * safety_factor
    optimal_batch_size = int(usable_vram // kv_cache_per_seq)
    return max(1, min(optimal_batch_size, 32))

def generate_batched(model, tokenizer, evaluation_data, formatted_prompts, desc="Generating"):
    """Sorts evaluation data by target length to perform variable-sized 

    bucket batching with dynamic, OOM-safe hardware utilization.
    """
    results_dict = {}
    
    # Sort data by reference token size to build optimal, contiguous length buckets
    sorted_indices = sorted(
        range(len(evaluation_data)), 
        key=lambda idx: len(tokenizer.encode(evaluation_data[idx]["reference"])),
        reverse=True
    )
    
    pbar = tqdm(total=len(evaluation_data), desc=desc, unit="sample")
    i = 0
    start_time = time.time()
    total_tokens_generated = 0
    
    while i < len(sorted_indices):
        sample_idx = sorted_indices[i]
        reference_tokens = len(tokenizer.encode(evaluation_data[sample_idx]["reference"]))
        
        # Allocate defensive 15% generation overhead ceiling buffer
        max_new_tokens_needed = max(128, int(reference_tokens * 1.15))
        
        # Recalculate OOM boundaries on the fly for the current bucket
        batch_size = calculate_optimal_batch_size(model, max_new_tokens_needed)
        batch_indices = sorted_indices[i : i + batch_size]
        batch_prompts = [formatted_prompts[idx] for idx in batch_indices]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_needed,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        for j, out in enumerate(outputs):
            global_idx = batch_indices[j]
            gen_tokens = out[inputs.input_ids.shape[1]:]
            total_tokens_generated += len(gen_tokens)
            results_dict[global_idx] = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            
        pbar.update(len(batch_indices))
        i += len(batch_indices)
        
    pbar.close()
    elapsed_time = time.time() - start_time
    tokens_per_sec = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0
    print(f"✨ [{desc}] Finished in {elapsed_time:.2f}s | Speed: {tokens_per_sec:.2f} tokens/sec\n")
    
    return [results_dict[k] for k in sorted(results_dict.keys())]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline, Custom LoRA, and Challenger Models emitting structured JSON databases.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or HF ID of the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--seq_length", type=int, default=16384, help="Context sequence length threshold")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--run_id", type=str, required=True, help="ClearML/Pipeline unique run execution identifier")
    args = parser.parse_args()

    textstat.set_lang("de")

    # Hardware Telemetry Check
    num_gpus = torch.cuda.device_count()
    print(f"\n[Hardware Check] Detected {num_gpus} GPU(s).")
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB VRAM)")

    # Read external markdown prompt values dynamically from repository paths
    repo_root = Path(__file__).resolve().parent.parent
    system_file = repo_root / "data" / "system-prompt.md"
    template_file = repo_root / "data" / "prompt-template.md"
    
    system_prompt = system_file.read_text(encoding="utf-8").strip() if system_file.exists() else ""
    template_prompt = template_file.read_text(encoding="utf-8").strip() if template_file.exists() else ""

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
    # STAGE 1: UNTUNED BASELINE MODEL
    # ========================================================
    print(f"\n--- INITIALIZING CORE BASE MODEL CONFIGURATION ---")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_prompts = [
        tokenizer.apply_chat_template(data["input_messages"], tokenize=False, add_generation_prompt=True)
        for data in evaluation_data
    ]

    args.base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    loading_kwargs = get_model_loading_kwargs(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **loading_kwargs)
    
    print(f"\n🚀 Executing native bucket batch generation for Baseline...")
    baseline_results = generate_batched(base_model, tokenizer, evaluation_data, formatted_prompts, desc="Evaluating Baseline")

    # ========================================================
    # STAGE 2: TUNED ADAPTER
    # ========================================================
    print(f"\n--- ATTACHING ADAPTER VIA PEFT: {args.adapter_path} ---")
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    print(f"\n🚀 Executing native bucket batch generation for Tuned Adapter...")
    tuned_results = generate_batched(adapter_model, tokenizer, evaluation_data, formatted_prompts, desc="Evaluating Tuned Adapter")

    print("\n[Wiping Memory for Core Models...]")
    del adapter_model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================
    # STAGE 3: CHALLENGER MODEL
    # ========================================================
    print(f"\n--- INITIALIZING CHALLENGER MODEL CONFIGURATION ---")
    chal_tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-instruct")
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

    print(f"\n🚀 Executing native bucket batch generation for Challenger...")
    challenger_results = generate_batched(chal_model, chal_tokenizer, evaluation_data, chal_formatted_prompts, desc="Evaluating Challenger")

    del chal_model
    del chal_base
    del chal_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # OUTPUT JSON STREAM COMPILATION
    # ==========================================
    print(f"\nCompiling structured JSON database...")
    json_output = {
        "system": system_prompt,
        "template": template_prompt,
        "entries": []
    }
    
    for i in range(len(evaluation_data)):
        data = evaluation_data[i]
        orig_text = data['original']
        ref_text = data['reference']
        base_text = baseline_results[i]
        tuned_text = tuned_results[i]
        chal_text = challenger_results[i]
        
        orig_fre, orig_swtf = get_raw_metrics(orig_text)
        ref_fre, ref_swtf = get_raw_metrics(ref_text)
        base_fre, base_swtf = get_raw_metrics(base_text)
        tuned_fre, tuned_swtf = get_raw_metrics(tuned_text)
        chal_fre, chal_swtf = get_raw_metrics(chal_text)

        # Matrix array row explicitly augmented with untuned baseline outputs
        json_output["entries"].append([
            orig_text, orig_fre, orig_swtf,
            base_text, base_fre, base_swtf,
            tuned_text, tuned_fre, tuned_swtf,
            chal_text, chal_fre, chal_swtf,
            ref_text, ref_fre, ref_swtf,
        ])

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"Local evaluation file saved to: {output_path}")

    # ==========================================
    # S3 SYNC AUTOMATION PHASE
    # ==========================================
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        s3_bucket = s3_bucket.strip().rstrip("/")
        s3_destination = f"s3://{s3_bucket}/evaluation_{args.run_id}.json"
        print(f"\n☁️ Syncing results to cloud storage: {s3_destination} ...")
        
        try:
            subprocess.run(
                ["aws", "s3", "cp", str(output_path), s3_destination],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✨ S3 copy operations successfully executed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to copy to S3. Error code: {e.returncode}")
            print(f"Reason: {e.stderr.decode('utf-8').strip()}")
    else:
        print("\nℹ️ Environment variable S3_BUCKET not set. Skipping cloud copy.")

    print("\n=== Evaluation pipeline successfully completed! ===")

if __name__ == "__main__":
    main()