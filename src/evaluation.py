# --- src/evaluation.py ---
import os
import gc
import sys
import json
from datetime import datetime, UTC
from pathlib import Path
import torch
import boto3
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel
import textstat

def get_raw_metrics(text: str) -> tuple:
    """Calculates German textstat metrics and returns rounded raw floats."""
    if not text.strip():
        return 0.0, 0.0
    fre = round(textstat.flesch_reading_ease(text), 1)
    wstf = round(textstat.wiener_sachtextformel(text, 1), 1)
    return fre, wstf

def read_file_with_extensions(base_path_str: str, extensions=[".txt", ".md"]) -> str:
    """Checks for the existence of a file across multiple extensions. Raises Error if absent."""
    for ext in extensions:
        full_path = f"{base_path_str}{ext}"
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read().strip()
                
    raise FileNotFoundError(
        f"❌ Data Integrity Violation: Required file not found for base path '{base_path_str}' "
        f"with extensions {extensions}."
    )

def run_model_spike(model_id, quantization_type, max_len=8192, adapter_id=None, evaluation_set=None):
    """Initializes vLLM engine, injects active adapters, handles batch evaluation sequences."""
    if evaluation_set is None:
        evaluation_set = []
        
    print("\n" + "="*60)
    print(f"🚀 LOADING MODEL FOR BATCH EVALUATION: {model_id}")
    if adapter_id:
        print(f"🧬 Hooking Active Local Adapter: {adapter_id}")
    else:
        print("💡 Running in Vanilla Baseline Mode (No Adapter)")
    print("="*60, flush=True)
    
    generated_responses = []
    
    try:
        # DYNAMIC HARDWARE DETECTION
        # Automatically scales tensor parallelism to match available GPUs (e.g., 4)
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Fallback security check to ensure it adheres to the strict power-of-2 rule
        if available_gpus not in [1, 2, 4, 8]:
            print(f"⚠️ Warning: Asymmetrical GPU count ({available_gpus}) detected. Falling back to 2.")
            available_gpus = 2

        llm_kwargs = {
            "model": model_id,
            "quantization": quantization_type,
            "tensor_parallel_size": available_gpus,
            "max_model_len": max_len,
            "trust_remote_code": True,
            "disable_custom_all_reduce": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.82
        }
        
        # Enforce text-only parameters for Mistral multimodal models to avoid profiling crashes
        if "mistral" in model_id.lower():
            print("🛑 Disabling vision modalities for text-only Mistral inference sequence...")
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
            
        if adapter_id:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            
        llm = LLM(**llm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        templated_inputs = []
        for prompt_item in evaluation_set:
            messages = [
                {"role": "system", "content": prompt_item["system"]},
                {"role": "user", "content": prompt_item["templated_user"]}
            ]
            full_string = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            templated_inputs.append(full_string)
            
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            max_tokens=4096
        )
        
        generate_kwargs = {}
        if adapter_id:
            lora_request = LoRARequest("spike_adapter_layer", 1, adapter_id)
            generate_kwargs["lora_request"] = lora_request
            
        print(f"⚡ Processing {len(templated_inputs)} prompts in parallel execution...", flush=True)
        outputs = llm.generate(templated_inputs, sampling_params, **generate_kwargs)
        
        for out in outputs:
            generated_responses.append(out.outputs[0].text.strip())
            
    except Exception as e:
        print(f"❌ Execution error encountered on {model_id}: {e}", flush=True)
        generated_responses = ["" for _ in evaluation_set]
        
    finally:
        print(f"♻️ Evacuating VRAM channels for next model tracking...", flush=True)
        try:
            if 'llm' in locals() and hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
                llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
            
        try:
            destroy_model_parallel()
        except Exception:
            pass
            
        if 'llm' in locals():
            del llm
        if 'tokenizer' in locals():
            del tokenizer
            
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ VRAM cleared.", flush=True)
        
    return generated_responses

def main():
    print("📋 Validating structural configurations...", flush=True)
    if not os.path.exists("data/system-prompt.md"):
        raise FileNotFoundError("❌ Pipeline Failure: Configuration file 'data/system-prompt.md' is missing.")
    if not os.path.exists("data/prompt-template.md"):
        raise FileNotFoundError("❌ Pipeline Failure: Configuration file 'data/prompt-template.md' is missing.")

    with open("data/system-prompt.md", "r", encoding="utf-8") as f:
        global_system_prompt = f.read().strip()
        
    with open("data/prompt-template.md", "r", encoding="utf-8") as f:
        global_template = f.read()

    evaluation_set = []
    
    # PART 1: APPEND INTEGRITY CHECK PROMPT
    evaluation_set.append({
        "is_integrity": True,
        "original_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "templated_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "system": "Du bist ein hilfreicher Assistent",
        "reference_text": None
    })
    
    # PART 2: APPEND REGULAR UNTAGGED TEST PROMPTS
    original_test_prompts = [
        "Warum ist der Himmel blau und nicht schwarz?",
        "# Magdeburg bundesweit vorn bei Hausärztinnen\n\nNirgendwo in Deutschland ist der Frauenanteil bei den Hausärzten so hoch wie in Magdeburg. Hausärztinnen haben in der Landeshauptstadt einen Anteil von 77,5 Prozent, wie aus einer Auswertung der Kassenärztlichen Bundesvereinigung (KBV) hervorgeht. Auf Magdeburg folgen in den Top 3 der Ilm-Kreis (76,2 Prozent) und das Altenburger Land (74,1 Prozent) in Thüringen.\n\nIn Sachsen-Anhalt insgesamt liegt der Anteil der Ärztinnen bei 58,7 Prozent und damit im bundesweiten Vergleich recht hoch. Berlin hat den höchsten Ärztinnen-Anteil mit 60,2 Prozent, gefolgt von Hamburg und Sachsen mit je 58,9 Prozent. Schlusslicht ist das Saarland mit 47,5 Prozent."
    ]
    
    for text_block in original_test_prompts:
        evaluation_set.append({
            "is_integrity": False,
            "original_user": text_block,
            "templated_user": global_template.replace("%INPUT%", text_block),
            "system": global_system_prompt,
            "reference_text": None
        })

    # PART 3: INGEST ADAPTER DATASET FROM JSONL
    jsonl_path = "data/train/dataset.jsonl"
    if os.path.exists(jsonl_path):
        print(f"📥 Parsing tuning records from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                prompt_id = str(entry.get("id", "")).strip()
                if not prompt_id:
                    raise KeyError(f"❌ Dataset Corruption: Missing 'id' field in {jsonl_path} on line {line_number}.")
                
                orig_base_path = f"data/raw/{prompt_id}_Standardsprache"
                ref_base_path = f"data/raw/{prompt_id}_Leichte_Sprache"
                
                orig_content = read_file_with_extensions(orig_base_path)
                ref_content = read_file_with_extensions(ref_base_path)
                
                evaluation_set.append({
                    "is_integrity": False,
                    "original_user": orig_content,
                    "templated_user": global_template.replace("%INPUT%", orig_content),
                    "system": global_system_prompt,
                    "reference_text": ref_content
                })

    # --- DYNAMIC ADAPTER CHECKPOINT RESOLUTION ---
    mistral_adapter = "/app/output/adapter/mistral4small"
    if not os.path.exists(os.path.join(mistral_adapter, "adapter_config.json")):
        mistral_adapter = "/app/output/adapter/train-mistral4small"
        
    gemma_adapter = "/app/output/adapter/gemma4"
    if not os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        gemma_adapter = "/app/output/adapter/train-gemma4"

    # REFACTORED MATRIX: Dynamically add baseline and adapter entries side-by-side
    EVALUATION_PIPELINE = []

    # A. Mistral Small 4 Matrix Layers
    EVALUATION_PIPELINE.append(("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", "compressed-tensors", 8192, None))
    if os.path.exists(os.path.join(mistral_adapter, "adapter_config.json")):
        EVALUATION_PIPELINE.append(("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", "compressed-tensors", 8192, mistral_adapter))

    # B. Gemma 4 Matrix Layers
    EVALUATION_PIPELINE.append(("cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit", "compressed-tensors", 8192, None))
    if os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        EVALUATION_PIPELINE.append(("cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit", "compressed-tensors", 8192, gemma_adapter))

    # C. Llama 3.1 Schomacker et. al. Challenger
    EVALUATION_PIPELINE.append(("meta-llama/Llama-3.1-8B-Instruct", None, 8192, "tschomacker/lora_adapter_llama_3.1_8B"))
    
    output_json = {
        "system": global_system_prompt,
        "template": global_template,
        "models": [],
        "prompts": []
    }
    
    for item in evaluation_set:
        record = {}
        if item["is_integrity"]:
            record["system"] = item["system"]
            record["template"] = ""
            
        input_fre, input_wstf = get_raw_metrics(item["original_user"])
        record["r"] = [[item["original_user"], input_fre, input_wstf]]
        output_json["prompts"].append(record)

    # Cascade Batch Inference through registered models
    for model_id, quant_type, max_len, adapter_id in EVALUATION_PIPELINE:
        display_name = model_id
        if adapter_id:
            display_name += f" ({adapter_id})"
        output_json["models"].append(display_name)
        
        responses = run_model_spike(model_id, quant_type, max_len, adapter_id, evaluation_set)
        
        for idx, text_response in enumerate(responses):
            resp_fre, resp_wstf = get_raw_metrics(text_response)
            output_json["prompts"][idx]["r"].append([text_response, resp_fre, resp_wstf])

    # POST-PROCESSING: Append tuning ground truth references
    print("\n📝 Appending ground-truth training references to dataset records...", flush=True)
    for idx, item in enumerate(evaluation_set):
        if item["reference_text"] is not None:
            ref_txt = item["reference_text"]
            ref_fre, ref_wstf = get_raw_metrics(ref_txt)
            output_json["prompts"][idx]["r"].append([ref_txt, ref_fre, ref_wstf])

    # Output Serialization and S3 Upload
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    bucket_name = os.environ.get("S3_BUCKET")
    json_payload = json.dumps(output_json, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60 + "\n🏁 EVALUATION MATRIX EXPORT\n" + "="*60)
    if bucket_name:
        print(f"📤 Exporting matrix results to S3 Bucket: {bucket_name} as {filename}...")
        try:
            s3_client = boto3.client('s3')
            s3_client.put_object(Bucket=bucket_name, Key=filename, Body=json_payload, ContentType='application/json')
            print("🚀 S3 Synchronization successful!")
        except Exception as s3_err:
            print(f"❌ Failed to transfer to S3 destination: {s3_err}\nSaving local copy as fallback...")
            with open(filename, "w", encoding="utf-8") as f: f.write(json_payload)
    else:
        print("⚠️ S3_BUCKET environment variable missing. Saving locally...")
        with open(filename, "w", encoding="utf-8") as f: f.write(json_payload)
        print(f"💾 Written to local file system: {filename}")

if __name__ == "__main__":
    main()