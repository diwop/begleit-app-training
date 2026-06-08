# --- src/spike.py ---
import os
import gc
import sys
import json
from datetime import datetime
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

def run_model_spike(model_id, quantization_type, max_len=8192, adapter_id=None, evaluation_set=None):
    """
    Initializes the engine, handles runtime LoRA space allocation, batch processes
    the entire evaluation set without printing outputs, and returns raw texts.
    """
    if evaluation_set is None:
        evaluation_set = []
        
    print("\n" + "="*60)
    print(f"🚀 LOADING MODEL FOR BATCH EVALUATION: {model_id}")
    if adapter_id:
        print(f"🧬 Active Adapter: {adapter_id}")
    print("="*60, flush=True)
    
    generated_responses = []
    
    try:
        # 1. Build Base Configuration Arguments
        llm_kwargs = {
            "model": model_id,
            "quantization": quantization_type,
            "tensor_parallel_size": 2,          # Slices layers across both L40S cards
            "max_model_len": max_len,           # Expanded to 8192 for input/output headroom
            "trust_remote_code": True,
            "disable_custom_all_reduce": True,  # Fix for virtualized network deadlocks
            "enforce_eager": True,              # Bypass for graph compilation deadlocks
            "gpu_memory_utilization": 0.82      # Mitigates memory fragmentation across steps
        }
        
        # 2. Dynamically allocate LoRA matrix space only if an adapter is passed
        if adapter_id:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            
        # Initialize engine and tokenizer
        llm = LLM(**llm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 3. Apply Chat Templates across the entire evaluation sequence
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
            
        # 4. Generate Batched Reponses (Silent Collection)
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
        
        # Collect results sequentially
        for out in outputs:
            generated_responses.append(out.outputs[0].text.strip())
            
    except Exception as e:
        print(f"❌ Execution error encountered on {model_id}: {e}", flush=True)
        # Populate empty fallbacks on failure to maintain matrix alignment
        generated_responses = ["" for _ in evaluation_set]
        
    finally:
        # 5. Evacuate engine instances from the active VRAM pool
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
    # 1. Load System Prompts and Data Blueprints (with fallback defaults)
    os.makedirs("data", exist_ok=True)
    
    try:
        with open("data/system-prompt.md", "r", encoding="utf-8") as f:
            global_system_prompt = f.read().strip()
    except FileNotFoundError:
        global_system_prompt = "Du bist ein hilfreicher Assistent."
        with open("data/system-prompt.md", "w", encoding="utf-8") as f:
            f.write(global_system_prompt)
            
    try:
        with open("data/prompt-template.md", "r", encoding="utf-8") as f:
            global_template = f.read()
    except FileNotFoundError:
        global_template = "Hier ist der Text:\n\n%INPUT%"
        with open("data/prompt-template.md", "w", encoding="utf-8") as f:
            f.write(global_template)

    # 2. Assemble Evaluation Targets
    original_test_prompts = [
        "Warum ist der Himmel blau und nicht schwarz?",
        "# Magdeburg bundesweit vorn bei Hausärztinnen\n\nNirgendwo in Deutschland ist der Frauenanteil bei den Hausärzten so hoch wie in Magdeburg. Hausärztinnen haben in der Landeshauptstadt einen Anteil von 77,5 Prozent, wie aus einer Auswertung der Kassenärztlichen Bundesvereinigung (KBV) hervorgeht. Auf Magdeburg folgen in den Top 3 der Ilm-Kreis (76,2 Prozent) und das Altenburger Land (74,1 Prozent) in Thüringen.\n\nIn Sachsen-Anhalt insgesamt liegt der Anteil der Ärztinnen bei 58,7 Prozent und damit im bundesweiten Vergleich recht hoch. Berlin hat den höchsten Ärztinnen-Anteil mit 60,2 Prozent, gefolgt von Hamburg und Sachsen mit je 58,9 Prozent. Schlusslicht ist das Saarland mit 47,5 Prozent."
    ]
    
    # Internal Structured Tracking Matrix
    evaluation_set = []
    
    # A. Append Part 1: Integrity Prompt Entry
    evaluation_set.append({
        "is_integrity": True,
        "original_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "templated_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "system": "Du bist ein hilfreicher Assistent"
    })
    
    # B. Append Part 2: Wrapped Test Prompts
    for text_block in original_test_prompts:
        evaluation_set.append({
            "is_integrity": False,
            "original_user": text_block,
            "templated_user": global_template.replace("%INPUT%", text_block),
            "system": global_system_prompt
        })

    # 3. Define Pipeline Infrastructure Matrix
    EVALUATION_PIPELINE = [
        ("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", "compressed-tensors", 8192, None),
        ("cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit", "compressed-tensors", 8192, None),
        ("meta-llama/Llama-3.1-8B-Instruct", None, 8192, "tschomacker/lora_adapter_llama_3.1_8B")
    ]
    
    # Initialize Master Output Dictionary Architecture
    output_json = {
        "system": global_system_prompt,
        "template": global_template,
        "models": [],
        "prompts": []
    }
    
    # Pre-populate prompts and compute initial input metrics
    for item in evaluation_set:
        record = {}
        if item["is_integrity"]:
            record["system"] = item["system"]
            record["template"] = ""
            
        # Initial tuple setup: [Original Input Text, FRE, WSTF]
        input_fre, input_wstf = get_raw_metrics(item["original_user"])
        record["r"] = [
            [item["original_user"], input_fre, input_wstf]
        ]
        output_json["prompts"].append(record)

    # 4. Cascade Inference through registered models
    for model_id, quant_type, max_len, adapter_id in EVALUATION_PIPELINE:
        # Build clean string mapping identifying adapters if present
        display_name = model_id
        if adapter_id:
            display_name += f" ({adapter_id})"
        output_json["models"].append(display_name)
        
        # Run silent batch generation pass
        responses = run_model_spike(model_id, quant_type, max_len, adapter_id, evaluation_set)
        
        # Compute metrics for outputs and append directly to records
        for idx, text_response in enumerate(responses):
            resp_fre, resp_wstf = get_raw_metrics(text_response)
            output_json["prompts"][idx]["r"].append([text_response, resp_fre, resp_wstf])

    # 5. Output Serialization and S3 Shipments
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    
    bucket_name = os.environ.get("S3_BUCKET")
    json_payload = json.dumps(output_json, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("🏁 PIPELINE COMPLETED")
    print("="*60)
    
    if bucket_name:
        print(f"📤 Exporting results to S3 Bucket: {bucket_name} as {filename}...")
        try:
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=bucket_name,
                Key=filename,
                Body=json_payload,
                ContentType='application/json'
            )
            print("🚀 S3 Shipment processed successfully!")
        except Exception as s3_err:
            print(f"❌ Failed to transfer to S3 destination: {s3_err}")
            print("Writing data locally as fallback...")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_payload)
    else:
        print("⚠️ S3_BUCKET environment variable missing. Saving locally...")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_payload)
        print(f"💾 Written to local file system: {filename}")

if __name__ == "__main__":
    main()