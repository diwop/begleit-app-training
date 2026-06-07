# This script downloads large language models from HuggingFace, quantizes them to 4-bit precision, and uploads them to an S3 bucket.
# It is used to pre-build the models for fine-tuning, evaluation and production.
# To run this script, start the container with command "bash /runner/repo/quantize.sh".
#
# Please note: This setup needs 350 GB of disk space and lots of VRAM, e.g. 6x L40S with 48 GB each.

import os
import shutil
import gc
import torch
import boto3
from botocore.exceptions import ClientError
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODELS_TO_PROCESS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-4-26B-A4B",
    "mistralai/Mistral-Small-4-119B-2603", 
    # TODO: "meta-llama/Llama-3.1-8B",
]

# Fix PyTorch memory fragmentation globally
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Route all raw downloads to the mounted folder
HF_CACHE_DIR = "/app/huggingface_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR

# Safe fallback for the S3 Bucket environment variable
s3_bucket = os.environ.get("S3_BUCKET", "runpod-leichte-sprache")
S3_BASE_PREFIX = "models/awq-4bit/" 
LOCAL_EXPORT_DIR = "/app/export_temp"

def check_s3_model_exists(s3_client, s3_prefix: str) -> bool:
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=f"{s3_prefix}config.json")
        return True
    except ClientError:
        return False

def upload_to_s3(s3_client, local_dir: str, s3_prefix: str):
    print(f"   📤 Uploading perfectly compressed model to s3://{s3_bucket}/{s3_prefix}...")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path)
            s3_client.upload_file(local_path, s3_bucket, s3_key)
    print("   ✅ S3 upload complete.")

def clear_system_resources():
    print("   🧹 Sweeping VRAM and Python Garbage Collector...")
    gc.collect()
    torch.cuda.empty_cache()
    if os.path.exists(HF_CACHE_DIR):
        shutil.rmtree(HF_CACHE_DIR)
    if os.path.exists(LOCAL_EXPORT_DIR):
        shutil.rmtree(LOCAL_EXPORT_DIR)

def run_smoke_test(model, tokenizer):
    print("\n   💨 Running Sanity Smoke Test...")
    prompt = "Warum ist der Himmel blau?"
    
    try:
        # Try to format it perfectly for the specific instruct model
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to raw text if no chat template is found
        text = prompt
        
    # Push the input to the GPU
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate the response
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=100)
        
    # Isolate and decode just the new generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("   --- 🧪 SMOKE TEST OUTPUT ---")
    print(f"   {result.strip()}")
    print("   ----------------------------\n")        

def process_model(model_id: str, s3_client):
    safe_name = model_id.replace("/", "_")
    s3_prefix = f"{S3_BASE_PREFIX}{safe_name}/"
    
    print(f"\n=======================================================")
    print(f"🎯 PROCESSING: {model_id}")
    print(f"=======================================================")
    
    if check_s3_model_exists(s3_client, s3_prefix):
        print(f"   ✨ Model already exists in S3. Skipping.")
        return

    print(f"   ⬇️ Downloading raw 16-bit weights to VRAM...")
    
    # Define the true AOT 4-bit quantization config
    quant_config = {
        "zero_point": True, 
        "q_group_size": 128, 
        "w_bit": 4, 
        "version": "GEMM" 
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id, 
        safetensors=True, 
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("   ⚙️ Compiling true 4-bit AWQ model. This takes heavy GPU math...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    run_smoke_test(model, tokenizer)
    
    print(f"   💾 Saving pristine, compact 4-bit weights to disk...")
    os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)
    # Notice we use save_quantized() instead of save_pretrained()
    model.save_quantized(LOCAL_EXPORT_DIR)
    tokenizer.save_pretrained(LOCAL_EXPORT_DIR)
    
    upload_to_s3(s3_client, LOCAL_EXPORT_DIR, s3_prefix)
    
    print("   🗑️ Unloading model from GPUs...")
    del model
    del tokenizer
    clear_system_resources()
    print(f"   ✅ Successfully processed and stored {model_id}")

def main():
    s3_client = boto3.client('s3')
    clear_system_resources()
    
    for model_id in MODELS_TO_PROCESS:
        try:
            process_model(model_id, s3_client)
        except Exception as e:
            print(f"   🚨 FAILED to process {model_id}. Error: {str(e)}")
            clear_system_resources()

if __name__ == "__main__":
    main()