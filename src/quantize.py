# This script downloads large language models from HuggingFace, quantizes them to 4-bit precision, and uploads them to an S3 bucket.
# It is used to pre-build the models for fine-tuning, evaluation and production.
# To run this script, start the container with command "bash /runner/repo/quantize.sh".

MODELS_TO_PROCESS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-4-26B-A4B",
    # TODO: Mistral-Small-Instruct
    # "meta-llama/Llama-3.1-8B", # for Schomacker et. al. challenger model
]

import os
import sys
import shutil
import gc
import torch
import boto3
from botocore.exceptions import ClientError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Fix PyTorch memory fragmentation globally
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Route all raw Hugging Face downloads to the mounted folder
HF_CACHE_DIR = "/app/huggingface_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR

s3_bucket = os.environ.get("S3_BUCKET")
S3_BASE_PREFIX = "models/bnb-4bit/"
LOCAL_EXPORT_DIR = "/app/export_temp"

def check_s3_model_exists(s3_client, s3_prefix: str) -> bool:
    """Checks if the model's config.json already exists in the S3 bucket."""
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=f"{s3_prefix}config.json")
        return True
    except ClientError:
        return False

def upload_to_s3(s3_client, local_dir: str, s3_prefix: str):
    """Recursively uploads a local directory to S3."""
    print(f"   Uploading to s3://{s3_bucket}/{s3_prefix}...")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Calculate the relative path to maintain folder structure
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path)
            
            s3_client.upload_file(local_path, s3_bucket, s3_key)
    print("   ✅ S3 upload complete.")

def clear_system_resources():
    """Aggressively clears VRAM and Disk Space to prepare for the next model."""
    print("   Sweeping VRAM and Python Garbage Collector...")
    gc.collect()
    torch.cuda.empty_cache()
    
    print("   Deleting raw 16-bit Hugging Face cache from disk...")
    if os.path.exists(HF_CACHE_DIR):
        shutil.rmtree(HF_CACHE_DIR)
        
    print("   Deleting local 4-bit export folder from disk...")
    if os.path.exists(LOCAL_EXPORT_DIR):
        shutil.rmtree(LOCAL_EXPORT_DIR)

def process_model(model_id: str, s3_client):
    # Create a safe S3 folder name (e.g., mistralai_Mixtral-8x7B-Instruct-v0.1)
    safe_name = model_id.replace("/", "_")
    s3_prefix = f"{S3_BASE_PREFIX}{safe_name}/"
    
    print(f"\nPROCESSING: {model_id}\n")
    
    if check_s3_model_exists(s3_client, s3_prefix):
        print(f"   Model already exists in S3 (s3://{s3_bucket}/{s3_prefix}). Skipping.")
        return

    print(f"   Not found in S3. Downloading and quantizing on-the-fly...")
    
    # 1. Define Quantization parameters
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True  # Allow offloading at edge of OOM
    )
    
    # 2. Download and load into VRAM
    # device_map="auto" lets it span across all available L40S GPUs naturally
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 3. Save the 4-bit model to local disk
    print(f"   Saving native 4-bit weights to {LOCAL_EXPORT_DIR}...")
    os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)
    model.save_pretrained(LOCAL_EXPORT_DIR)
    tokenizer.save_pretrained(LOCAL_EXPORT_DIR)
    
    # 4. Upload the crushed model to S3
    upload_to_s3(s3_client, LOCAL_EXPORT_DIR, s3_prefix)
    
    # 5. CRITICAL: Destroy the model object so PyTorch releases the VRAM
    print("   Unloading model from GPUs...")
    del model
    del tokenizer
    
    # 6. Purge the environment for the next iteration
    clear_system_resources()
    print(f"   ✅ Successfully processed and stored {model_id}")

def main():
    s3_client = boto3.client('s3')
    
    # Ensure a clean slate before starting
    clear_system_resources()
    
    for model_id in MODELS_TO_PROCESS:
        try:
            process_model(model_id, s3_client)
        except Exception as e:
            print(f"   FAILED to process {model_id}. Error: {str(e)}")
            # Even if it fails, we must clean the disk before the next model
            clear_system_resources()

if __name__ == "__main__":
    main()