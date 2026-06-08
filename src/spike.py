# --- src/spike.py ---
import os
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # 1. Target the 4-bit AWQ quantized Mistral Small 4 flagship model
    MODEL_ID = "cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit"
    SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent."
    USER_PROMPTS = ["Warum ist der Himmel blau?"]
    
    print(f"📦 Booting 119B MoE Architecture: {MODEL_ID}", flush=True)
    print(f"📟 Allocating 128 experts evenly across 2x L40S GPUs...", flush=True)
    
    # 2. Configure the Engine for the 60GB Model Footprint
    llm = LLM(
        model=MODEL_ID,
        quantization="compressed-tensors",
        tensor_parallel_size=2,       # Splits the 119B parameter footprint across both cards
        max_model_len=8192,           # Plenty of context headroom for document digestion
        trust_remote_code=True,
        disable_custom_all_reduce=True, # Bypasses broken custom peer-to-peer cloud memory links
        enforce_eager=True,           # Bypasses CUDA graph compilation for direct execution stability
        gpu_memory_utilization=0.92   # Maximize allocation safety for the KV cache pool
    )
    
    # 3. Pull the native tokenizer supporting Mistral Small 4 rules
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 4. Clean, Native Multi-Role Structural Payloads (No more hacks required!)
    formatted_payloads = []
    for user_query in USER_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
        
        # Mistral Small 4 cleanly wraps system tokens into native structural block boundaries
        full_templated_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_payloads.append(full_templated_string)
        
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=256
    )
    
    print("\n⚡ Processing token generation sequence...", flush=True)
    outputs = llm.generate(formatted_payloads, sampling_params)
    
    # 5. Flush and Output Results
    print("\n===============================================", flush=True)
    print("🧪 MISTRAL SMALL 4 GENERATION RESULTS", flush=True)
    print("===============================================", flush=True)
    
    for output in outputs:
        print(f"Generated Response:\n{output.outputs[0].text.strip()}", flush=True)
        
    print("===============================================", flush=True)
    
    sys.stdout.flush()

if __name__ == "__main__":
    main()