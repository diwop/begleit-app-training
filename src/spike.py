# --- src/spike.py ---
import os
import gc
import sys
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest # Mandatory for hot-swapping adapters
from vllm.distributed.parallel_state import destroy_model_parallel

def run_model_spike(model_id, quantization_type, max_len=4096, adapter_id=None):
    """
    A modular function that isolates engine initialization, optional LoRA injection,
    execution, and complete VRAM evacuation for sequential model testing.
    """
    print("\n" + "="*60)
    print(f"🚀 INITIALIZING ENGINE PIPELINE: {model_id}")
    if adapter_id:
        print(f"🧬 Injecting Active Adapter: {adapter_id}")
    print(f"📟 Mode: {quantization_type or 'Unquantized (16-bit)'} | Context: {max_len}")
    print("="*60, flush=True)
    
    try:
        # 1. Build Base Configuration Arguments
        llm_kwargs = {
            "model": model_id,
            "quantization": quantization_type,
            "tensor_parallel_size": 2,          # Slices layers across both L40S cards
            "max_model_len": max_len,           # Protects memory pool sizes
            "trust_remote_code": True,
            "disable_custom_all_reduce": True,  # Fix for virtualized network deadlocks
            "enforce_eager": True,              # Bypass for graph compilation deadlocks
            "gpu_memory_utilization": 0.82      # Mitigates memory fragmentation across steps
        }
        
        # 2. Dynamically allocate LoRA matrix space only if an adapter is passed
        if adapter_id:
            print("⚙️ Pre-allocating runtime slots for dynamic adapter routing...")
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            
        # Initialize the engine instance
        llm = LLM(**llm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 3. Formulate the Evaluation Prompt (Native System Tag Compatible)
        messages = [
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": "Warum ist der Himmel blau?"}
        ]
        
        full_templated_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            max_tokens=4096
        )
        
        # 4. Configure Generation Directives
        generate_kwargs = {}
        if adapter_id:
            # Instantiate the tracking pointer for the dynamic adapter injection layer
            # Parameters: (Reference Alias, Unique Numeric ID, Repo ID or Local Path)
            lora_request = LoRARequest("spike_adapter_layer", 1, adapter_id)
            generate_kwargs["lora_request"] = lora_request
            
        print(f"\n⚡ Processing token generation loop...", flush=True)
        outputs = llm.generate([full_templated_string], sampling_params, **generate_kwargs)
        
        print(f"\n🧪 [RESULTS FOR {model_id}]:")
        print(f"{outputs[0].outputs[0].text.strip()}", flush=True)
        
    except Exception as e:
        print(f"\n❌ Execution error encountered on {model_id}: {e}", flush=True)
        
    finally:
        # 5. THE CLEANUP SECTOR: Purges process and VRAM states for the next run
        print(f"\n♻️ Deconstructing engine and evacuating VRAM channels...", flush=True)
        
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
        print("✅ VRAM completely evacuated. Ready for next slot.", flush=True)

def main():
    # Expanded Multi-Model Pipeline Sequence Matrix
    # Layout Schema: (HuggingFace Model ID, Quantization Flag, Max Context Length, Optional Adapter ID)
    EVALUATION_PIPELINE = [
        # (
        #     "cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", 
        #     "compressed-tensors", 
        #     4096,
        #     None  # No adapter required
        # ),
        (
            "cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit", 
            "awq", 
            4096,
            None  # No adapter required
        ),
        # (
        #     "meta-llama/Llama-3.1-8B-Instruct", 
        #     None, # Full unquantized 16-bit precision mode
        #     4096,
        #     "tschomacker/lora_adapter_llama_3.1_8B" # Dynamic adapter injection target
        # )
    ]
    
    print(f"🎬 Starting pipeline matrix execution ({len(EVALUATION_PIPELINE)} models registered)...")
    
    for model_id, quant_type, max_len, adapter_id in EVALUATION_PIPELINE:
        run_model_spike(model_id, quant_type, max_len, adapter_id)
        
    print("\n🏁 Complete pipeline sequence processed successfully!")

if __name__ == "__main__":
    main()