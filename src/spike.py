# --- src/spike.py ---
import os
import gc
import sys
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import destroy_model_parallel

def run_model_spike(model_id, quantization_type, max_len=4096):
    """
    A modular function that isolates engine initialization, execution,
    and complete VRAM evacuation for a sequential model evaluation.
    """
    print("\n" + "="*60)
    print(f"🚀 INITIALIZING ENGINE PIPELINE: {model_id}")
    print(f"📟 Mode: {quantization_type or 'Unquantized (16-bit)'} | Context: {max_len}")
    print("="*60, flush=True)
    
    try:
        # 1. Initialize the specific multi-GPU vLLM engine instance
        llm = LLM(
            model=model_id,
            quantization=quantization_type,
            tensor_parallel_size=2,          # Slices layers across both L40S cards
            max_model_len=max_len,           # Conservative limits to shield sequential memory slots
            trust_remote_code=True,
            disable_custom_all_reduce=True,  # Mandatory fix for virtualized network deadlocks
            enforce_eager=True,              # Mandatory bypass for graph compilation errors
            gpu_memory_utilization=0.82      # Protects against cross-run memory fragmentation
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 2. Build Chat Structures (Gemma 4 & Llama 3.1 support native system tags)
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
        
        print(f"\n⚡ Processing token generation loop...", flush=True)
        outputs = llm.generate([full_templated_string], sampling_params)
        
        print(f"\n🧪 [RESULTS FOR {model_id}]:")
        print(f"{outputs[0].outputs[0].text.strip()}", flush=True)
        
    except Exception as e:
        print(f"\n❌ Execution error encountered on {model_id}: {e}", flush=True)
        
    finally:
        # 3. THE CRITICAL SECTOR: Force complete multi-GPU process and VRAM evacuation
        print(f"\n♻️ Deconstructing engine and evacuating VRAM channels...", flush=True)
        
        # Shutdown background process managers within the active vLLM V1 core
        try:
            if 'llm' in locals() and hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
                llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
            
        # Terminate cross-GPU tensor parallel coordination channels
        try:
            destroy_model_parallel()
        except Exception:
            pass
            
        # Evacuate local pointer identities
        if 'llm' in locals():
            del llm
        if 'tokenizer' in locals():
            del tokenizer
            
        # Flush garbage collection layers and clear GPU registers
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ VRAM completely evacuated. Ready for next slot.", flush=True)

def main():
    # Multi-Model Pipeline Sequence Matrix
    # Configuration Layout: (HuggingFace Model ID, Quantization Flag, Max Context Length)
    EVALUATION_PIPELINE = [
        (
            "cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", 
            "compressed-tensors", 
            4096
        ),
        # (
        #     "cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit", 
        #     "awq", 
        #     4096
        # ),
        # (
        #     "meta-llama/Llama-3.1-8B-Instruct", 
        #     None, # Passing None tells vLLM to run in full unquantized 16-bit
        #     4096
        # )
    ]
    
    print(f"🎬 Starting pipeline matrix execution ({len(EVALUATION_PIPELINE)} models registered)...")
    
    for model_id, quant_type, max_len in EVALUATION_PIPELINE:
        run_model_spike(model_id, quant_type, max_len)
        
    print("\n🏁 Complete pipeline sequence processed successfully!")

if __name__ == "__main__":
    main()