# --- src/spike.py ---
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # 1. Base Spike Parameters
    MODEL_ID = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
    SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent."
    USER_PROMPTS = ["Warum ist der Himmel blau?"]
    
    print(f"📦 Booting vLLM Engine on top of Axolotl Environment: {MODEL_ID}")
    
    # 2. Map Weights Directly Into VRAM Across Both Cards
    llm = LLM(
        model=MODEL_ID,
        # OPTIMIZATION: Upgraded back to Marlin for flawless weight unpacking on Ada Lovelace GPUs
        quantization="awq_marlin",       
        tensor_parallel_size=2,   
        max_model_len=8192, 
        trust_remote_code=True,
        disable_custom_all_reduce=True, 
        # OPTIMIZATION: Keeps CUDA graphs disabled so Marlin will bypass profiling and never freeze
        enforce_eager=True        
    )
    
    # 3. Pull Tokenizer to cleanly build Mistral chat structures
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 4. Enforce Token Output Constraints
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=256
    )
    
    # 5. Build and format payloads with clear structural role separation
    formatted_payloads = []
    for user_query in USER_PROMPTS:
        # OPTIMIZATION: Inject explicit markdown boundaries within the single turn
        combined_content = f"System: {SYSTEM_PROMPT}\n\nUser: {user_query}"
        
        messages = [
            {"role": "user", "content": combined_content}
        ]
        
        full_templated_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_payloads.append(full_templated_string)
        
    print("\n⚡ Processing token generation sequence...", flush=True)
    outputs = llm.generate(formatted_payloads, sampling_params)
    
    # 6. Output Evaluation Results with Mandatory Stream Flushes
    print("\n===============================================", flush=True)
    print("🧪 SPIKE SYSTEM RESULTS (FORCED FLUSH)", flush=True)
    print("===============================================", flush=True)
    
    for output in outputs:
        prompt_query = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"\nPrompt Context Passed:\n{prompt_query}", flush=True)
        print(f"\nGenerated Response:\n{generated_text.strip()}", flush=True)
        
    print("\n===============================================", flush=True)
    
    import sys
    sys.stdout.flush()

if __name__ == "__main__":
    main()