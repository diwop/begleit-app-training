# --- src/spike.py ---
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # 1. Pipeline Definition
    MODEL_ID = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
    SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent."
    USER_PROMPTS = ["Warum ist der Himmel blau?"]
    
    print(f"📦 Booting vLLM Engine on top of: {MODEL_ID}")
    
    # 2. Initialize Engine Weights Natively Into VRAM
    # We allocate tensor_parallel_size=2 to split across both L40S cards perfectly.
    llm = LLM(
        model=MODEL_ID,
        quantization="awq",       # vLLM handles offline static AWQ weights with max efficiency
        tensor_parallel_size=2,   # Distributes computing workload across your 2 GPUs
        max_model_len=2048,
        trust_remote_code=True
    )
    
    # 3. Pull Tokenizer to cleanly build out Mistral chat structures
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 4. Enforce Token Output Configurations
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=4096
    )
    
    # 5. Build and format payloads with consistent system prompts
    formatted_payloads = []
    for user_query in USER_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
        # Generates exact prompt formatting structural bounds [INST] ... [/INST]
        full_templated_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_payloads.append(full_templated_string)
        
    print("\n⚡ Processing token generation sequence...")
    outputs = llm.generate(formatted_payloads, sampling_params)
    
    # 6. Output Evaluation Results
    print("\n--- 🧪 SPIKE SYSTEM RESULTS ---")
    for output in outputs:
        print(f"Output response:\n{output.outputs[0].text.strip()}")
    print("--------------------------------\n")

if __name__ == "__main__":
    main()