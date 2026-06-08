# --- src/spike.py ---
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # 1. Test Pipeline Parameter Definition
    MODEL_ID = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
    SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent."
    USER_PROMPTS = [
        "Warum ist der Himmel blau?"
    ]
    
    print(f"📦 Booting vLLM Engine on top of Axolotl Environment: {MODEL_ID}")
    
    # 2. Bind Engine Weights Natively Into VRAM
    # Sets tensor_parallel_size=2 to evenly slice the 24GB matrix across both L40S cards
    llm = LLM(
        model=MODEL_ID,
        quantization="awq",       # Natively processes legacy fused AWQ layouts smoothly
        tensor_parallel_size=2,   # Distributes computing tensor workloads across both GPUs
        max_model_len=2048,
        trust_remote_code=True
    )
    
    # 3. Pull Tokenizer to compile Mistral chat templates
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 4. Enforce Token Generation Constraints
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=256
    )
    
    # 5. Structure Inputs to preserve the global System Prompt context
    formatted_payloads = []
    for user_query in USER_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
        # Parses the chat dictionary into standard target strings
        full_templated_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_payloads.append(full_templated_string)
        
    print("\n⚡ Processing execution batch sequence...")
    outputs = llm.generate(formatted_payloads, sampling_params)
    
    # 6. Output Generation Metrics
    print("\n--- 🧪 SPIKE SYSTEM RESULTS ---")
    for output in outputs:
        print(f"Output response:\n{output.outputs[0].text.strip()}")
    print("--------------------------------\n")

if __name__ == "__main__":
    main()