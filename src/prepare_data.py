import json
import os

def load_and_format_data(input_file: str, output_file: str) -> list:
    """
    Loads a JSONL file and formats it into the modern OpenAI Chat format
    (required by Axolotl's chat_template), saving it to output_file.
    """
    formatted_data = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            system = entry.get("system", "")
            user = entry.get("user", "")
            assistant = entry.get("assistant", "")
            
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if user:
                messages.append({"role": "user", "content": user})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
                
            if messages:
                formatted_data.append({"messages": messages})
                
    with open(output_file, "w", encoding="utf-8") as out_f:
        for item in formatted_data:
            out_f.write(json.dumps(item) + "\n")
            
    return formatted_data

if __name__ == "__main__":
    output_path = "data/axolotl_dataset.jsonl"
    ds = load_and_format_data("data/sample_dataset.jsonl", output_path)
    print(f"Loaded {len(ds)} examples. Saved to {output_path}")
    print("Sample:", ds[0])