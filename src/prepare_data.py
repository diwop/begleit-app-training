import json
import os

def load_and_format_data(input_file: str, output_file: str) -> list:
    """
    Loads a JSONL file and formats it into Axolotl's ShareGPT format, saving it to output_file.
    Returns the loaded structured data.
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
            
            conversations = []
            if system:
                conversations.append({"from": "system", "value": system})
            if user:
                conversations.append({"from": "human", "value": user})
            if assistant:
                conversations.append({"from": "gpt", "value": assistant})
                
            if conversations:
                formatted_data.append({"conversations": conversations})
                
    with open(output_file, "w", encoding="utf-8") as out_f:
        for item in formatted_data:
            out_f.write(json.dumps(item) + "\n")
            
    return formatted_data

if __name__ == "__main__":
    output_path = "data/axolotl_dataset.jsonl"
    ds = load_and_format_data("data/sample_dataset.jsonl", output_path)
    print(f"Loaded {len(ds)} examples. Saved to {output_path}")
    print("Sample:", ds[0])
