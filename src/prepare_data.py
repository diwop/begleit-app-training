import json
from datasets import Dataset

def load_and_format_data(file_path: str) -> Dataset:
    """
    Loads a JSONL file and formats it using ChatML-like structure.
    """
    formatted_data = {"text": []}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            system = entry.get("system", "")
            user = entry.get("user", "")
            assistant = entry.get("assistant", "")
            
            # Use ChatML formatting (approximate)
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
            
            if prompt.strip():
                formatted_data["text"].append(prompt)
                
    return Dataset.from_dict(formatted_data)

if __name__ == "__main__":
    ds = load_and_format_data("data/sample_dataset.jsonl")
    print(f"Loaded {len(ds)} examples.")
    print("Sample:", ds[0])
