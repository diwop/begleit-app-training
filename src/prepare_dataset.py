import os
import re
import json
from pathlib import Path

def read_file(path: Path) -> str:
    """Helper to read file contents cleanly."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    # 1. Define paths based on your dvc.yaml structure
    src_data_dir = Path("data/raw")
    output_file = Path("data/train/dataset.jsonl")
    system_prompt_path = Path("data/system-prompt.md")
    prompt_template_path = Path("data/prompt-template.md")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 2. Read global system prompt and user template
    system_prompt = read_file(system_prompt_path)
    prompt_template = read_file(prompt_template_path)

    # 3. Scan the src-data directory using regex to capture the exact ID string
    # Matches both .md and .txt variations
    pattern = re.compile(r"^(\d+)_(Standardsprache|Leichte_Sprache)\.(md|txt)$")

    standardsprache_files = {}
    leichte_sprache_files = {}

    if not src_data_dir.exists():
        print(f"Error: Directory '{src_data_dir}' does not exist.")
        return

    for file_path in src_data_dir.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                num_str, lang_type, _ = match.groups()
                if lang_type == "Standardsprache":
                    standardsprache_files[num_str] = file_path
                elif lang_type == "Leichte_Sprache":
                    leichte_sprache_files[num_str] = file_path

    # 4. Find matched IDs and sort them lexicographically
    # Ensuring '001' != '1' by treating them strictly as strings
    common_ids = set(standardsprache_files.keys()) & set(leichte_sprache_files.keys())
    sorted_ids = sorted(list(common_ids))

    print(f"Found {len(sorted_ids)} matching file pairs.")

    # 5. Process pairs and construct the JSONL dataset
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx in sorted_ids:
            # Read specific text contents
            standard_text = read_file(standardsprache_files[idx])
            leichte_text = read_file(leichte_sprache_files[idx])

            # Prepare user prompt by replacing the input token
            user_prompt = prompt_template.replace("%INPUT%", standard_text)

            # Construct the line payload
            entry = {
                "id": idx,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": leichte_text}
                ]
            }

            # Write single JSON line
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully generated compiled dataset at: {output_file}")

if __name__ == "__main__":
    main()