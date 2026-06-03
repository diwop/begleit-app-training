import os
import re
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

def read_file(path: Path) -> str:
    """Helper to read file contents cleanly."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def calculate_token_count(messages: list, tokenizer) -> int:
    """Calculates the sequence length (number of tokens) of a conversation."""
    try:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        return len(tokens)
    except Exception:
        try:
            tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            if isinstance(tokens, dict) and "input_ids" in tokens:
                return len(tokens["input_ids"])
            else:
                return len(tokens)
        except Exception:
            raw_text = "\n".join([m["content"] for m in messages])
            return len(tokenizer.encode(raw_text))

def calculate_percentile(values: list, percentile: float) -> int:
    """Calculates a simple percentile from a list of integer values."""
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile)
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]

def main():
    # 1. Define paths based on your dvc.yaml structure
    src_data_dir = Path("data/raw")
    output_file = Path("data/train/dataset.jsonl")
    system_prompt_path = Path("data/system-prompt.md")
    prompt_template_path = Path("data/prompt-template.md")
    base_yml_path = Path("config/base.yml")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 2. Extract sequence length from base.yml using regex
    try:
        with open(base_yml_path, "r", encoding="utf-8") as f:
            base_content = f.read()
        seq_match = re.search(r"^sequence_len:\s*(\d+)", base_content, re.MULTILINE)
        
        if not seq_match:
            print(f"ERROR: Could not locate 'sequence_len' in {base_yml_path} via pattern matching.")
            sys.exit(1)
            
        MAX_LEN = int(seq_match.group(1))
        print(f"Loaded sequence limit from base.yml: {MAX_LEN} tokens")
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {base_yml_path}")
        sys.exit(1)

    # 3. Read global system prompt and user template
    system_prompt = read_file(system_prompt_path)
    prompt_template = read_file(prompt_template_path)

    TOKENIZER_NAME = "cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit"
    print(f"Loading tokenizer from {TOKENIZER_NAME}...")
    FALLBACK_TOKENIZER = "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit"
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    except Exception as e:
        if "TokenizersBackend" in str(e):
            print(f"Warning: Tokenizer class incompatible ({e}). Falling back to {FALLBACK_TOKENIZER}...")
        else:
            print("Tokenizer not found in local cache. Fetching from Hugging Face Hub (this may take a moment)...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
            except Exception as remote_err:
                print(f"Warning: Failed to load tokenizer {TOKENIZER_NAME} due to: {remote_err}. Falling back to {FALLBACK_TOKENIZER}...")
                
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)
        except Exception as fallback_err:
            print(f"ERROR: Failed to load fallback tokenizer {FALLBACK_TOKENIZER}: {fallback_err}")
            sys.exit(1)

    # 4. Scan the src-data directory using regex to capture the exact ID string
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

    # Find matched IDs and sort them lexicographically
    common_ids = set(standardsprache_files.keys()) & set(leichte_sprache_files.keys())
    sorted_ids = sorted(list(common_ids))
    print(f"Found {len(sorted_ids)} matching file pairs.")

    # Validation trackers
    total_samples = 0
    exceeded_count = 0
    max_token_count = 0
    sum_total_tokens = 0
    sum_assistant_tokens = 0
    max_assistant_token_count = 0
    all_total_tokens = []
    all_assistant_tokens = []

    # 5. Process pairs, validate tokens, and construct the JSONL dataset
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx in sorted_ids:
            total_samples += 1
            
            # Read specific text contents
            standard_text = read_file(standardsprache_files[idx])
            leichte_text = read_file(leichte_sprache_files[idx])

            # Prepare user prompt by replacing the input token
            user_prompt = prompt_template.replace("%INPUT%", standard_text)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": leichte_text}
            ]

            # Validate token count
            token_count = calculate_token_count(messages, tokenizer)
            assistant_token_count = len(tokenizer.encode(leichte_text))

            # Update trackers
            all_total_tokens.append(token_count)
            all_assistant_tokens.append(assistant_token_count)
            sum_total_tokens += token_count
            sum_assistant_tokens += assistant_token_count

            if token_count > max_token_count:
                max_token_count = token_count

            if assistant_token_count > max_assistant_token_count:
                max_assistant_token_count = assistant_token_count

            if token_count > MAX_LEN:
                print(f"[WARNING] Pair {idx} is {token_count} tokens (Exceeds {MAX_LEN} limit)")
                exceeded_count += 1

            # Construct the line payload
            entry = {"id": idx, "messages": messages}

            # Write single JSON line
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    avg_total_tokens = sum_total_tokens / total_samples if total_samples > 0 else 0
    avg_assistant_tokens = sum_assistant_tokens / total_samples if total_samples > 0 else 0

    p50_total = calculate_percentile(all_total_tokens, 0.5)
    p75_total = calculate_percentile(all_total_tokens, 0.75)
    p90_total = calculate_percentile(all_total_tokens, 0.9)
    p50_assistant = calculate_percentile(all_assistant_tokens, 0.5)
    p75_assistant = calculate_percentile(all_assistant_tokens, 0.75)
    p90_assistant = calculate_percentile(all_assistant_tokens, 0.9)

    # 6. Print Summary Report
    print("\n" + "=" * 60)
    print("                  DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f" Total Samples Checked : {total_samples}")
    print(" Tokens:")
    print(f"   50% : {p50_total}")
    print(f"   75% : {p75_total}")
    print(f"   90% : {p90_total}")
    print(f"   Max : {max_token_count} (Limit: {MAX_LEN})")
    print(f"   Avg : {avg_total_tokens}")
    print(f" Samples Over Limit    : {exceeded_count}")
    print("\n Assistant Tokens:")
    print(f"   50% : {p50_assistant}")
    print(f"   75% : {p75_assistant}")
    print(f"   90% : {p90_assistant}")
    print(f"   Max : {max_assistant_token_count}")
    print(f"   Avg : {avg_assistant_tokens}")
    print("=" * 60 + "\n")

    # 7. Fail the pipeline if necessary
    if exceeded_count > 0:
        print(f"[FATAL ERROR] {exceeded_count} sample(s) exceeded the sequence length limit!")
        print(f"Note: You can extend 'sequence_len' in config/base.yml to accommodate this data.")
        print(f"HOWEVER, expanding the sequence length will require more GPU VRAM during training.")
        print(f"You may need to upgrade your hardware tier or reduce your micro_batch_size to compensate.")
        sys.exit(1)

    print(f"Successfully generated clean, validated dataset at: {output_file}")

if __name__ == "__main__":
    main()