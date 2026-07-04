# --- src/evaluation.py ---
import os
import re
import gc
import sys
import json
from datetime import datetime, UTC
from pathlib import Path
import torch
import boto3
import sglang as sgl
from transformers import AutoTokenizer
import textstat

def get_raw_metrics(text: str) -> tuple:
    """Calculates German textstat metrics and returns rounded raw floats."""
    if not text.strip():
        return 0.0, 0.0
    fre = round(textstat.flesch_reading_ease(text), 1)
    wstf = round(textstat.wiener_sachtextformel(text, 1), 1)
    return fre, wstf

def read_file_with_extensions(base_path_str: str, extensions=[".txt", ".md"]) -> str:
    """Checks for the existence of a file across multiple extensions. Raises Error if absent."""
    for ext in extensions:
        full_path = f"{base_path_str}{ext}"
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read().strip()
                
    raise FileNotFoundError(
        f"❌ Data Integrity Violation: Required file not found for base path '{base_path_str}' "
        f"with extensions {extensions}."
    )

def patch_sglang_gemma4_mm():
    """
    Patches SGLang's gemma4_mm.py file to handle '_moe' suffixes and all individual
    attention/MLP modules (q_proj, k_proj, v_proj, gate_proj, up_proj) in get_hidden_dim
    to prevent NotImplementedError crashes in LoRA initialization.
    """
    try:
        import sglang.srt.models.gemma4_mm as gemma4_mm
        target_path = gemma4_mm.__file__
        if target_path.endswith(".pyc"):
            target_path = target_path[:-1]
            
        if not os.path.exists(target_path):
            print(f"ℹ️ SGLang gemma4_mm.py path {target_path} not found. Skipping file patch.", flush=True)
            return
            
        print(f"🛠️ Patching {target_path} to support all module keys in get_hidden_dim...", flush=True)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if 'Robust get_hidden_dim supporting standard, merged, and MoE layers' in content and 'is_moe = module_name' in content:
            print("✅ Already patched on disk.", flush=True)
        else:
            # Sophisticated replacement for get_hidden_dim
            pattern = r"    def get_hidden_dim\(self, module_name, layer_idx\):.*?return self\.config\.hidden_size"
            replacement = (
                "    def get_hidden_dim(self, module_name, layer_idx):\n"
                "        # Robust get_hidden_dim supporting standard, merged, and MoE layers\n"
                "        is_moe = module_name.endswith(\"_moe\")\n"
                "        base_name = module_name[:-4] if is_moe else module_name\n"
                "        \n"
                "        if base_name in [\"gate_proj\", \"up_proj\"]:\n"
                "            return self.config.intermediate_size if not is_moe else self.config.moe_intermediate_size\n"
                "        if base_name == \"down_proj\":\n"
                "            return self.config.hidden_size\n"
                "        return self.config.hidden_size"
            )
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("✅ Successfully patched gemma4_mm.py on disk.", flush=True)
    except Exception as e:
        print(f"⚠️ Error while patching gemma4_mm.py: {e}", flush=True)

def patch_sglang_gemma4_causal():
    """
    Patches SGLang's gemma4_causal.py file to add get_hidden_dim supporting hybrid attention and MoE.
    """
    try:
        import sglang.srt.models.gemma4_causal as gemma4_causal
        target_path = gemma4_causal.__file__
        if target_path.endswith(".pyc"):
            target_path = target_path[:-1]
            
        if not os.path.exists(target_path):
            print(f"ℹ️ SGLang gemma4_causal.py path {target_path} not found. Skipping file patch.", flush=True)
            return
            
        print(f"🛠️ Patching {target_path} to support get_hidden_dim...", flush=True)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if 'Robust get_hidden_dim supporting standard, merged, and MoE layers' in content and 'is_moe = module_name' in content:
            print("✅ Already patched causal on disk.", flush=True)
        else:
            patch_code = (
                "\n    def get_hidden_dim(self, module_name, layer_idx):\n"
                "        # Robust get_hidden_dim supporting standard, merged, and MoE layers\n"
                "        is_moe = module_name.endswith(\"_moe\")\n"
                "        base_name = module_name[:-4] if is_moe else module_name\n"
                "        \n"
                "        if base_name in [\"gate_proj\", \"up_proj\"]:\n"
                "            return self.config.intermediate_size if not is_moe else self.config.moe_intermediate_size\n"
                "        if base_name == \"down_proj\":\n"
                "            return self.config.hidden_size\n"
                "        return self.config.hidden_size\n"
            )
            if "class Gemma4ForCausalLM(nn.Module):" in content:
                content = content.replace("class Gemma4ForCausalLM(nn.Module):", "class Gemma4ForCausalLM(nn.Module):" + patch_code)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("✅ Successfully patched gemma4_causal.py on disk.", flush=True)
    except Exception as e:
        print(f"⚠️ Error while patching gemma4_causal.py: {e}", flush=True)

def patch_sglang_lora_mem_pool():
    """
    Patches SGLang's mem_pool.py file to ensure both standard 3D buffers
    and MoE 4D buffers are initialized for hybrid models (like Gemma 4)
    that contain both dense MLP layers and MoE layers.
    """
    try:
        import sglang.srt.lora.mem_pool as mem_pool
        target_path = mem_pool.__file__
        if target_path.endswith(".pyc"):
            target_path = target_path[:-1]
            
        if not os.path.exists(target_path):
            print(f"ℹ️ SGLang mem_pool.py path {target_path} not found. Skipping file patch.", flush=True)
            return
            
        print(f"🛠️ Patching {target_path} to support hybrid model buffers...", flush=True)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if "if is_moe:" in content and "else:" in content:
             print("✅ mem_pool.py on disk is already patched.", flush=True)
        else:
            target_str = "        self.pool = torch.zeros(self.pool_shape, dtype=dtype, device=device)"
            replacement = (
                "        self.pool = torch.zeros(self.pool_shape, dtype=dtype, device=device)\n"
                "        # Ensure 4D pool for MoE exists if 3D was requested as primary\n"
                "        if len(self.pool_shape) == 3:\n"
                "            moe_shape = (self.pool_shape[0], 1, self.pool_shape[1], self.pool_shape[2])\n"
                "            self.moe_pool = torch.zeros(moe_shape, dtype=dtype, device=device)\n"
                "        else:\n"
                "            self.moe_pool = self.pool"
            )
            if target_str in content:
                content = content.replace(target_str, replacement)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("✅ Successfully patched mem_pool.py on disk.", flush=True)
    except Exception as e:
        print(f"⚠️ Error while patching mem_pool.py: {e}", flush=True)

def patch_sglang_lora_manager():
    """
    Patches SGLang's lora_manager.py file to ensure should_apply_lora is respected.
    """
    try:
        import sglang.srt.lora.lora_manager as lora_manager
        target_path = lora_manager.__file__
        if target_path.endswith(".pyc"):
            target_path = target_path[:-1]
            
        if not os.path.exists(target_path):
            print(f"ℹ️ SGLang lora_manager.py path {target_path} not found. Skipping file patch.", flush=True)
            return
            
        print(f"🛠️ Patching {target_path} to respect should_apply_lora...", flush=True)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if 'hasattr(self.base_model, "should_apply_lora")' in content:
            print("✅ lora_manager.py on disk is already patched.", flush=True)
        else:
            target_str = (
                "            # The module should be converted if it is included in target_names\n"
                "            if module_name.split(\".\")[-1] in self.target_modules:\n"
                "                layer_id = get_layer_id(module_name)"
            )
            replacement = (
                "            # The module should be converted if it is included in target_names\n"
                "            if module_name.split(\".\")[-1] in self.target_modules:\n"
                "                if hasattr(self.base_model, \"should_apply_lora\") and not self.base_model.should_apply_lora(module_name):\n"
                "                    continue\n"
                "                layer_id = get_layer_id(module_name)"
            )
            if target_str in content:
                content = content.replace(target_str, replacement)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("✅ Successfully patched lora_manager.py on disk.", flush=True)
    except Exception as e:
        print(f"⚠️ Error while patching lora_manager.py: {e}", flush=True)

def patch_sglang_compressed_tensors_moe():
    """Patch CompressedTensorsW8A8Fp8MoE to support get_triton_quant_info"""
    try:
        import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe as scheme_module
        if hasattr(scheme_module.CompressedTensorsW8A8Fp8MoE, "get_triton_quant_info"):
             print("✅ CompressedTensorsW8A8Fp8MoE already has get_triton_quant_info.", flush=True)
             return

        def get_triton_quant_info(self, layer):
            from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
            return TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_fp8_w8a8=True,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
        scheme_module.CompressedTensorsW8A8Fp8MoE.get_triton_quant_info = get_triton_quant_info
        print("✅ In-memory monkeypatch applied to CompressedTensorsW8A8Fp8MoE.get_triton_quant_info.", flush=True)
    except Exception as e:
        print(f"⚠️ Error while patching SGLang scheme: {e}", flush=True)

def preprocess_adapter(adapter_id: str):
    """
    Cleans up the adapter config and safetensors for Gemma 4 architecture.
    """
    if not adapter_id or not os.path.exists(adapter_id):
        return
        
    config_path = os.path.join(adapter_id, "adapter_config.json")
    safetensors_path = os.path.join(adapter_id, "adapter_model.safetensors")
    
    if os.path.exists(config_path):
        print(f"🛠️ Preprocessing adapter config in {config_path}...")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        target_modules = config.get("target_modules", [])
        if any("language_model" in m for m in target_modules):
            new_targets = [m.replace("language_model.model.", "model.").replace("language_model.", "model.") for m in target_modules]
            config["target_modules"] = new_targets
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"✅ Filtered target modules.")

    if os.path.exists(safetensors_path):
        print(f"🛠️ Preprocessing safetensors in {safetensors_path}...")
        try:
            from safetensors.torch import load_file, save_file
            tensors = load_file(safetensors_path)
            new_tensors = {}
            for key, tensor in tensors.items():
                new_key = key.replace("language_model.model.", "model.").replace("language_model.", "model.")
                new_tensors[new_key] = tensor
            save_file(new_tensors, safetensors_path)
            print("✅ Safetensors keys cleaned.")
        except Exception as e:
            print(f"⚠️ Error preprocessing safetensors: {e}")

def run_evaluation(model_id, quantization_type, max_len=8192, adapter_id=None, evaluation_set=None, reasoning_parser=None):
    """
    Initializes the engine and processes conversations.
    """
    if evaluation_set is None:
        evaluation_set = []
        
    print("\n" + "="*60)
    print(f"🚀 LOADING MODEL FOR BATCH EVALUATION: {model_id}")
    if adapter_id:
        print(f"🧬 Active Adapter: {adapter_id}")
    print("="*60, flush=True)
    
    generated_responses = []
    is_gemma = "gemma" in model_id.lower()
    
    try:
        # DYNAMIC HARDWARE DETECTION
        available_gpus = int(os.environ.get("TP_SIZE", torch.cuda.device_count() if torch.cuda.is_available() else 2))
        
        print("📖 Loading tokenizer and formatting chat prompts...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        prompts = []
        for prompt_item in evaluation_set:
            messages = [
                {"role": "system", "content": prompt_item["system"]},
                {"role": "user", "content": prompt_item["templated_user"]}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(formatted_prompt)

        engine_kwargs = {
            "model_path": model_id,
            "tp_size": available_gpus,
            "context_length": max_len,
            "trust_remote_code": True,
            "reasoning_parser": reasoning_parser,
        }
        
        if quantization_type:
            engine_kwargs["quantization"] = quantization_type
            
        if adapter_id:
            preprocess_adapter(adapter_id)
            engine_kwargs["enable_lora"] = True
            engine_kwargs["lora_paths"] = [f"adapter0={adapter_id}"]
            engine_kwargs["max_loras_per_batch"] = 1
            
        if is_gemma:
            print("🔧 Applying Gemma-specific engine overrides", flush=True)
            engine_kwargs["attention_backend"] = "triton"
            engine_kwargs["moe_runner_backend"] = "triton"
            engine_kwargs["disable_cuda_graph"] = True

        print(f"⚙️ Initializing SGLang Engine with parameters: {engine_kwargs}", flush=True)
        llm = sgl.Engine(**engine_kwargs)
        
        sampling_params = {
            "temperature": 1.0 if is_gemma else 0.3,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "skip_special_tokens": False
        }
            
        generate_kwargs = {}
        if adapter_id:
            generate_kwargs["lora_path"] = "adapter0"
            
        print(f"⚡ Processing {len(prompts)} prompts via unified SGLang Engine...", flush=True)
        outputs = llm.generate(prompt=prompts, sampling_params=sampling_params, **generate_kwargs)
        
        for out in outputs:
            raw_text = out["text"].strip()
            reasoning_trace = ""
            
            think_match = re.search(
                r"(?:<\|channel>thought\n|<\|channel\|>thought|<|thought\|>|<(?:think|thought)>|\[(?:think|thought)\])(.*?)(?:<channel\|>|</(?:think|thought)>|\[/(?:think|thought)\]|$)",
                raw_text, 
                re.DOTALL | re.IGNORECASE
            )
            if think_match:
                reasoning_trace = think_match.group(1).strip()
                
            raw_text = re.sub(
                r"(?:<\|channel>thought\n|<\|channel\|>thought|<|thought\|>|<(?:think|thought)>|\[(?:think|thought)\]).*?(?:<channel\|>|</(?:think|thought)>|\[/(?:think|thought)\]|$)",
                "", 
                raw_text, 
                flags=re.DOTALL | re.IGNORECASE
            ).strip()
                
            generated_responses.append((raw_text, reasoning_trace))
            
    except Exception as e:
        print(f"❌ Execution error encountered on {model_id}: {e}", flush=True)
        generated_responses = [("", "") for _ in evaluation_set]
        
    finally:
        print(f"♻️ Evacuating VRAM channels for next model tracking...", flush=True)
        if 'llm' in locals():
            try:
                llm.shutdown()
            except:
                pass
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ VRAM cleared.", flush=True)
        
    return generated_responses

def main():
    # 1. APPLY PATCHES
    patch_sglang_gemma4_mm()
    patch_sglang_gemma4_causal()
    patch_sglang_lora_mem_pool()
    patch_sglang_lora_manager()
    patch_sglang_compressed_tensors_moe()

    # 2. LOAD CONFIGS
    with open("data/system-prompt.md", "r", encoding="utf-8") as f:
        global_system_prompt = f.read().strip()
    with open("data/prompt-template.md", "r", encoding="utf-8") as f:
        global_template = f.read()

    evaluation_set = []
    # Integrity check
    evaluation_set.append({
        "is_integrity": True,
        "original_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "templated_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "system": "Du bist ein hilfreicher Assistent",
        "reference_text": None
    })
    
    # Dataset records
    jsonl_path = "data/train/dataset.jsonl"
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                prompt_id = str(entry.get("id", "")).strip()
                orig_content = read_file_with_extensions(f"data/raw/{prompt_id}_Standardsprache")
                ref_content = read_file_with_extensions(f"data/raw/{prompt_id}_Leichte_Sprache")
                evaluation_set.append({
                    "is_integrity": False,
                    "original_user": orig_content,
                    "templated_user": global_template.replace("%INPUT%", orig_content),
                    "system": global_system_prompt,
                    "reference_text": ref_content
                })

    # LIMIT FOR VERIFICATION
    evaluation_set = evaluation_set[:8]

    # 3. CONFIGURE PIPELINE
    mistral_adapter = "/app/output/adapter/mistral4small"
    if not os.path.exists(os.path.join(mistral_adapter, "adapter_config.json")):
        mistral_adapter = "/app/output/adapter/train-mistral4small"
        
    gemma_adapter = "/app/output/adapter/gemma4"
    if not os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        gemma_adapter = "/app/output/adapter/train-gemma4"

    EVALUATION_PIPELINE = []
    
    # Model 1: Gemma Plain
    base_gemma = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
    EVALUATION_PIPELINE.append((base_gemma, None, 8192, None, "Gemma 4 (Plain)", None))
    
    # Model 2: Gemma Reasoning
    EVALUATION_PIPELINE.append((base_gemma, None, 8192, None, "Gemma 4 (Reasoning)", "gemma4"))
    
    # Model 3: Gemma Fine-tuned
    merged_gemma = "/app/output/merged/train-gemma4-fp8"
    if os.path.exists(merged_gemma):
        EVALUATION_PIPELINE.append((merged_gemma, None, 8192, None, "Gemma 4 (Fine-tuned)", "gemma4"))
    elif os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        EVALUATION_PIPELINE.append((base_gemma, None, 8192, gemma_adapter, "Gemma 4 (Fine-tuned)", "gemma4"))

    # Model 4: Mistral Plain
    # base_mistral = "cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit"
    # EVALUATION_PIPELINE.append((base_mistral, "compressed-tensors", 8192, None, "Mistral 119B (Plain)", None))
    
    # Model 5: Mistral Fine-tuned
    # if os.path.exists(os.path.join(mistral_adapter, "adapter_config.json")):
    #     EVALUATION_PIPELINE.append((base_mistral, "compressed-tensors", 8192, mistral_adapter, "Mistral 119B (Fine-tuned)", None))

    # 4. EXECUTE PIPELINE
    output_json = {
        "system": global_system_prompt,
        "template": global_template,
        "models": [],
        "prompts": []
    }
    
    for item in evaluation_set:
        input_fre, input_wstf = get_raw_metrics(item["original_user"])
        output_json["prompts"].append({"r": [[item["original_user"], input_fre, input_wstf, ""]]})

    for model_id, quant_type, max_len, adapter_id, display_name, parser in EVALUATION_PIPELINE:
        output_json["models"].append(display_name)
        responses = run_evaluation(model_id, quant_type, max_len, adapter_id, evaluation_set, reasoning_parser=parser)
        
        for idx, (text, trace) in enumerate(responses):
            fre, wstf = get_raw_metrics(text)
            output_json["prompts"][idx]["r"].append([text, fre, wstf, trace])

        # Save checkpoint
        os.makedirs("evaluation", exist_ok=True)
        cp_file = f"evaluation/checkpoint_{display_name.replace(' ', '_')}.json"
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"💾 Checkpoint saved: {cp_file}")

    # 5. FINAL EXPORT
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    filename = f"evaluation/{timestamp}_evaluation.json"
    bucket = os.environ.get("S3_BUCKET", "diwop-leichte-sprache")
    json_payload = json.dumps(output_json, ensure_ascii=False, indent=2)
    
    try:
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket, Key=filename, Body=json_payload, ContentType='application/json')
        print(f"🚀 S3 Export successful: {filename}")
    except Exception as e:
        print(f"❌ S3 Error: {e}")
        with open(filename, "w", encoding="utf-8") as f: f.write(json_payload)

if __name__ == "__main__":
    main()