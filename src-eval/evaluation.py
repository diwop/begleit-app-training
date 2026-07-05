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
            # We match until the next 'def ' to ensure we clear the whole old function body
            pattern = r"    def get_hidden_dim\(self, module_name, layer_idx\):.*?(?=\n    def )"
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
                "        return self.config.hidden_size\n"
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

def run_evaluation(model_id, quantization_type, max_len=8192, adapter_id=None, evaluation_set=None, reasoning_parser=None, mem_fraction_static=0.8):
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
        
        # Proactive fix for 2-GPU deadlocks on virtualized PCIe/InfiniBand (RunPod)
        if available_gpus == 2:
            if "NCCL_P2P_DISABLE" not in os.environ:
                print("ℹ️ 2-GPU cluster detected. Auto-disabling NCCL P2P to prevent deadlocks.", flush=True)
                os.environ["NCCL_P2P_DISABLE"] = "1"
            if "NCCL_IB_DISABLE" not in os.environ:
                os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        
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
            "mem_fraction_static": mem_fraction_static,
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
            
            # 1. Primary Source: SGLang reasoning parser metadata
            if reasoning_parser:
                meta = out.get("meta_info", {})
                reasoning_trace = meta.get("reasoning_content", "").strip()
            
            # 2. Fallback: Robust regex extraction if reasoning_trace is empty
            if not reasoning_trace:
                # Catching common Gemma/Llama reasoning delimiters
                think_match = re.search(
                    r"(?:<\|channel>thought\n|<\|channel\|>thought|<\|thought\|>|<(?:think|thought)>|\[(?:think|thought)\])(.*?)(?:<\|channel\|>|<channel\|>|</(?:think|thought)>|\[/(?:think|thought)\]|$)",
                    raw_text, 
                    re.DOTALL | re.IGNORECASE
                )
                if think_match:
                    reasoning_trace = think_match.group(1).strip()
                
            # 3. Cleanup: Ensure reasoning tokens are removed from final text
            raw_text = re.sub(
                r"(?:<\|channel>thought\n|<\|channel\|>thought|<\|thought\|>|<(?:think|thought)>|\[(?:think|thought)\]).*?(?:<\|channel\|>|<channel\|>|</(?:think|thought)>|\[/(?:think|thought)\]|$)",
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
    # Part 1: Integrity check
    evaluation_set.append({
        "is_integrity": True,
        "original_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "templated_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "system": "Du bist ein hilfreicher Assistent",
        "reference_text": None
    })
    
    # Part 2: Original Test Prompts
    original_test_prompts = [
        "Warum ist der Himmel blau und nicht schwarz?",
        "Was ist die Quadratwurzel aus 16?",
        "# Magdeburg bundesweit vorn bei Hausärztinnen\n\nNirgendwo in Deutschland ist der Frauenanteil bei den Hausärzten so hoch wie in Magdeburg...",
        "Guten Tag! Wie geht es Ihnen?",
        "Herr Müller, beim letzten Mal haben wir über Bluthochdruck gesprochen. Erinnern sie sich noch, was das bedeutet?",
        "Das hier sind Ihre Blutdruckwerte aus der letzten Woche. Da können Sie sehen, dass der Blutdruck immer noch zu hoch ist. Sie sollten versuchen, Ihren Blutdruck zu senken. Das können Sie tun, indem Sie weniger Salz essen und mehr Sport treiben. Ansonsten können Sie auch einen Blutdrucksenker einnehmen. Aber erstmal sollten wir es mit den Anpassungen bei Ihrem Lebensstil versuchen. Haben Sie dazu Fragen?",
        "Die Quantenchromodynamik (kurz QCD) ist eine Quantenfeldtheorie zur Beschreibung der starken Wechselwirkung. Sie beschreibt die Wechselwirkung von Quarks und Gluonen, also der fundamentalen Bausteine der Atomkerne.\nDie QCD ist wie die Quantenelektrodynamik (QED) eine Eichtheorie. Während die QED jedoch auf der abelschen Eichgruppe U(1) beruht und die Wechselwirkung elektrisch geladener Teilchen (z. B. Elektron oder Positron) mit Photonen beschreibt, wobei die Photonen selbst ungeladen sind, ist die Eichgruppe der QCD, die SU(3), nicht-abelsch. Es handelt sich also um eine Yang-Mills-Theorie. Die Wechselwirkungsteilchen der QCD sind die Gluonen, und an die Stelle der elektrischen Ladung als Erhaltungsgröße tritt die Farbladung (daher der Name Chromodynamik). Die Gluonen selbst sind im Gegensatz zu den Eichteilchen der QED „geladen“, das heißt Träger von Farbladungen, und wechselwirken auch untereinander.",
        "# Lachs im Sesammantel auf Erbsenpüree und Zuckerschotenstroh\nZutaten Für 4 Portionen:\n* 4 Lachssteak(s) küchenfertig, à 140 g\n* 4 EL Sesam geröstet, weiß und schwarz\n* 2 EL Öl (Woköl mit Sesamaroma)\n* 2 EL Butter\n* 2 Schalotte(n)\n* 400 g Erbsen, TK\n* 2 EL Sahne\n* Salz und Pfeffer\n* Muskat\n* Zucker\n* 100 g Zuckerschote(n)\n* 1 EL Butter\n* Erbsensprossen (Erbsenspargelsprossen) für die Dekoration\nGesamtzeit: 35 Min.\nArbeitszeit: 25 Min.\nKoch-/Backzeit: 10 Min.\n1. Die Schalotten abziehen und in Würfel schneiden. Diese in einem Topf mit der Butter angehen lassen, die aufgetauten Erbsen zufügen. Etwas angehen lassen und mit Salz, Pfeffer, Zucker und Muskat würzen. Sahne zufügen, ca. fünf Minuten dünsten und danach im Mixer sehr fein pürieren.\n2. Den Lachs im Sesam wenden und in einer Pfanne mit dem Öl bei mittlerer Hitze von beiden Seiten je zwei Minuten braten und anschließend zwei Minuten ruhen lassen. Mit Salz und Pfeffer würzen.\n3. Die Zuckerschoten in dünne Streifen schneiden und in Butter glacieren. Mit Salz, Muskat und etwas Zucker würzen.\n4. Anrichten: Das Püree auf einem tiefen Teller anrichten, den aufgeschnittenen Lachs darauf setzen und von den glacierten Schoten einen Löffel dararauf verteilen. Mit Erbsspargelsprossen dekorieren.\n5. Guten Appetit!",
        "The Creation of the World\nIn the beginning, God created the heavens and the earth. The earth was without form and void, and darkness was over the face of the deep. And the Spirit of God was hovering over the face of the waters.\nAnd God said, “Let there be light,” and there was light. And God saw that the light was good. And God separated the light from the darkness. God called the light Day, and the darkness he called Night. And there was evening and there was morning, the first day.",
        "Remigration (von lateinisch remigrare „zurückwandern“, „zurückkehren“), auch Rückwanderung oder Rückkehrmigration, bezeichnet den Teil eines Migrationsprozesses, bei dem Menschen nach einer beträchtlichen Zeitspanne in einem anderen Land oder einer anderen Region in ihr Herkunftsland oder ihre Herkunftsregion zurückkehren. Remigration findet in umgekehrter Richtung zur vorangegangenen Migration statt. Der Begriff wurde von der Neuen Rechten als Kampfbegriff und Euphemismus für Vertreibung und Deportation etabliert. Eine Jury wählte ihn zum „Unwort des Jahres 2023“ in Deutschland.",
        "Unsere einst stolzen Städte verwahrlosen immer mehr und sind Brutstätten von Kriminalität und Gewalt und leider oftmals Heimstätte von radikalen Islamisten. Unser einst fruchtbares Land verliert seine Bewohner, verödet aufgrund einer desaströsen und völlig falsch angelegten Strukturpolitik. Unsere einst schöne Heimat wird zusehends durch hässliche Bauten, Windräder und eine chaotische Besiedlung verunstaltet. Unsere einst kraftvolle Wirtschaft ist nur noch ein Wrack, neoliberal ausgezehrt. Unser einst beneideter, unser einst weltweit beneideter sozialer Friede ist durch den steigenden Missbrauch und die Aufgabe der national begrenzten Solidargemeinschaft sowie durch den Import fremder Völkerschaften und die zwangsläufigen Konflikte existenziell gefährdet. Liebe Freunde, und unser liebes Volk ist im inneren tief gespalten und durch den Geburtenrückgang sowie die Masseneinwanderung, erstmals in seiner Existenz tatsächlich elementar bedroht.",
        "Macht was ihr wollt, aber schreibt nicht \"Wir sind das Volk!\" Ihr seid nicht das Volk, ihr seid der verblendete, verblödete, braune Bodensatz des Volkes. Ihr seid der widerliche, nervende kleine Pickel am Arsch der Gesellschaft, aber sicherlich nicht das Volk!",
        "Inzwischen könnte ich beidem Wort \"bunt\" nur noch kotzen. Solange wirklich Fachkräfte kommen, hat ja kein Mensch was dagegen. Auch die Spanier und Italiener, die hier ihre Ausbildung machen, sind doch willkommen. Dieses Getue in den Medien geht mir tierisch auf den Senkel. Und sie wissen immer noch nicht (oder wollen es nicht wissen) worum es uns geht.",
        "Diese Pisser!!! völliger Quatsch, welche Partei mit den Grünen oder Linken sympathisiert kann nichts gutes für das deutsche Volk wollen ebenso wie die komischen Christlichen.",
        "Ja die DDR lässt überall grüßen, ich wundere mich auch jeden Tag. Zensur, Einheitsmeinung, Volksentscheid unerwünscht. Propaganda-Medien. und eine durchgeknallte Staatsratsvorsitzende....",
        "die sollten sich von den skandinavischen gruppenvergewaltigungsopfern tips geben lassen,wie man das blut aus den klamotten bekommt! eigentlich traurig,dass man solche beispiele bringen muss! linda aus oslo ist ein schlimmesbld u läßt nur ansatzweise erahnen,was sie durchgemacht haben muss...",
        "Die verdammten Drecksvölker,und Deutschland will sich das Dreckspack ins Land holen!"
    ]
    
    for text_block in original_test_prompts:
        evaluation_set.append({
            "is_integrity": False,
            "original_user": text_block,
            "templated_user": global_template.replace("%INPUT%", text_block),
            "system": global_system_prompt,
            "reference_text": None
        })

    # Part 3: Dataset records
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
    print(f"📊 Evaluation set size: {len(evaluation_set)} prompts", flush=True)

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

    # Model 4: Mistral (Parked on next-models-eval)
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
        print("\n" + "="*60, flush=True)
        print(f"🚀 LOADING MODEL FOR BATCH EVALUATION: {display_name}", flush=True)
        print("="*60, flush=True)
        
        import time
        start_eval = time.time()
        start_str = time.strftime('%H:%M:%S', time.localtime(start_eval))
        print(f"⏰ Start Time: {start_str}", flush=True)

        output_json["models"].append(display_name)
        responses = run_evaluation(
            model_id, 
            quant_type, 
            max_len, 
            adapter_id, 
            evaluation_set, 
            reasoning_parser=parser,
            mem_fraction_static=0.8
        )
        
        end_eval = time.time()
        end_str = time.strftime('%H:%M:%S', time.localtime(end_eval))
        duration = end_eval - start_eval
        print(f"✅ Finished: {display_name}", flush=True)
        print(f"⏰ End Time: {end_str}", flush=True)
        print(f"⏱️ Elapsed: {duration:.2f}s ({duration/60:.2f} min)", flush=True)
        
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