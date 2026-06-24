# --- src/evaluation.py ---
from transformers.models.big_bird import modeling_big_bird
from asyncio import coroutines
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

def run_evaluation(model_id, quantization_type, max_len=8192, adapter_id=None, evaluation_set=None):
    """
    Initializes the engine, handles conditional FP8 and architecture properties,
    and processes conversations through the safe native parallel llm.chat backend.
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
    is_mistral = "mistral" in model_id.lower()
    
    try:
        # DYNAMIC HARDWARE DETECTION
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if available_gpus not in [1, 2, 4, 8]:
            print(f"⚠️ Warning: Asymmetrical GPU count ({available_gpus}) detected. Falling back to 2.")
            available_gpus = 2

        # 1. LOAD TOKENIZER AND FORMAT CHAT PROMPTS
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

        # 2. CONFIGURE SGLANG ENGINE
        engine_kwargs = {
            "model_path": model_id,
            "tp_size": available_gpus,
            "context_length": max_len,
            "trust_remote_code": True,
        }
        
        if quantization_type:
            engine_kwargs["quantization"] = quantization_type
            
        if adapter_id:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["lora_paths"] = [f"adapter0={adapter_id}"]
            engine_kwargs["max_loras_per_batch"] = 1
            
        print(f"⚙️ Initializing SGLang Engine with parameters: {engine_kwargs}", flush=True)
        llm = sgl.Engine(**engine_kwargs)
        
        # 3. SET SAMPLING PARAMS
        if is_gemma:
            sampling_params = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "max_new_tokens": 4096,
                "skip_special_tokens": False
            }
        else:
            sampling_params = {
                "temperature": 0.3,
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
            
            # Universal string extraction regex for plain tags and Gemma hardware tokens
            think_match = re.search(
                r"(?:<\|channel>thought\n|<\|channel\|>thought|<|thought\|>|<(?:think|thought)>|\[(?:think|thought)\])(.*?)(?:<channel\|>|</(?:think|thought)>|\[/(?:think|thought)\]|$)",
                raw_text, 
                re.DOTALL | re.IGNORECASE
            )
            if think_match:
                reasoning_trace = think_match.group(1).strip()
                
            # Wipe structural thought text sequences entirely out of final textstat payloads
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
            except Exception as e:
                print(f"⚠️ Error shutting down SGLang Engine: {e}", flush=True)
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ VRAM cleared.", flush=True)
        
    return generated_responses

def has_local_adapter(adapter_dir="/app/output/adapter"):
    if not os.path.exists(adapter_dir):
        return False
    for root, dirs, files in os.walk(adapter_dir):
        if "adapter_config.json" in files:
            return True
    return False

def get_newest_run_prefix(bucket_name: str) -> str:
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    prefixes = []
    
    # Try using delimiter first (efficient)
    for page in paginator.paginate(Bucket=bucket_name, Delimiter='/'):
        for cp in page.get('CommonPrefixes', []):
            prefix = cp.get('Prefix', '')
            base = prefix.strip('/')
            if '_run' in base:
                prefixes.append(prefix)
                
    if not prefixes:
        # Fallback: list all objects and parse prefixes
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                key = obj.get('Key', '')
                parts = key.split('/')
                if len(parts) > 1:
                    first_dir = parts[0]
                    if '_run' in first_dir:
                        prefixes.append(first_dir + '/')
                        
    if not prefixes:
        raise ValueError(f"No run folders (containing '_run') found in bucket {bucket_name}")
        
    return sorted(list(set(prefixes)))[-1]

def download_s3_folder(bucket_name: str, prefix: str, local_dir: str):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    print(f"📥 Downloading s3://{bucket_name}/{prefix} to {local_dir}...", flush=True)
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj.get('Key', '')
            if key.endswith('/'):
                continue
                
            rel_path = key[len(prefix):]
            local_file_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            print(f"  Downloading: {key} -> {local_file_path}", flush=True)
            s3_client.download_file(bucket_name, key, local_file_path)

def main():
    # Ensure S3 adapters are downloaded if not present locally
    adapter_base_dir = "/app/output/adapter"
    if not has_local_adapter(adapter_base_dir):
        print("🔍 No local adapters found in /app/output/adapter. Checking S3...", flush=True)
        bucket_name = os.environ.get("S3_BUCKET", "diwop-leichte-sprache")
        s3_adapter_run = os.environ.get("S3_ADAPTER_RUN")
        
        if s3_adapter_run:
            print(f"📦 $S3_ADAPTER_RUN specified: {s3_adapter_run}", flush=True)
            prefix = s3_adapter_run
        else:
            print("📦 $S3_ADAPTER_RUN not specified. Listing S3 bucket to find the newest run...", flush=True)
            try:
                prefix = get_newest_run_prefix(bucket_name)
                print(f"✨ Identified newest run: {prefix}", flush=True)
            except Exception as e:
                print(f"❌ Error identifying newest run: {e}", flush=True)
                prefix = None
                
        if prefix:
            if not prefix.endswith("/"):
                prefix += "/"
            try:
                download_s3_folder(bucket_name, prefix, adapter_base_dir)
                print("✅ S3 download completed successfully!", flush=True)
            except Exception as e:
                print(f"❌ Error downloading adapter from S3: {e}", flush=True)
        else:
            print("⚠️ Skipping S3 download as no run could be identified.", flush=True)
    else:
        print("✅ Local adapter already present in /app/output/adapter. Skipping S3 download.", flush=True)

    print("📋 Validating structural configurations...", flush=True)
    if not os.path.exists("data/system-prompt.md") or not os.path.exists("data/prompt-template.md"):
        raise FileNotFoundError("❌ Pipeline Failure: Configuration blueprint files are missing.")

    with open("data/system-prompt.md", "r", encoding="utf-8") as f:
        global_system_prompt = f.read().strip()
    with open("data/prompt-template.md", "r", encoding="utf-8") as f:
        global_template = f.read()

    evaluation_set = []
    
    # PART 1: APPEND INTEGRITY CHECK PROMPT
    evaluation_set.append({
        "is_integrity": True,
        "original_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "templated_user": "Warum ist der Himmel blau? Gib eine kurze Antwort!",
        "system": "Du bist ein hilfreicher Assistent",
        "reference_text": None
    })
    
    # PART 2: APPEND REGULAR UNTAGGED TEST PROMPTS
    original_test_prompts = [
        "Warum ist der Himmel blau und nicht schwarz?",
        "Was ist die Quadratwurzel aus 16?",
        "# Magdeburg bundesweit vorn bei Hausärztinnen\n\nNirgendwo in Deutschland ist der Frauenanteil bei den Hausärzten so hoch wie in Magdeburg...",
        "Guten Tag! Wie geht es Ihnen?",
        "guten Tag wie geht es ihnen",
        "Guten Tag, Herr Müller! Wie geht es Ihnen?",
        "guten Tag Herr Müller wie geht es ihnen",
        "Herr Müller, beim letzten Mal haben wir über Bluthochdruck gesprochen. Erinnern sie sich noch, was das bedeutet?",
        "Herr Müller beim letzten mal haben wir über Bluthochdruck gesprochen erinnern sie sich noch was das bedeutet",
        "Das hier sind Ihre Blutdruckwerte aus der letzten Woche. Da können Sie sehen, dass der Blutdruck immer noch zu hoch ist. Sie sollten versuchen, Ihren Blutdruck zu senken. Das können Sie tun, indem Sie weniger Salz essen und mehr Sport treiben. Ansonsten können Sie auch einen Blutdrucksenker einnehmen. Aber erstmal sollten wir es mit den Anpassungen bei Ihrem Lebensstil versuchen. Haben Sie dazu Fragen?",
        "Das hier sind ihre Blutdruckwerte aus der letzten Woche da können sie sehen das der Blutdruck immer noch zu hoch ist sie sollten versuchen ihren Blutdruck zu senken das können sie tun indem sie weniger Salz essen und mehr Sport treiben ansonsten können sie auch einen Blutdrucksenker einnehmen aber erstmal sollten wir es mit den Anpassungen bei ihrem Lebensstil versuchen haben sie dazu Fragen",
        "Haben Sie noch eine angenehme Woche. Bis zum nächsten Mal!",
        "haben Sie noch eine angenehme Woche bis zum nächsten mal",
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

    # PART 3: INGEST ADAPTER DATASET FROM JSONL
    jsonl_path = "data/train/dataset.jsonl"
    if os.path.exists(jsonl_path):
        print(f"📥 Parsing tuning records from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip(): continue
                entry = json.loads(line)
                prompt_id = str(entry.get("id", "")).strip()
                # 
                orig_content = read_file_with_extensions(f"data/raw/{prompt_id}_Standardsprache")
                ref_content = read_file_with_extensions(f"data/raw/{prompt_id}_Leichte_Sprache")
                # 
                evaluation_set.append({
                    "is_integrity": False,
                    "original_user": orig_content,
                    "templated_user": global_template.replace("%INPUT%", orig_content),
                    "system": global_system_prompt,
                    "reference_text": ref_content
                })

    # --- DYNAMIC ADAPTER CHECKPOINT RESOLUTION ---
    mistral_adapter = "/app/output/adapter/mistral4small"
    if not os.path.exists(os.path.join(mistral_adapter, "adapter_config.json")):
        mistral_adapter = "/app/output/adapter/train-mistral4small"
        
    gemma_adapter = "/app/output/adapter/gemma4"
    if not os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        gemma_adapter = "/app/output/adapter/train-gemma4"
    EVALUATION_PIPELINE = []
    # Mistral stays on compressed-tensors AWQ layout
    EVALUATION_PIPELINE.append(("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", "compressed-tensors", 8192, None))
    if os.path.exists(os.path.join(mistral_adapter, "adapter_config.json")):
        EVALUATION_PIPELINE.append(("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit", "compressed-tensors", 8192, mistral_adapter))

    # Gemma routes through the official model repo with hardware native FP8 execution
    EVALUATION_PIPELINE.append(("google/gemma-4-26b-a4b-it", "fp8", 8192, None))
    if os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        EVALUATION_PIPELINE.append(("google/gemma-4-26b-a4b-it", "fp8", 8192, gemma_adapter))

    EVALUATION_PIPELINE.append(("meta-llama/Llama-3.1-8B-Instruct", None, 8192, "tschomacker/lora_adapter_llama_3.1_8B"))
    
    output_json = {
        "system": global_system_prompt,
        "template": global_template,
        "models": [],
        "prompts": []
    }
    
    # Pre-populate matrix rows with uniform 4-element columns [Text, FRE, WSTF, Reasoning]
    for item in evaluation_set:
        record = {}
        if item["is_integrity"]:
            record["system"] = item["system"]
            record["template"] = ""
            
        input_fre, input_wstf = get_raw_metrics(item["original_user"])
        record["r"] = [[item["original_user"], input_fre, input_wstf, ""]]
        output_json["prompts"].append(record)

    # Cascade Batch Inference through registered models
    for model_id, quant_type, max_len, adapter_id in EVALUATION_PIPELINE:
        display_name = model_id
        if adapter_id: display_name += f" ({adapter_id})"
        output_json["models"].append(display_name)
        
        responses = run_evaluation(model_id, quant_type, max_len, adapter_id, evaluation_set)
        
        for idx, (text_response, reasoning_trace) in enumerate(responses):
            resp_fre, resp_wstf = get_raw_metrics(text_response)
            output_json["prompts"][idx]["r"].append([text_response, resp_fre, resp_wstf, reasoning_trace])

    # POST-PROCESSING: Append tuning ground truth references
    print("\n📝 Appending ground-truth training references to dataset records...", flush=True)
    for idx, item in enumerate(evaluation_set):
        if item["reference_text"] is not None:
            ref_txt = item["reference_text"]
            ref_fre, ref_wstf = get_raw_metrics(ref_txt)
            output_json["prompts"][idx]["r"].append([ref_txt, ref_fre, ref_wstf, ""])

    # Output Serialization and S3 Upload
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    bucket_name = os.environ.get("S3_BUCKET")
    json_payload = json.dumps(output_json, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60 + "\n🏁 EVALUATION MATRIX EXPORT\n" + "="*60)
    if bucket_name:
        print(f"📤 Exporting matrix results to S3 Bucket: {bucket_name} as {filename}...")
        try:
            s3_client = boto3.client('s3')
            s3_client.put_object(Bucket=bucket_name, Key=filename, Body=json_payload, ContentType='application/json')
            print("🚀 S3 Synchronization successful!")
        except Exception as s3_err:
            print(f"❌ Failed to transfer to S3 destination: {s3_err}\nSaving local copy as fallback...")
            with open(filename, "w", encoding="utf-8") as f: f.write(json_payload)
    else:
        with open(filename, "w", encoding="utf-8") as f: f.write(json_payload)
        print(f"💾 Written to local file system: {filename}")

# def apply_vllm_mla_hotfix():
#     """Automated hotfix patch for an active vLLM regression (Issue #43263)."""
#     target_file = "/workspace/axolotl-venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py"
#     if os.path.exists(target_file):
#         with open(target_file, "r", encoding="utf-8") as f: code = f.read()
#         broken_string = "kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)"
#         fixed_string  = "kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)"
#         if broken_string in code:
#             with open(target_file, "w", encoding="utf-8") as f: f.write(code.replace(broken_string, fixed_string))

if __name__ == "__main__":
    # apply_vllm_mla_hotfix()
    main()