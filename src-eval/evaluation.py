# --- src/evaluation.py ---
import os
import re
import gc
import sys
import json
import time
from datetime import datetime, UTC
from pathlib import Path
import torch
import boto3
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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

def run_evaluation(model_id, quantization_type, max_len=8192, adapter_id=None, evaluation_set=None, reasoning_parser=None, gpu_memory_utilization=0.9):
    """
    Initializes the vLLM engine and processes conversations.
    """
    if evaluation_set is None:
        evaluation_set = []
        
    print("\n" + "="*60)
    print(f"🚀 LOADING MODEL FOR BATCH EVALUATION (vLLM): {model_id}")
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

        # Configure vLLM Engine
        engine_args = {
            "model": model_id,
            "tensor_parallel_size": available_gpus,
            "max_model_len": max_len,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        
        if quantization_type:
            engine_args["quantization"] = quantization_type
            
        if adapter_id:
            engine_args["enable_lora"] = True
            engine_args["max_loras"] = 1
            
        print(f"⚙️ Initializing vLLM Engine with parameters: {engine_args}", flush=True)
        llm = LLM(**engine_args)
        
        sampling_params = SamplingParams(
            temperature=1.0 if is_gemma else 0.3,
            top_p=0.95,
            max_tokens=4096,
            skip_special_tokens=False
        )
            
        lora_request = None
        if adapter_id:
            lora_request = LoRARequest("adapter0", 1, adapter_id)
            
        print(f"⚡ Processing {len(prompts)} prompts via vLLM Engine...", flush=True)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        
        for out in outputs:
            raw_text = out.outputs[0].text.strip()
            reasoning_trace = ""
            
            # Robust regex extraction for reasoning trace
            # Pattern catches common Gemma/Llama reasoning delimiters
            think_pattern = r"(?:<\|channel>thought\n|<\|channel\|>thought|<\|thought\|>|<(?:think|thought)>|\[(?:think|thought)\])(.*?)(?:<\|channel\|>|<channel\|>|</(?:think|thought)>|\[/(?:think|thought)\]|$)"
            think_match = re.search(think_pattern, raw_text, re.DOTALL | re.IGNORECASE)
            
            if think_match:
                reasoning_trace = think_match.group(1).strip()
                # Clean the text
                raw_text = re.sub(think_pattern, "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
                
            generated_responses.append((raw_text, reasoning_trace))
            
    except Exception as e:
        print(f"❌ Execution error encountered on {model_id}: {e}", flush=True)
        generated_responses = [("", "") for _ in evaluation_set]
        
    finally:
        print(f"♻️ Clearing VRAM for next model...", flush=True)
        if 'llm' in locals():
            # vLLM doesn't have a direct shutdown() like SGLang, 
            # we rely on gc and torch.cuda.empty_cache()
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ VRAM cleared.", flush=True)
        
    return generated_responses

def main():
    # 1. LOAD CONFIGS
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
    gemma_adapter = "/app/output/adapter/gemma4"
    if not os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        gemma_adapter = "/app/output/adapter/train-gemma4"

    EVALUATION_PIPELINE = []
    
    # Model 1: Gemma Plain
    base_gemma = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
    EVALUATION_PIPELINE.append((base_gemma, None, 8192, None, "Gemma 4 (Plain)", None))
    
    # Model 2: Gemma Reasoning
    EVALUATION_PIPELINE.append((base_gemma, None, 8192, None, "Gemma 4 (Reasoning)", "gemma4"))
    
    # Model 3: Gemma Fine-tuned (Merged or Adapter)
    merged_gemma = "/app/output/merged/train-gemma4-fp8"
    if os.path.exists(merged_gemma):
        EVALUATION_PIPELINE.append((merged_gemma, None, 8192, None, "Gemma 4 (Fine-tuned)", "gemma4"))
    elif os.path.exists(os.path.join(gemma_adapter, "adapter_config.json")):
        EVALUATION_PIPELINE.append((base_gemma, None, 8192, gemma_adapter, "Gemma 4 (Fine-tuned)", "gemma4"))

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
            gpu_memory_utilization=0.9
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