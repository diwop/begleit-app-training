Phase 1: High-Precision Weights MergeSince both your base model and the Axolotl-tuned adapter are in standard $\text{BF16}$ precision, you can merge them on a single GPU (or high-RAM CPU instance) using Hugging Face's PEFT wrapper. This consolidates the parameters into a standard Gemma4ForConditionalGeneration directory.Create a script named merge_weights.py:

import os
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from peft import PeftModel

def merge_gemma4_lora(base_model_id: str, adapter_dir: str, output_dir: str):
    print(f"Loading base model in BF16: {base_model_id}")
    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("Loading processor configurations...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    
    print(f"Attaching LoRA adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=torch.bfloat16
    )
    
    print("Unloading PEFT towers and merging weights...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving unquantized consolidated model to: {output_dir}")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    processor.save_pretrained(output_dir)
    print("Weight merging completed successfully.")

if __name__ == "__main__":
    merge_gemma4_lora(
        base_model_id="google/gemma-4-26B-A4B-it",
        adapter_dir="./outputs/gemma4-lora-checkpoint",
        output_dir="./models/merged-gemma4-bf16"
    )

Phase 2: Post-Training FP8 QuantizationOnce you have your merged $\text{BF16}$ model on disk, you can compress it using the LLM Compressor library (developed by Neural Magic and the vLLM project).  To achieve high-quality $\text{FP8}$ quantization, we utilize the FP8_DYNAMIC scheme. This applies static per-channel quantization to the weights, while scaling the activations dynamically per-token during inference. Because the activation scale factors are computed on-the-fly, this quantization pipeline is completely data-free and requires no calibration datasets.  Crucial Architectural Safeguard:When quantizing Gemma 4 26B-A4B, you must instruct the quantizer to skip the MoE routers, normalization layers, token embeddings, and the vision tower. If you attempt to quantize the MoE routing gates or normalization parameters, the model's routing mechanism will collapse, severely degrading its reasoning capabilities.  Install the prerequisites inside your compression environment:

pip install "git+https://github.com/vllm-project/llm-compressor.git@main"
pip install "transformers>=5.8.1"

Create and run a script named quantize_fp8.py:

import os
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def run_fp8_compression(merged_bf16_dir: str, output_fp8_dir: str):
    print(f"Loading merged BF16 model: {merged_bf16_dir}")
    # Load model and processor using Gemma 4's native conditional generation classes
    model = Gemma4ForConditionalGeneration.from_pretrained(
        merged_bf16_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(merged_bf16_dir)

    # Configure the FP8_DYNAMIC scheme targeting linear projections
    # CRITICAL: Add target exclusions to preserve MoE routing and embedding accuracy
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=
    )

    print("Applying post-training FP8 quantization via one-shot API...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving quantized weights in compressed-tensors format to: {output_fp8_dir}")
    # save_compressed=True writes packed FP8 values on disk to achieve ~50% VRAM reductions
    model.save_pretrained(output_fp8_dir, save_compressed=True)
    processor.save_pretrained(output_fp8_dir)
    print("FP8 Quantization process successfully completed!")

if __name__ == "__main__":
    run_fp8_compression(
        merged_bf16_dir="./models/merged-gemma4-bf16",
        output_fp8_dir="./models/merged-gemma4-fp8"
    )

Phase 3: Serving the Merged FP8 Model on SGLang
Because your adapter is already baked into the quantized base weights, you do not need the SGLang --enable-lora flags at runtime. This completely bypasses the regex key-parsing crashes, the multi-LoRA CUDA Graph memory leaks, and any missing LoRA layer wrapper errors (like the ClippableRowParallelLinear exception).  

However, because Gemma 4 uses a highly heterogeneous attention mechanism (local sliding window attention layers interleaved with global full-attention layers using different head dimensions), you must still explicitly configure the Triton attention backend.  

Launch your newly minted FP8 model using SGLang with the following optimized production parameters:

python3 -m sglang.launch_server \
    --model-path./models/merged-gemma4-fp8 \
    --attention-backend triton \
    --tp-size 2 \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.90