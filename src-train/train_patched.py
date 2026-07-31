# --- src/train_patched.py ---
import os
import sys
import fire

# Apple Silicon only, and it must precede `import axolotl`: Axolotl eagerly imports a
# Triton-only module that cannot exist on macOS. No-op on RunPod. See mps_patch.py.
from mps_patch import apply_mps_device_map_patch, install_triton_stub

install_triton_stub()

# Import target modules
import axolotl.train
import axolotl.cli.train

# Also Apple Silicon only: Axolotl forces a device_map spelling that hangs on Metal.
apply_mps_device_map_patch()

# --- Apply tokenizer patch to avoid mistral-common validation errors ---
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    
    # Define official Mistral template as fallback if the model config is missing it or loaded via MistralCommonTokenizer
    MISTRAL_DEFAULT_CHAT_TEMPLATE = (
        "{%- set default_system_message = '' %}"
        "{{- '<s>' }}"
        "{%- if messages[0]['role'] == 'system' %}"
        "    {{- '[SYSTEM_PROMPT]' -}}"
        "    {%- if messages[0]['content'] is string %}"
        "        {{- messages[0]['content'] -}}"
        "    {%- else %}"
        "        {%- for block in messages[0]['content'] %}"
        "            {%- if block['type'] == 'text' %}"
        "                {{- block['text'] }}"
        "            {%- else %}"
        "                {{- raise_exception('Only text chunks are supported in system message contents.') }}"
        "            {%- endif %}"
        "        {%- endfor %}"
        "    {%- endif %}"
        "    {{- '[/SYSTEM_PROMPT]' -}}"
        "    {%- set loop_messages = messages[1:] %}"
        "{%- else %}"
        "    {%- set loop_messages = messages %}"
        "    {%- if default_system_message != '' %}"
        "        {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}"
        "    {%- endif %}"
        "{%- endif %}"
        "{%- set tools_definition = '' %}"
        "{%- set has_tools = false %}"
        "{%- if tools is defined and tools is not none and tools|length > 0 %}"
        "    {%- set has_tools = true %}"
        "    {%- set tools_definition = '[AVAILABLE_TOOLS]' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}"
        "    {{- tools_definition }}"
        "{%- endif %}"
        "{%- set reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'none' %}"
        "{%- if reasoning_effort not in ['none', 'high'] %}"
        "    {{- raise_exception('reasoning_effort must be either \"none\" or \"high\"') }}"
        "{%- endif %}"
        "{%- set model_settings = '[MODEL_SETTINGS]{\"reasoning_effort\": \"' + reasoning_effort + '\"}[/MODEL_SETTINGS]' %}"
        "{{- model_settings }}"
        "{%- set ns = namespace(index=0) %}"
        "{%- for message in loop_messages %}"
        "    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}"
        "        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}"
        "            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.') }}"
        "        {%- endif %}"
        "        {%- set ns.index = ns.index + 1 %}"
        "    {%- endif %}"
        "{%- endfor %}"
        "{%- for message in loop_messages %}"
        "    {%- if message['role'] == 'user' %}"
        "        {%- if message['content'] is string %}"
        "            {{- '[INST]' + message['content'] + '[/INST]' }}"
        "        {%- elif message['content'] | length > 0 %}"
        "            {{- '[INST]' }}"
        "            {%- if message['content'] | length == 2 %}"
        "                {%- set blocks = message['content'] | sort(attribute='type') %}"
        "            {%- else %}"
        "                {%- set blocks = message['content'] %}"
        "            {%- endif %}"
        "            {%- for block in blocks %}"
        "                {%- if block['type'] == 'text' %}"
        "                    {{- block['text'] }}"
        "                {%- elif block['type'] in ['image', 'image_url'] %}"
        "                    {{- '[IMG]' }}"
        "                {%- else %}"
        "                    {{- raise_exception('Only text, image and image_url chunks are supported in user message content.') }}"
        "                {%- endif %}"
        "            {%- endfor %}"
        "            {{- '[/INST]' }}"
        "        {%- else %}"
        "            {{- raise_exception('User message must have a string or a list of chunks in content') }}"
        "        {%- endif %}"
        "    {%- elif message['role'] == 'assistant' %}"
        "        {%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}"
        "            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}"
        "        {%- endif %}"
        "        {%- if message['content'] is string and message['content'] != '' %}"
        "            {{- message['content'] }}"
        "        {%- elif message['content'] | length > 0 %}"
        "            {%- for block in message['content'] %}"
        "                {%- if block['type'] == 'text' %}"
        "                    {{- block['text'] }}"
        "                {%- elif block['type'] == 'thinking' %}"
        "                    {{- '[THINK]' + block['thinking'] + '[/THINK]' }}"
        "                {%- else %}"
        "                    {{- raise_exception('Only text and thinking chunks are supported in assistant message contents.') }}"
        "                {%- endif %}"
        "            {%- endfor %}"
        "        {%- endif %}"
        "        {%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}"
        "            {%- for tool in message['tool_calls'] %}"
        "                {{- '[TOOL_CALLS]' }}"
        "                {%- set name = tool['function']['name'] %}"
        "                {%- set arguments = tool['function']['arguments'] %}"
        "                {%- if arguments is not string %}"
        "                    {%- set arguments = arguments|tojson|safe %}"
        "                {%- elif arguments == '' %}"
        "                    {%- set arguments = '{}' %}"
        "                {%- endif %}"
        "                {{- name + '[ARGS]' + arguments }}"
        "            {%- endfor %}"
        "        {%- endif %}"
        "        {{- '</s>' }}"
        "    {%- elif message['role'] == 'tool' %}"
        "        {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}"
        "    {%- else %}"
        "        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role'] + '.') }}"
        "    {%- endif %}"
        "{%- endfor %}"
    )

    # Dynamic getter/setter property for chat_template to avoid "null tokenizer chat_template" checks in Axolotl
    def get_chat_template(self):
        val = getattr(self, "_chat_template", None)
        if val is None:
            class_name = self.__class__.__name__
            model_name = getattr(self, "name_or_path", "")
            if "Mistral" in class_name or "mistral" in str(model_name).lower() or "TokenizersBackend" in class_name:
                return MISTRAL_DEFAULT_CHAT_TEMPLATE
        return val

    def set_chat_template(self, value):
        self._chat_template = value

    PreTrainedTokenizerBase.chat_template = property(get_chat_template, set_chat_template)
    print("🔧 MONKEYPATCH: Dynamically injected fallback chat_template property getter on PreTrainedTokenizerBase")

    # Patch save_pretrained to pop save_jinja_files parameter for Mistral tokenizers
    original_base_save_pretrained = PreTrainedTokenizerBase.save_pretrained
    def patched_base_save_pretrained(self, *args, **kwargs):
        class_name = self.__class__.__name__
        if "Mistral" in class_name or "TokenizersBackend" in class_name:
            kwargs.pop("save_jinja_files", None)
        return original_base_save_pretrained(self, *args, **kwargs)
    PreTrainedTokenizerBase.save_pretrained = patched_base_save_pretrained
    print("🔧 MONKEYPATCH: Successfully patched PreTrainedTokenizerBase.save_pretrained")

    def make_patched_save_pretrained(original_save_fn):
        def patched_save_pretrained(self, *args, **kwargs):
            kwargs.pop("save_jinja_files", None)
            return original_save_fn(self, *args, **kwargs)
        return patched_save_pretrained

    # Patch MistralCommonTokenizer if present
    try:
        from transformers.tokenization_mistral_common import MistralCommonTokenizer
        MistralCommonTokenizer.apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
        MistralCommonTokenizer.get_chat_template = PreTrainedTokenizerBase.get_chat_template
        if hasattr(MistralCommonTokenizer, "save_pretrained"):
            MistralCommonTokenizer.save_pretrained = make_patched_save_pretrained(MistralCommonTokenizer.save_pretrained)
            print("🔧 MONKEYPATCH: Successfully patched MistralCommonTokenizer.save_pretrained")
        print("🔧 MONKEYPATCH: Successfully patched MistralCommonTokenizer.apply_chat_template and get_chat_template")
    except ImportError:
        pass

    # Patch TokenizersBackend if present
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        TokenizersBackend.apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
        TokenizersBackend.get_chat_template = PreTrainedTokenizerBase.get_chat_template
        if hasattr(TokenizersBackend, "save_pretrained"):
            TokenizersBackend.save_pretrained = make_patched_save_pretrained(TokenizersBackend.save_pretrained)
            print("🔧 MONKEYPATCH: Successfully patched TokenizersBackend.save_pretrained")
        print("🔧 MONKEYPATCH: Successfully patched TokenizersBackend.apply_chat_template and get_chat_template")
    except ImportError:
        pass
except Exception as e:
    print(f"⚠️ Warning: Failed to apply tokenizer monkeypatch: {e}")

# --- Apply quantization validation patch to allow training FP8 models under LoRA ---
try:
    import transformers.trainer_utils
    import transformers.trainer
    
    def dummy_validate_quantization_for_training(model):
        print("🔧 MONKEYPATCH: Bypassed validate_quantization_for_training for FP8 model")
        return
        
    transformers.trainer_utils.validate_quantization_for_training = dummy_validate_quantization_for_training
    transformers.trainer.validate_quantization_for_training = dummy_validate_quantization_for_training
    print("🔧 MONKEYPATCH: Successfully bypassed validate_quantization_for_training")
except Exception as e:
    print(f"⚠️ Warning: Failed to apply quantization validation monkeypatch: {e}")

# --- ROPE_DEBUG=1: report the shapes entering the rotary embedding, once ---
# Every evaluation so far has died at
#     modeling_gemma4.py:1170  freqs = inv_freq_expanded.float() @ position_ids_expanded.float()
#     RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE ... cublasSgemm
# on four hardware/attention combinations. CUBLAS_STATUS_INVALID_VALUE is a host-side
# argument check, so one of m/n/k is zero -- but which is a guess until someone prints it.
# m comes from inv_freq, n from position_ids. Under ZeRO-3 an ungathered parameter has a
# zero-element .data, which is the shape this would take if the eval forward is bypassing
# the DeepSpeed engine (its frame is absent from the traceback).
if os.environ.get("ROPE_DEBUG") == "1":
    import importlib
    import inspect

    _rope_seen = []

    def _make_rope_reporter(label, original):
        def reporting_forward(self, x, position_ids, *args, **kwargs):
            if not _rope_seen:
                _rope_seen.append(True)
                parts = [f"class={label}", f"extra_args={args}"]
                for name, tensor in (("x", x), ("position_ids", position_ids)):
                    try:
                        parts.append(f"{name} shape={tuple(tensor.shape)} "
                                     f"numel={tensor.numel()} dtype={tensor.dtype} "
                                     f"dev={tensor.device}")
                    except Exception as exc:  # noqa: BLE001 -- a report must not add a failure
                        parts.append(f"{name} UNREADABLE ({exc})")
                for name, buf in self.named_buffers(recurse=False):
                    parts.append(f"BUF {name} shape={tuple(buf.shape)} "
                                 f"numel={buf.numel()} dev={buf.device}")
                for name, param in self.named_parameters(recurse=False):
                    parts.append(f"PARAM {name} shape={tuple(param.shape)} "
                                 f"numel={param.numel()} "
                                 f"ds_status={getattr(param, 'ds_status', 'n/a')} "
                                 f"ds_shape={getattr(param, 'ds_shape', 'n/a')}")

                # The first report showed m=256, n=1600, k=1 on the right device -- valid
                # arguments that cuBLAS rejected anyway. So the question is no longer "what
                # is wrong with these tensors" but "is cuBLAS working at all here". A 2x2
                # matmul on the same device answers that: if it fails too, the context is
                # already broken and the rotary embedding is an innocent bystander.
                import torch as _torch

                parts.append(f"current_device=cuda:{_torch.cuda.current_device()}")
                parts.append(f"x.stride={x.stride()} pos.stride={position_ids.stride()}")
                parts.append(f"alloc={_torch.cuda.memory_allocated(x.device) >> 20}MiB "
                             f"reserved={_torch.cuda.memory_reserved(x.device) >> 20}MiB")
                # The failing call is cublasSgemm -- SINGLE precision. Everything else in
                # this model is bf16, and the rotary embedding is the first thing to force
                # fp32 (maybe_autocast(enabled=False) plus .float()). TF32 is a mode that
                # applies to exactly that: fp32 matmuls on tensor cores. config/base.yml
                # sets tf32: true, and the image is a *mutable* `main` tag on CUDA 13.0, so
                # a regression there would show up here and nowhere else.
                #
                # tf32 OFF is tried FIRST, while the CUDA context is still clean: a failure
                # can leave it in a state where anything afterwards fails regardless.
                def _try_fp32_matmul():
                    probe = _torch.ones(2, 2, device=x.device, dtype=_torch.float32)
                    return (probe @ probe).sum().item()

                _tf32_was = _torch.backends.cuda.matmul.allow_tf32
                parts.append(f"tf32_allowed={_tf32_was} "
                             f"precision={_torch.get_float32_matmul_precision()}")
                for _label, _setting in (("tf32_OFF", False), ("tf32_AS_CONFIGURED", _tf32_was)):
                    _torch.backends.cuda.matmul.allow_tf32 = _setting
                    try:
                        _try_fp32_matmul()
                        parts.append(f"FP32_MATMUL[{_label}]=ok")
                    except Exception as exc:  # noqa: BLE001
                        parts.append(f"FP32_MATMUL[{_label}]=FAILED "
                                     f"{type(exc).__name__}: {str(exc)[:90]}")
                _torch.backends.cuda.matmul.allow_tf32 = _tf32_was
                print("🔬 ROPE FIRST CALL || " + " || ".join(parts), flush=True)
            return original(self, x, position_ids, *args, **kwargs)
        return reporting_forward

    # Discovered, not guessed. `Gemma4RotaryEmbedding` was a guess and the class is called
    # something else, so the reporter silently did nothing for a whole pod run. Scan the
    # module and print what is actually there, so a miss is visible rather than quiet.
    for _mod_name in ("transformers.models.gemma4.modeling_gemma4",
                      "transformers.models.gemma4_unified.modeling_gemma4_unified"):
        try:
            _mod = importlib.import_module(_mod_name)
        except Exception as e:  # noqa: BLE001
            print(f"ℹ️  ROPE_DEBUG: {_mod_name} not importable ({e})")
            continue
        _classes = [(n, o) for n, o in vars(_mod).items()
                    if inspect.isclass(o) and "rotary" in n.lower() and hasattr(o, "forward")]
        if not _classes:
            print(f"⚠️  ROPE_DEBUG: no *Rotary* class in {_mod_name}. Classes present: "
                  f"{sorted(n for n, o in vars(_mod).items() if inspect.isclass(o))}")
            continue
        for _name, _cls in _classes:
            _cls.forward = _make_rope_reporter(f"{_mod_name}.{_name}", _cls.forward)
            print(f"🔬 MONKEYPATCH: ROPE_DEBUG reporting on {_mod_name}.{_name}")

original_train = axolotl.train.train

def patched_train(cfg, *args, **kwargs):
    print("\n" + "="*60)
    print("🔧 MONKEYPATCH: Overriding gradient_checkpointing_kwargs to use_reentrant=True")
    print("This bypasses the DeepSpeed ZeRO-3 parameter sharding metadata mismatch.")
    print("="*60 + "\n", flush=True)
    
    if hasattr(cfg, "gradient_checkpointing_kwargs") and cfg.gradient_checkpointing_kwargs:
        cfg.gradient_checkpointing_kwargs["use_reentrant"] = True
    else:
        cfg.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
    return original_train(cfg, *args, **kwargs)

# Apply monkeypatch globally
axolotl.train.train = patched_train
if hasattr(axolotl.cli.train, "train"):
    axolotl.cli.train.train = patched_train

if __name__ == "__main__":
    fire.Fire(axolotl.cli.train.do_cli)
