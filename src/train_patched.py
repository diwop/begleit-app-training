# --- src/train_patched.py ---
import sys
import fire

# Import target modules
import axolotl.train
import axolotl.cli.train

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

    # Patch MistralCommonTokenizer if present
    try:
        from transformers.tokenization_mistral_common import MistralCommonTokenizer
        MistralCommonTokenizer.apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
        MistralCommonTokenizer.get_chat_template = PreTrainedTokenizerBase.get_chat_template
        print("🔧 MONKEYPATCH: Successfully patched MistralCommonTokenizer.apply_chat_template and get_chat_template")
    except ImportError:
        pass

    # Patch TokenizersBackend if present
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        TokenizersBackend.apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
        TokenizersBackend.get_chat_template = PreTrainedTokenizerBase.get_chat_template
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
