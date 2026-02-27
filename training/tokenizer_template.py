import json
import os
import re


START_MARKER = "{#- chem_llm_default_system_prompt_start -#}"
END_MARKER = "{#- chem_llm_default_system_prompt_end -#}"

# https://huggingface.co/docs/transformers/en/chat_templating_writing
#https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-2-chat.jinja
def marker_block(system_prompt):
    quoted = json.dumps(system_prompt)
    return (
        f"    {START_MARKER}\n"
        f"    {{%- set system_message = {quoted} %}}\n"
        f"    {END_MARKER}"
    )

def insert_default_system_prompt(chat_template, system_prompt): # puts a new system prompt into chat template
    prompt = (system_prompt or "").strip()
    if not prompt:
        return chat_template

    block = marker_block(prompt)
    if START_MARKER in chat_template and END_MARKER in chat_template:
        pattern = re.compile(
            rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}", # find the block between the start and end markers and replace it with the new block
            flags=re.DOTALL, # search for the in-between text of the markers and later replace with the block
        )
        return pattern.sub(lambda x: block.strip(), chat_template, count=1) # if you find the pattern then replace it with the block and do it only once

    # old method but can still use as a fallback
    target = '{%- set system_message = "" %}'
    if target in chat_template:
        return chat_template.replace(target, block.strip(), 1)

    return chat_template


# some of this might not be used anymore but was done previously 
def patch_text_file(path, system_prompt):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    updated = insert_default_system_prompt(original, system_prompt)
    if updated == original:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return True

def patch_tokenizer_config(tokenizer_config_path, system_prompt):
    if not os.path.exists(tokenizer_config_path):
        return False

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_template = data.get("chat_template")
    if not isinstance(chat_template, str):
        return False

    updated = insert_default_system_prompt(chat_template, system_prompt)
    if updated == chat_template:
        return False

    data["chat_template"] = updated
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True

def patch_chat_template_jinja(chat_template_path, system_prompt):
    return patch_text_file(chat_template_path, system_prompt)
