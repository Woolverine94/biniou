# https://github.com/Woolverine94/biniou
# txt2prompt.py
import gradio as gr
import os
import random
from transformers import pipeline, set_seed
from ressources.common import *

device_label_txt2prompt, model_arch = detect_device()
device_txt2prompt = torch.device(device_label_txt2prompt)

model_path_txt2prompt = "./models/Prompt_generator/"
os.makedirs(model_path_txt2prompt, exist_ok=True)

model_list_txt2prompt = []

for filename in os.listdir(model_path_txt2prompt):
    f = os.path.join(model_path_txt2prompt, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.bin')):
        model_list_txt2prompt.append(f)

model_list_txt2prompt_builtin = [
    "PulsarAI/prompt-generator",
    "RamAnanth1/distilgpt2-sd-prompts",
]

for k in range(len(model_list_txt2prompt_builtin)):
    model_list_txt2prompt.append(model_list_txt2prompt_builtin[k])

# Bouton Cancel
stop_txt2prompt = False

def initiate_stop_txt2prompt() :
    global stop_txt2prompt
    stop_txt2prompt = True

def check_txt2prompt(step, timestep, latents) :
    global stop_txt2prompt
    if stop_txt2prompt == False :
        return
    elif stop_txt2prompt == True :
        stop_txt2prompt = False
        try:
            del ressources.txt2prompt.pipe_txt2prompt
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def text_txt2prompt(
    modelid_txt2prompt, 
    max_tokens_txt2prompt, 
    repetition_penalty_txt2prompt,
    seed_txt2prompt, 
    num_prompt_txt2prompt,
    prompt_txt2prompt, 
    output_type_txt2prompt, 
    progress_txt2prompt=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Prompt generator 📝 ]: starting module")
    output_txt2prompt=""
    prompt_txt2prompt_origin = prompt_txt2prompt

    if output_type_txt2prompt == "ChatGPT":   
         prompt_txt2prompt = f"\nAction:{prompt_txt2prompt}\nPrompt:"

    if seed_txt2prompt == 0:
        seed_txt2prompt = random.randint(0, 4294967295)

    pipe_txt2prompt = pipeline(
        task="text-generation",
        model=modelid_txt2prompt,
        torch_dtype=model_arch,
        device=device_txt2prompt,
        local_files_only=True if offline_test() else None 
    )
    
    set_seed(seed_txt2prompt)

    generator_txt2prompt = pipe_txt2prompt(
        prompt_txt2prompt, 
        do_sample=True, 
        max_new_tokens=max_tokens_txt2prompt,
        num_return_sequences=num_prompt_txt2prompt, 
    )

    for i in range(len(generator_txt2prompt)):
        output_txt2prompt_int = generator_txt2prompt[i]["generated_text"]
        if output_type_txt2prompt == "ChatGPT": 
            output_txt2prompt_int = output_txt2prompt_int.replace(prompt_txt2prompt,"")
        output_txt2prompt += output_txt2prompt_int+ "\n\n"

    output_txt2prompt = output_txt2prompt.rstrip()
    filename_txt2prompt = write_seeded_file(seed_txt2prompt, output_txt2prompt)

    print(f">>>[Prompt generator 📝 ]: generated {num_prompt_txt2prompt} prompt(s)")
    reporting_txt2prompt = f">>>[Prompt generator 📝 ]: "+\
        f"Settings : Model={modelid_txt2prompt} | "+\
        f"Max tokens={max_tokens_txt2prompt} | "+\
        f"Repetition penalty={repetition_penalty_txt2prompt} | "+\
        f"Output type={output_type_txt2prompt} | "+\
        f"Prompt={prompt_txt2prompt_origin} | "+\
        f"Initial seed={seed_txt2prompt}"
    print(reporting_txt2prompt)

    del pipe_txt2prompt
    clean_ram()

    print(f">>>[Prompt generator 📝 ]: leaving module")
    return output_txt2prompt
