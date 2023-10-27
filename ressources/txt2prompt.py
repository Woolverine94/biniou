# https://github.com/Woolverine94/biniou
# txt2prompt.py
import gradio as gr
import os
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from ressources.common import *

device_txt2prompt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def text_txt2prompt(
    modelid_txt2prompt, 
    max_tokens_txt2prompt, 
    repetition_penalty_txt2prompt,
    prompt_txt2prompt, 
    output_type_txt2prompt, 
    progress_txt2prompt=gr.Progress(track_tqdm=True)
    ):
          
    if output_type_txt2prompt == "ChatGPT":   
         prompt_txt2prompt = f"\nAction:{prompt_txt2prompt}\nPrompt:"

    if output_type_txt2prompt == "ChatGPT": 
        pipeline_txt2prompt = pipeline(
            task="text-generation",
            model=modelid_txt2prompt,
            local_files_only=True if offline_test() else None                
        )
        output_txt2prompt = pipeline_txt2prompt(prompt_txt2prompt, do_sample=True, max_new_tokens=max_tokens_txt2prompt)[0]["generated_text"]
        output_txt2prompt = output_txt2prompt.replace(prompt_txt2prompt,"")

        filename_txt2prompt = write_file(output_txt2prompt)
    
        del pipeline_txt2prompt
        clean_ram()
        
    elif output_type_txt2prompt == "SD":
        tokenizer_txt2prompt = GPT2Tokenizer.from_pretrained(
            modelid_txt2prompt, 
            cache_dir=model_path_txt2prompt,
            use_safetensors=True, 
            resume_download=True, 
            local_files_only=True if offline_test() else None
        )
        automodel_txt2prompt = GPT2LMHeadModel.from_pretrained(
            modelid_txt2prompt,
            cache_dir=model_path_txt2prompt,
            use_safetensors=True,         
            resume_download=True,
            local_files_only=True if offline_test() else None                
        )
        inputs_txt2prompt = tokenizer_txt2prompt(prompt_txt2prompt, return_tensors="pt").to(device_txt2prompt) 	
        input_ids_txt2prompt = inputs_txt2prompt.input_ids.to(device_txt2prompt)
        attention_mask_txt2prompt = inputs_txt2prompt.attention_mask.to(device_txt2prompt)
        generated_ids_txt2prompt = automodel_txt2prompt.generate(
            input_ids_txt2prompt, 
            attention_mask=attention_mask_txt2prompt, 
            repetition_penalty=repetition_penalty_txt2prompt, 
            max_length=max_tokens_txt2prompt, 
            eos_token_id=tokenizer_txt2prompt.eos_token_id
        )
        output_txt2prompt = tokenizer_txt2prompt.decode(generated_ids_txt2prompt[0], skip_special_tokens=False)

        filename_txt2prompt = write_file(output_txt2prompt)
    
        del tokenizer_txt2prompt, automodel_txt2prompt
        clean_ram()

    return output_txt2prompt
