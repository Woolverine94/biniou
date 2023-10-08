# https://github.com/Woolverine94/biniou
# llamacpp.py
from llama_cpp import Llama
import gradio as gr
import os
from huggingface_hub import snapshot_download, hf_hub_download
from ressources.common import *

# Gestion des modèles
model_path_llamacpp = "./models/llamacpp/"
os.makedirs(model_path_llamacpp, exist_ok=True)

model_list_llamacpp = {}

for filename in os.listdir(model_path_llamacpp):
    f = os.path.join(model_path_llamacpp, filename)
    if os.path.isfile(f) and filename.endswith('.gguf') :
        model_list_llamacpp.update(f)

model_list_llamacpp_builtin = {
    "TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF": "collectivecognition-v1.1-mistral-7b.Q5_K_S.gguf", 
    "TheBloke/Airoboros-L2-13B-2.1-GGUF": "airoboros-l2-13b-2.1.Q5_K_S.gguf", 
    "TheBloke/Airoboros-L2-7B-2.1-GGUF": "airoboros-l2-7b-2.1.Q5_K_S.gguf", 
    "TheBloke/Vigogne-2-13B-Instruct-GGUF": "vigogne-2-13b-instruct.Q5_K_S.gguf", 
    "TheBloke/Vigogne-2-7B-Instruct-GGUF": "vigogne-2-7b-instruct.Q5_K_S.gguf", 
    "TheBloke/CodeLlama-13B-Instruct-GGUF": "codellama-13b-instruct.Q5_K_S.gguf", 
}

model_list_llamacpp.update(model_list_llamacpp_builtin)

def download_model(modelid_llamacpp):
    if modelid_llamacpp[0:9] != "./models/":
        hf_hub_path_llamacpp = hf_hub_download(
            repo_id=modelid_llamacpp, 
            filename=model_list_llamacpp[modelid_llamacpp], 
            repo_type="model", 
            cache_dir=model_path_llamacpp, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        modelid_llamacpp = hf_hub_path_llamacpp
    return modelid_llamacpp        

def text_llamacpp(
    modelid_llamacpp, 
    max_tokens_llamacpp, 
    seed_llamacpp, 
    stream_llamacpp, 
    n_ctx_llamacpp, 
    repeat_penalty_llamacpp, 
    temperature_llamacpp, 
    top_p_llamacpp, 
    top_k_llamacpp, 
    prompt_llamacpp, 
    history_llamacpp, 
    progress_txt2vid_ze=gr.Progress(track_tqdm=True)
    ):

    modelid_llamacpp = download_model(modelid_llamacpp)

    if history_llamacpp != "" :
        prompt_final_llamacpp = f"{history_llamacpp}\n\n{prompt_llamacpp}"
    else :
        prompt_final_llamacpp = prompt_llamacpp

    llm = Llama(model_path=modelid_llamacpp,  seed=seed_llamacpp, n_ctx=n_ctx_llamacpp)
    output_llamacpp = llm(
        f"{prompt_final_llamacpp}\n", 
        max_tokens=max_tokens_llamacpp, 
        stream=stream_llamacpp, 
        repeat_penalty=repeat_penalty_llamacpp, 
        temperature=temperature_llamacpp, 
        top_p=top_p_llamacpp, 
        top_k=top_k_llamacpp, 
        echo=True
    )    
    
    answer_llamacpp = (output_llamacpp['choices'][0]['text'])
    last_answer_llamacpp = answer_llamacpp.replace(f"{prompt_final_llamacpp}\n", "")
    write_file(answer_llamacpp)
    
    del llm, output_llamacpp
    clean_ram()
    
    return last_answer_llamacpp, answer_llamacpp
    
def text_llamacpp_continue(
    modelid_llamacpp, 
    max_tokens_llamacpp, 
    seed_llamacpp, 
    stream_llamacpp, 
    n_ctx_llamacpp, 
    repeat_penalty_llamacpp, 
    temperature_llamacpp, 
    top_p_llamacpp, 
    top_k_llamacpp, 
    history_llamacpp, 
    recent_history_llamacpp
    ):

    modelid_llamacpp = download_model(modelid_llamacpp)

    llm = Llama(model_path=modelid_llamacpp,  seed=seed_llamacpp, n_ctx=n_ctx_llamacpp)
    output_llamacpp = llm.create_completion(
        f"{recent_history_llamacpp}", 
        max_tokens=max_tokens_llamacpp, 
        stream=stream_llamacpp, 
        repeat_penalty=repeat_penalty_llamacpp, 
        temperature=temperature_llamacpp, 
        top_p=top_p_llamacpp, 
        top_k=top_k_llamacpp, 
    )    
    
    answer_llamacpp = (output_llamacpp['choices'][0]['text'])
    last_answer_llamacpp = f"{recent_history_llamacpp}{answer_llamacpp}"
    global_answer_llamacpp = f"{history_llamacpp}{answer_llamacpp}"
    write_file(global_answer_llamacpp)

    del llm, output_llamacpp
    clean_ram()
        
    return last_answer_llamacpp, global_answer_llamacpp
