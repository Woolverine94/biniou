# https://github.com/Woolverine94/biniou
# llava.py
import gradio as gr
import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
from ressources.common import *

device_label_llava, model_arch = detect_device()
device_llava = torch.device(device_label_llava)

# Gestion des modèles
model_path_llava = "./models/llava/"
os.makedirs(model_path_llava, exist_ok=True)


model_list_llava = []

for filename in os.listdir(model_path_llava):
    f = os.path.join(model_path_llava, filename)
    if os.path.isfile(f) and (filename.endswith('.gguf')):
        model_list_llava.append(f)

model_list_llava_builtin = [
    "mys/ggml_bakllava-1",
    "mys/ggml_llava-v1.5-7b",
    "mys/ggml_llava-v1.5-13b",
]

for k in range(len(model_list_llava_builtin)):
    model_list_llava.append(model_list_llava_builtin[k])

def download_model(modelid_llava):
    if modelid_llava[0:9] != "./models/":
        hf_hub_path_llava = hf_hub_download(
            repo_id=modelid_llava,
            filename="ggml-model-q5_k.gguf",
            repo_type="model",
            cache_dir=model_path_llava,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        modelid_llava = hf_hub_path_llava
    return modelid_llava

def download_mmproj(modelid_mmproj):
    if modelid_mmproj[0:9] != "./models/":
        hf_hub_path_llava = hf_hub_download(
            repo_id=modelid_mmproj, 
            filename="mmproj-model-f16.gguf",
            repo_type="model",
            cache_dir=model_path_llava,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        modelid_mmproj = hf_hub_path_llava
    return modelid_mmproj

@metrics_decoration
def text_llava(
    modelid_llava,
    max_tokens_llava,
    seed_llava,
    stream_llava,
    n_ctx_llava,
    repeat_penalty_llava,
    temperature_llava,
    top_p_llava,
    top_k_llava,
    img_llava,
    prompt_llava,
    history_llava,
    prompt_template_llava,
    progress_txt2vid_ze=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Llava 1.5 👁️ ]: starting answer generation")

    modelid_llava_origin = modelid_llava
    modelid_llava = download_model(modelid_llava_origin)
    modelid_mmproj_llava = download_mmproj(modelid_llava_origin)
    image_url = "https://localhost:7860/file="+ img_llava

    if prompt_template_llava == "" :
	    prompt_template_llava = "{prompt}"

    prompt_full_llava = prompt_template_llava.replace("{prompt}", prompt_llava)

    if history_llava != "[]" :
        history_final = ""
        for i in range(len(history_llava)):
            history_final += history_llava[i][0]+ "\n"
            history_final += history_llava[i][1]+ "\n"
        prompt_final_llava = f"{history_final}\n{prompt_full_llava}"
    else :
        prompt_final_llava = prompt_full_llava

    chat_handler_llava = Llava15ChatHandler(clip_model_path=modelid_mmproj_llava)

    llm = Llama(
              model_path=modelid_llava,
              seed=seed_llava,
              n_ctx=n_ctx_llava,
              chat_handler=chat_handler_llava,
              logits_all=True,
    )

    messages_llava = [
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type" : "text", "text": prompt_final_llava}
            ]
        }
    ]

    output_llava = chat_handler_llava(
        llama=llm,
        messages=messages_llava,
        max_tokens=max_tokens_llava,
        repeat_penalty=repeat_penalty_llava,
        temperature=temperature_llava,
        top_p=top_p_llava,
        top_k=top_k_llava,
        echo=True
    )

    answer_llava = (output_llava["choices"][0]["message"]["content"])
    last_answer_llava = answer_llava.replace(f"{prompt_final_llava}", "")
    filename_llava = write_seeded_file(seed_llava, history_final, prompt_llava, last_answer_llava)
    history_llava.append((prompt_llava, last_answer_llava))

    print(f">>>[Llava 1.5 👁️ ]: generated 1 answer")
    reporting_llava = f">>>[Llava 1.5 👁️ ]: "+\
        f"Settings : Model={modelid_llava_origin} | "+\
        f"Max tokens={max_tokens_llava} | "+\
        f"Stream results={stream_llava} | "+\
        f"n_ctx={n_ctx_llava} | "+\
        f"Repeat penalty={repeat_penalty_llava} | "+\
        f"Temperature={temperature_llava} | "+\
        f"Top_k={top_k_llava} | "+\
        f"Top_p={top_p_llava} | "+\
        f"Prompt template={prompt_template_llava} | "+\
        f"Prompt={prompt_llava} | "+\
        f"Seed={seed_llava}"
    print(reporting_llava) 

    del chat_handler_llava, llm, output_llava
    clean_ram()

    print(f">>>[Llava 1.5 👁️ ]: leaving module")
    return history_llava, history_llava[-1][1], filename_llava

@metrics_decoration    
def text_llava_continue(
    modelid_llava, 
    max_tokens_llava, 
    seed_llava, 
    stream_llava, 
    n_ctx_llava, 
    repeat_penalty_llava, 
    temperature_llava, 
    top_p_llava, 
    top_k_llava,
    img_llava,
    history_llava, 
    ):

    print(">>>[Llava 1.5 👁️ ]: continuing answer generation")
    modelid_llava_origin = modelid_llava
    modelid_llava = download_model(modelid_llava)

    if history_llava != "[]" :
        history_final = ""
        for i in range(len(history_llava)) : 
            history_final += history_llava[i][0]+ "\n"
            history_final += history_llava[i][1]+ "\n"
        history_final = history_final.rstrip()

    llm = Llama(model_path=modelid_llava, seed=seed_llava, n_ctx=n_ctx_llava)
    output_llava = llm.create_completion(
        f"{history_final}", 
        max_tokens=max_tokens_llava, 
        stream=stream_llava, 
        repeat_penalty=repeat_penalty_llava, 
        temperature=temperature_llava, 
        top_p=top_p_llava, 
        top_k=top_k_llava, 
    )    
    
    answer_llava = (output_llava['choices'][0]['text'])
    last_answer_llava = answer_llava.replace(f"{history_final}", "")
    last_answer_llava = last_answer_llava.replace("<|im_end|>", "")
    last_answer_llava = last_answer_llava.replace("<|im_start|>user", "")
    last_answer_llava = last_answer_llava.replace("<|im_start|>assistant", "")
    global_answer_llava = f"{history_final}{answer_llava}"
    filename_llava = write_seeded_file(seed_llava, global_answer_llava)
    history_llava[-1][1] += last_answer_llava
#    history_llava.append((prompt_llava, last_answer_llava))

    print(f">>>[Llava 1.5 👁️ ]: continued 1 answer")
    reporting_llava = f">>>[Llava 1.5 👁️ ]: "+\
        f"Settings : Model={modelid_llava_origin} | "+\
        f"Max tokens={max_tokens_llava} | "+\
        f"Stream results={stream_llava} | "+\
        f"n_ctx={n_ctx_llava} | "+\
        f"Repeat penalty={repeat_penalty_llava} | "+\
        f"Temperature={temperature_llava} | "+\
        f"Top_p={top_p_llava} | "+\
        f"Top_k={top_k_llava} | "+\
        f"Seed={seed_llava}"
    print(reporting_llava) 
   
    del llm, output_llava
    clean_ram()
    
    print(f">>>[Llava 1.5 👁️ ]: leaving module")
    return history_llava, history_llava[-1][1], filename_llava
