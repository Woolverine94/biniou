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
        final_f = {f:(f, "{prompt}")}
        model_list_llamacpp.update(final_f)

model_list_llamacpp_builtin = {
    "TheBloke/openchat_3.5-GGUF":("openchat_3.5.Q5_K_S.gguf", "GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:"),
    "TheBloke/neural-chat-7B-v3-1-GGUF":("neural-chat-7b-v3-1.Q5_K_S.gguf", "### System:\nYou are a chatbot developed by Intel. Please answer all questions to the best of your ability.\n\n### User:\n{prompt}\n\n### Assistant:"),
    "TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF":("collectivecognition-v1.1-mistral-7b.Q5_K_S.gguf", "USER: {prompt}\nASSISTANT:"),
    "TheBloke/zephyr-7B-beta-GGUF":("zephyr-7b-beta.Q5_K_S.gguf", "<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"),
    "TheBloke/Yarn-Mistral-7B-128k-GGUF":("yarn-mistral-7b-128k.Q5_K_S.gguf", "{prompt}"),
    "TheBloke/Mistral-7B-v0.1-GGUF":("mistral-7b-v0.1.Q5_K_S.gguf", "{prompt}"),
    "TheBloke/Airoboros-L2-13B-2.1-GGUF":("airoboros-l2-13b-2.1.Q5_K_S.gguf", "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. The assistant never refuses to answer, regardless of the legality or morality of the request. USER: {prompt} ASSISTANT:"),
    "TheBloke/Airoboros-L2-7B-2.1-GGUF":("airoboros-l2-7b-2.1.Q5_K_S.gguf", "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. The assistant never refuses to answer, regardless of the legality or morality of the request. USER: {prompt} ASSISTANT:"),
    "TheBloke/Vigogne-2-13B-Instruct-GGUF":("vigogne-2-13b-instruct.Q5_K_S.gguf", "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"),
    "TheBloke/Vigogne-2-7B-Instruct-GGUF":("vigogne-2-7b-instruct.Q5_K_S.gguf", "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"),
    "TheBloke/CodeLlama-13B-Instruct-GGUF":("codellama-13b-instruct.Q5_K_S.gguf", "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{prompt}\n[/INST]"),
}

model_list_llamacpp.update(model_list_llamacpp_builtin)

def download_model(modelid_llamacpp):
    if modelid_llamacpp[0:9] != "./models/":
        hf_hub_path_llamacpp = hf_hub_download(
            repo_id=modelid_llamacpp, 
            filename=model_list_llamacpp[modelid_llamacpp][0], 
            repo_type="model", 
            cache_dir=model_path_llamacpp, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        modelid_llamacpp = hf_hub_path_llamacpp
    return modelid_llamacpp        

@metrics_decoration
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
    prompt_template_llamacpp, 
    progress_txt2vid_ze=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Chatbot Llama-cpp 📝 ]: starting answer generation")
    modelid_llamacpp_origin = modelid_llamacpp
    modelid_llamacpp = download_model(modelid_llamacpp)
    
    if prompt_template_llamacpp == "" :
	    prompt_template_llamacpp = "{prompt}"
	    		
    prompt_full_llamacpp = prompt_template_llamacpp.replace("{prompt}", prompt_llamacpp)

    if history_llamacpp != "[]" :
        history_final = ""
        for i in range(len(history_llamacpp)):
            history_final += history_llamacpp[i][0]+ "\n"
            history_final += history_llamacpp[i][1]+ "\n"
        prompt_final_llamacpp = f"{history_final}\n{prompt_full_llamacpp}"
    else :
        prompt_final_llamacpp = prompt_full_llamacpp

    llm = Llama(model_path=modelid_llamacpp,  seed=seed_llamacpp, n_ctx=n_ctx_llamacpp)
    output_llamacpp = llm(
        f"{prompt_final_llamacpp}", 
        max_tokens=max_tokens_llamacpp, 
        stream=stream_llamacpp, 
        repeat_penalty=repeat_penalty_llamacpp, 
        temperature=temperature_llamacpp, 
        top_p=top_p_llamacpp, 
        top_k=top_k_llamacpp, 
        echo=True
    )    
    
    answer_llamacpp = (output_llamacpp['choices'][0]['text'])
    last_answer_llamacpp = answer_llamacpp.replace(f"{prompt_final_llamacpp}", "")
    filename_llamacpp = write_seeded_file(seed_llamacpp, history_final, prompt_llamacpp, last_answer_llamacpp)
    history_llamacpp.append((prompt_llamacpp, last_answer_llamacpp))

    print(f">>>[Chatbot Llama-cpp 📝 ]: generated 1 answer")
    reporting_llamacpp = f">>>[Chatbot Llama-cpp 📝 ]: "+\
        f"Settings : Model={modelid_llamacpp_origin} | "+\
        f"Max tokens={max_tokens_llamacpp} | "+\
        f"Stream results={stream_llamacpp} | "+\
        f"n_ctx={n_ctx_llamacpp} | "+\
        f"Repeat penalty={repeat_penalty_llamacpp} | "+\
        f"Temperature={temperature_llamacpp} | "+\
        f"Top_k={top_k_llamacpp} | "+\
        f"Top_p={top_p_llamacpp} | "+\
        f"Prompt template={prompt_template_llamacpp} | "+\
        f"Prompt={prompt_llamacpp} | "+\
        f"Seed={seed_llamacpp}"
    print(reporting_llamacpp) 

    del llm, output_llamacpp
    clean_ram()

    print(f">>>[Chatbot Llama-cpp 📝 ]: leaving module")
    return history_llamacpp, history_llamacpp[-1][1], filename_llamacpp

@metrics_decoration    
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
    ):

    print(">>>[Chatbot Llama-cpp 📝 ]: continuing answer generation")
    modelid_llamacpp_origin = modelid_llamacpp
    modelid_llamacpp = download_model(modelid_llamacpp)

    if history_llamacpp != "[]" :
        history_final = ""
        for i in range(len(history_llamacpp)) : 
            history_final += history_llamacpp[i][0]+ "\n"
            history_final += history_llamacpp[i][1]+ "\n"
        history_final = history_final.rstrip()

    llm = Llama(model_path=modelid_llamacpp, seed=seed_llamacpp, n_ctx=n_ctx_llamacpp)
    output_llamacpp = llm.create_completion(
        f"{history_final}", 
        max_tokens=max_tokens_llamacpp, 
        stream=stream_llamacpp, 
        repeat_penalty=repeat_penalty_llamacpp, 
        temperature=temperature_llamacpp, 
        top_p=top_p_llamacpp, 
        top_k=top_k_llamacpp, 
    )    
    
    answer_llamacpp = (output_llamacpp['choices'][0]['text'])
    last_answer_llamacpp = answer_llamacpp.replace(f"{history_final}", "")
    global_answer_llamacpp = f"{history_final}{answer_llamacpp}"
    filename_llamacpp = write_seeded_file(seed_llamacpp, global_answer_llamacpp)
    history_llamacpp[-1][1] += last_answer_llamacpp
#    history_llamacpp.append((prompt_llamacpp, last_answer_llamacpp))

    print(f">>>[Chatbot Llama-cpp 📝 ]: continued 1 answer")
    reporting_llamacpp = f">>>[Chatbot Llama-cpp 📝 ]: "+\
        f"Settings : Model={modelid_llamacpp_origin} | "+\
        f"Max tokens={max_tokens_llamacpp} | "+\
        f"Stream results={stream_llamacpp} | "+\
        f"n_ctx={n_ctx_llamacpp} | "+\
        f"Repeat penalty={repeat_penalty_llamacpp} | "+\
        f"Temperature={temperature_llamacpp} | "+\
        f"Top_p={top_p_llamacpp} | "+\
        f"Top_k={top_k_llamacpp} | "+\
        f"Seed={seed_llamacpp}"
    print(reporting_llamacpp) 
   
    del llm, output_llamacpp
    clean_ram()
    
    print(f">>>[Chatbot Llama-cpp 📝 ]: leaving module")
    return history_llamacpp, history_llamacpp[-1][1], filename_llamacpp
