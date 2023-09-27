# https://github.com/Woolverine94/biniou
# img2txt_git.py
import gradio as gr
import os
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, VisionEncoderDecoderModel
import torch
import time
from ressources.common import *

device_img2txt_git = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_img2txt_git = "./models/GIT"
os.makedirs(model_path_img2txt_git, exist_ok=True)

model_list_img2txt_git = [
    "microsoft/git-large-coco",
]

def text_img2txt_git(
    modelid_img2txt_git, 
    max_tokens_img2txt_git, 
    min_tokens_img2txt_git, 
    num_beams_img2txt_git, 
    num_beam_groups_img2txt_git, 
    diversity_penalty_img2txt_git, 
    img_img2txt_git, 
    progress_img2txt_git=gr.Progress(track_tqdm=True)
    ):
    
    processor_img2txt_git = AutoProcessor.from_pretrained(modelid_img2txt_git, cache_dir=model_path_img2txt_git)    
    pipe_img2txt_git = AutoModelForCausalLM.from_pretrained(
        modelid_img2txt_git, 
        cache_dir=model_path_img2txt_git, 
        torch_dtype=torch.float32, 
        use_safetensors=True,
        resume_download=True,
        local_files_only=True if offline_test() else None
        )
    pipe_img2txt_git = pipe_img2txt_git.to(device_img2txt_git)
    inpipe_img2txt_git = processor_img2txt_git(images=img_img2txt_git, return_tensors="pt").to(device_img2txt_git)

    ids_img2txt_git = pipe_img2txt_git.generate(
        pixel_values=inpipe_img2txt_git.pixel_values, 
        max_new_tokens=max_tokens_img2txt_git, 
        min_new_tokens=min_tokens_img2txt_git, 
        early_stopping=False, 
        num_beams=num_beams_img2txt_git, 
        num_beam_groups=num_beam_groups_img2txt_git, 
#        diversity_penalty=diversity_penalty_img2txt_git,
    )
    captions_img2txt_git = processor_img2txt_git.batch_decode(ids_img2txt_git, skip_special_tokens=True)[0]
    write_file(captions_img2txt_git)
    
    del processor_img2txt_git, pipe_img2txt_git, inpipe_img2txt_git, ids_img2txt_git
    clean_ram()
    
    return captions_img2txt_git
