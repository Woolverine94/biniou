# https://github.com/Woolverine94/biniou
# gfpgan.py
import gradio as gr
import os
import cv2
import PIL
import time
import torch
import numpy as np
from gfpgan.utils import GFPGANer
from ressources.common import *
from huggingface_hub import snapshot_download, hf_hub_download

device_gfpgan = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_gfpgan = "./models/gfpgan/"
os.makedirs(model_path_gfpgan, exist_ok=True)

model_list_gfpgan = [
    "leonelhs/gfpgan",
]

variant_list_gfpgan = [
    "GFPGANv1.pth",
    "GFPGANCleanv1-NoCE-C2.pth",    
    "GFPGANv1.2.pth",
    "GFPGANv1.3.pth",
    "GFPGANv1.4.pth",
#    "RestoreFormer.pth",
]

@metrics_decoration
def image_gfpgan(modelid_gfpgan, variantid_gfpgan, img_gfpgan, progress_gfpgan=gr.Progress(track_tqdm=True)):
    path_gfpgan = os.path.join(model_path_gfpgan, variantid_gfpgan)
    device = torch.device(device_gfpgan)
    
    if os.path.exists(path_gfpgan) == False :
        snapshot_path_gfpgan = snapshot_download(
            repo_id=modelid_gfpgan, 
            local_dir=model_path_gfpgan, 
            local_dir_use_symlinks=False,
            resume_download=True,
#            local_files_only=True if offline_test() else None
            )
        path_gfpgan = os.path.join(snapshot_path_gfpgan, variantid_gfpgan)    
    
    model_gfpgan = GFPGANer(
        model_path=path_gfpgan, 
        upscale=1, 
        arch='clean', 
        channel_multiplier=2
    )
    
    image_inter_gfpgan = np.array(Image.open(img_gfpgan).convert('RGB'))
    image_input_gfpgan = cv2.cvtColor(image_inter_gfpgan, cv2.COLOR_RGB2BGR)
    
    _, _, image_gfpgan = model_gfpgan.enhance(
        image_input_gfpgan, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )
    
    final_image = []
    timestamp = time.time()
    savename = f"outputs/{timestamp}.png"
    image_gfpgan = cv2.cvtColor(image_gfpgan, cv2.COLOR_BGR2RGB)
    image_gfpgan_save = Image.fromarray(image_gfpgan)
    image_gfpgan_save.save(savename)
    final_image.append(image_gfpgan_save)

    del model_gfpgan, image_inter_gfpgan, image_input_gfpgan, image_gfpgan
    clean_ram()
    return final_image, final_image

def image_gfpgan_mini(img_gfpgan):
    modelid_gfpgan = model_list_gfpgan[0]
    variantid_gfpgan = variant_list_gfpgan[4]
    path_gfpgan = os.path.join(model_path_gfpgan, variantid_gfpgan)
    device = torch.device(device_gfpgan)
    
    if os.path.exists(path_gfpgan) == False :
        snapshot_path_gfpgan = snapshot_download(
            repo_id=modelid_gfpgan, 
            local_dir=model_path_gfpgan, 
            local_dir_use_symlinks=False,
            resume_download=True,
#            local_files_only=True if offline_test() else None            
        )
        path_gfpgan = os.path.join(snapshot_path_gfpgan, variantid_gfpgan)
        
    model_gfpgan = GFPGANer(
        model_path=path_gfpgan, 
        upscale=1, 
        arch='clean', 
        channel_multiplier=2
    )
    
    if isinstance(img_gfpgan, np.ndarray):
        image_inter_gfpgan = img_gfpgan
    else :        
        image_inter_gfpgan = np.array(img_gfpgan.convert('RGB'))
    
    image_input_gfpgan = cv2.cvtColor(image_inter_gfpgan, cv2.COLOR_RGB2BGR)
    _, _, image_gfpgan = model_gfpgan.enhance(
        image_input_gfpgan, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )
    
    image_gfpgan = cv2.cvtColor(image_gfpgan, cv2.COLOR_BGR2RGB)    
    image_gfpgan_output = Image.fromarray(image_gfpgan)

    del model_gfpgan, image_inter_gfpgan, image_input_gfpgan, image_gfpgan
    clean_ram()
        
    return image_gfpgan_output
