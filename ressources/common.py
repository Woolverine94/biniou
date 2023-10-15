# https://github.com/Woolverine94/biniou
# common.py
import os
from PIL import Image, ExifTags
from io import BytesIO
import gradio as gr
import torch
import base64
import re
import zipfile as zf
import time
import gc
import psutil
import requests as rq
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTION_LIST = [
    "Outputs",
    "Inputs",
    "Both",
]

MODULES_DB = {
# Structure "Nom onglet" : {"Code module"},
    "Choose ...": {"codename": "dummy"}, 
    "Stable Beluga": {"codename": "chatbot"}, 
    "Stable Diffusion" : {"codename": "txt2img_sd"},  
    "Kandinsky" : {"codename": "txt2img_kd"},
    "img2img" : {"codename": "img2img"},
    "pix2pix" : {"codename": "pix2pix"},
    "inpaint" : {"codename": "inpaint"},
    "Real ESRGAN" : {"codename": "resrgan"},
    "GFPGAN" : {"codename": "gfpgan"},
}

MODULES_LIST = list(MODULES_DB.keys())

RESRGAN_SCALES = { 
    "x2": 2,
#    "x3": 3,
    "x4": 4,
    "x8": 8,    
}    

# {'codename': 'txt2img_sd', 'tab': 'tab_image_num', 'tab_item': 'tab_txt2img_sd_num', 'inject': None, 'extract': None, 'outputs': ['gs_out_txt2img_sd', 'sel_out_txt2img_sd'], 'inputs': ['prompt_txt2img_sd', 'negative_prompt_txt2img_sd'], 'return_tabitem': 'tabs_image'}

# txt2img_sd_img2img.click(fn=send_to_module, inputs=[gs_out_txt2img_sd, sel_out_txt2img_sd, tab_image_num, tab_img2img_num], outputs=[img_img2img, tabs, tabs_image])

# send_to_module ['gs_out_txt2img_sd', 'sel_out_txt2img_sd', 'tab_image_num', 'tab_img2img_num'] ['prompt_img2img', 'negative_prompt_img2img', 'img_img2img', ', tabs', 'tab_img2img_num']

def send_input():
    return True
    
def zipper(content):
    timestamp = time.time()
    savename = f"./.tmp/{timestamp}.zip"
    with zf.ZipFile(savename, 'w') as myzip:
        for idx, file in enumerate(content):
            file_name=file["name"].replace("\\", "/")
            myzip.write(file["name"], f"{idx}_"+ file_name.split("/")[-1])
    return savename

def correct_size(width, height, max_size) :
    if (width>height) :
        dim_max = width
        dim_min = height
        orientation = "l"
    else :
        dim_max = height
        dim_min = width
        orientation = "p"
    approx_int = round(dim_min/(dim_max/max_size))
    approx = (approx_int + 7) & (-8)
    if  (orientation == "l") :
        final_width = max_size
        final_height = approx
    elif (orientation == "p") :
        final_width = approx
        final_height = max_size
    return (final_width, final_height)
    
def round_size(image) :
    width = (image.size[0] + 7) & (-8)
    height = (image.size[1] + 7) & (-8)
    return width, height

def image_upload_event(im):
    if (im != None):
        if (im[0:11] != "data:image/"):
            image_out = Image.open(im)
        else :
            imbis = re.sub('^data:image/.+;base64,', '', im)
            image_out = Image.open(BytesIO(base64.b64decode(imbis)))
        return (image_out.size)
    else : 
        return (512, 512)

def image_upload_event_inpaint(im):
    type_image = type(im)
    rotation_img = 360            
    if (type_image == str):
        image_out = Image.open(im)
    else :
        imbis = re.sub('^data:image/.+;base64,', '', im["image"])
        image_out = Image.open(BytesIO(base64.b64decode(imbis)))
    try :
        exif_data = image_out._getexif()
        orientation_img = exif_data[274]
    except Exception as e:
        orientation_img = ""
    if (orientation_img == 3):
        image_out = image_out.rotate(180, expand=True)
        rotation_img = 180
    elif (orientation_img == 6):
        image_out = image_out.rotate(270, expand=True)
        rotation_img = 270 
    elif (orientation_img == 8):
        image_out = image_out.rotate(90, expand=True)
        rotation_img = 90
    dim = correct_size(image_out.size[0], image_out.size[1], 512)
    image_out = image_out.convert("RGB").resize(dim)
    return (image_out.size[0], image_out.size[1], image_out, rotation_img)
    
def image_upload_event_inpaint_b(im):
    if (im != None):
       imbis = re.sub('^data:image/.+;base64,', '', im)
       image_out = Image.open(BytesIO(base64.b64decode(imbis)))
       dim = correct_size(image_out.size[0], image_out.size[1], 512)
       return (dim)
    else : 
        return (512, 512)

def scale_image(im, size):
    max_size = int(size)
    if (im != None):
        type_image = type(im)
        if (type_image == str) :
            image_out = Image.open(im)
        else :
            imbis = re.sub('^data:image/.+;base64,', '', im["image"])
            image_out = Image.open(BytesIO(base64.b64decode(imbis)))
        if image_out.size[0] > max_size or image_out.size[1] > max_size :
            dim = correct_size(image_out.size[0], image_out.size[1], max_size)            
            image_out = image_out.convert("RGB").resize(dim)
        return (image_out.size[0], image_out.size[1], image_out)
    return (max_size, max_size, "")
        
def scale_resrgan_change(scale_resrgan):
    if (RESRGAN_SCALES[scale_resrgan] < 3):
        scale_model_resrgan: str = "RealESRGAN_x2.pth"
    elif (RESRGAN_SCALES[scale_resrgan] < 5):     
        scale_model_resrgan: str  = "RealESRGAN_x4.pth"
    else :     
        scale_model_resrgan: str = "RealESRGAN_x8.pth"  
    return scale_model_resrgan
        

def preview_image(step, timestep, latents, pipe):
    final_preview=[]
#    print(step, timestep, latents[0][0][0][0])
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # convert to PIL Images
        image = pipe.numpy_to_pil(image)
#        for img in enumerate(image):
        for j in range(len(image)):
            timestamp = time.time()
            savename = f"/tmp/gradio/{timestamp}.png"
            image[j].save(savename)
            final_preview.append(savename)
    return final_preview

        # do something with the Images
        
#            output_gallery.update(value=img[i])
#             img.save(f"step_{step}_img{i}.png")

def clean_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return

def ram_size():
    return int(psutil.virtual_memory().total/(1000**3))

# 0 = Pas de filtre (safety_checker=None)/ 1 = filtre actif
# Safety_Check : 
safety_checker_model = "CompVis/stable-diffusion-safety-checker"

def safety_checker_sd(model_path, device, nsfw_filter):
    if nsfw_filter == "0" :
        safecheck = None
        feat_ex = None
    elif nsfw_filter == "1" :
        safecheck = StableDiffusionSafetyChecker.from_pretrained(
            safety_checker_model, 
            cache_dir=model_path, 
            torch_dtype=torch.float32,
            resume_download=True,
            local_files_only=True if offline_test() else None
        ).to(device)
        feat_ex = AutoFeatureExtractor.from_pretrained(
            safety_checker_model, 
            cache_dir=model_path, 
            torch_dtype=torch.float32,
            resume_download=True,
            local_files_only=True if offline_test() else None            
            )
    return safecheck, feat_ex

def offline_test():
    try:
        rq.get("https://www.google.com", timeout=5)
        return False
    except rq.ConnectionError:
        return True

def write_file(*args) :
    timestamp = time.time()
    savename = f"outputs/{timestamp}.txt"
    content = ""
    for idx, data in enumerate(args):
        content += f"{data} \n"
    with open(savename, 'w') as savefile:
        savefile.write(content)
    return

def set_timestep_vid_ze(numstep) :
    factor = round(numstep/10)
    t1 = numstep-(factor+1)
    t0 = t1-factor
    return t0, t1

def set_num_beam_groups_img2txt_git(numbeam, numbeam_groups) :
    if numbeam>1 and numbeam_groups<2 :
        numbeam_groups = 2
    elif numbeam<2 and numbeam_groups>1 :
        numbeam_groups = 1
    return numbeam_groups
    
def write_ini(module, *args) :
    savename = f".ini/{module}.cfg"
    content = ""
    for idx, data in enumerate(args):
        content += f"{data} \n"
        content = content.replace("False", "0")
        content = content.replace("True", "1")
    with open(savename, 'w') as savefile:
        savefile.write(content)
    return
   
def read_ini(module) :
    filename = f".ini/{module}.cfg"
    content = []
    with open(filename, "r") as fichier :
        lignes = fichier.readlines()
        for ligne in lignes : 
            ligne = ligne.strip(' \n')
            content.append(ligne)
        return content

def test_cfg_exist(module) :
    if os.path.isfile(f".ini/{module}.cfg") :
        return True
    else :
        return False   
