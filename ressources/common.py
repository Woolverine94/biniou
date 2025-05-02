# https://github.com/Woolverine94/biniou
# common.py
import os
import sys
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
import gradio as gr
import torch
import base64
import re
import zipfile as zf
import time
import gc
from math import ceil
import psutil
import requests as rq
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import numpy as np
from ressources.scheduler import *
import exiv2
import ffmpeg
import music_tag
import shutil

device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_lora_sd = "./models/lora/SD"
model_path_lora_sdxl = "./models/lora/SDXL"
model_path_lora_sd3 = "./models/lora/SD3"
model_path_lora_sd35 = "./models/lora/SD35"
model_path_lora_flux = "./models/lora/flux"
model_path_txtinv_sd = "./models/TextualInversion/SD"
model_path_txtinv_sdxl = "./models/TextualInversion/SDXL"
os.makedirs(model_path_lora_sd, exist_ok=True)
os.makedirs(model_path_lora_sdxl, exist_ok=True)
os.makedirs(model_path_lora_sd3, exist_ok=True)
os.makedirs(model_path_lora_sd35, exist_ok=True)
os.makedirs(model_path_lora_flux, exist_ok=True)
os.makedirs(model_path_txtinv_sd, exist_ok=True)
os.makedirs(model_path_txtinv_sdxl, exist_ok=True)

ACTION_LIST = [
    "Outputs",
    "Inputs",
    "Both",
]

IMAGES_FORMAT_LIST = [
    "png",
    "jpg",
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
    savename = f"./.tmp/{timestamper()}.zip"
    with zf.ZipFile(savename, 'w') as myzip:
        for idx, file in enumerate(content):
            file_name=file["name"].replace("\\", "/")
            myzip.write(file["name"], f"{idx}_"+ file_name.split("/")[-1])
    return savename

def zipper_file(content):
    savename = f"./.tmp/{timestamper()}.zip"
    with zf.ZipFile(savename, 'w') as myzip:
        file_name=content.replace("\\", "/")
        myzip.write(content, file_name.split("/")[-1])
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
    image_out = image_out.convert("RGB").resize(dim, resample=Image.LANCZOS)
    return (image_out.size[0], image_out.size[1], image_out, rotation_img)
    
def image_upload_event_inpaint_b(im):
    if (im != None):
       imbis = re.sub('^data:image/.+;base64,', '', im)
       image_out = Image.open(BytesIO(base64.b64decode(imbis)))
       dim = correct_size(image_out.size[0], image_out.size[1], 512)
       return (dim)
    else : 
        return (512, 512)

def image_upload_event_inpaint_c(im, model):
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
    if ("XL" in model.upper()):
        dim = correct_size(image_out.size[0], image_out.size[1], 1024)
    else:
        dim = correct_size(image_out.size[0], image_out.size[1], 512)
    image_out = image_out.convert("RGB").resize(dim, resample=Image.LANCZOS)
    return (image_out.size[0], image_out.size[1], image_out, rotation_img)


def scale_image(im, size):
    max_size = int(size)
    if (im != None):
        type_image = type(im)
        if (type_image == str):
            image_out = Image.open(im)
        else:
            imbis = re.sub('^data:image/.+;base64,', '', im["image"])
            image_out = Image.open(BytesIO(base64.b64decode(imbis)))
        if image_out.size[0] > max_size or image_out.size[1] > max_size :
            dim = correct_size(image_out.size[0], image_out.size[1], max_size)
            image_out = image_out.convert("RGB").resize(dim, resample=Image.LANCZOS)
        return (image_out.size[0], image_out.size[1], image_out)
    return (max_size, max_size, "")

def scale_image_any(im, size):
    max_size = int(size)
    if (im != None):
        type_image = type(im)
        if (type_image == str):
            image_out = Image.open(im)
        else:
            imbis = re.sub('^data:image/.+;base64,', '', im["image"])
            image_out = Image.open(BytesIO(base64.b64decode(imbis)))
        dim = correct_size(image_out.size[0], image_out.size[1], max_size)
        image_out = image_out.convert("RGB").resize(dim, resample=Image.LANCZOS)
        return (image_out)
    return ""

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
            savename = f"/tmp/gradio/{timestamper()}.png"
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
        device, model_arch = detect_device()
        safecheck = StableDiffusionSafetyChecker.from_pretrained(
            safety_checker_model, 
            cache_dir=model_path, 
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=True if offline_test() else None
        ).to(device)
        feat_ex = AutoFeatureExtractor.from_pretrained(
            safety_checker_model, 
            cache_dir=model_path, 
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )
    return safecheck, feat_ex

def safety_checker_sdxl(model_path, image, nsfw_filter):
    if nsfw_filter == "0":
        return image
    elif nsfw_filter == "1":
        device, model_arch = detect_device()
        safecheck = StableDiffusionSafetyChecker.from_pretrained(
            safety_checker_model,
            cache_dir=model_path,
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=True if offline_test() else None
        ).to(device)

        feat_ex = AutoFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=model_path,
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )

        safety_checker_input = feat_ex(image, return_tensors="pt").to(device)
        image_np = np.array(image)

        _, is_nsfw = safecheck(
            images=image_np,
            clip_input=safety_checker_input.pixel_values.to(device),
        )
        if is_nsfw[0]:
            image = Image.new("RGB", (image.width, image.height), (0, 0, 0))
        return image

def offline_test():
    try:
#        rq.get("https://www.google.com", timeout=5)
        test = rq.get("https://huggingface.co/", timeout=5)
        if (str(test) == "<Response [503]>") or (str(test) == "<Response [501]>"):
            print(">>>[biniou 🧠]: Using offline mode")
            return True
        else:
            return False
    except rq.ConnectionError:
        print(">>>[biniou 🧠]: Using offline mode")
        return True

def write_file(*args) :
    savename = f"outputs/{timestamper()}.txt"
    content = ""
    for idx, data in enumerate(args):
        content += f"{data} \n"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return savename

def write_seeded_file(seed, *args) :
    savename = f"outputs/{timestamper()}_{seed}.txt"
    content = ""
    for idx, data in enumerate(args):
        content += f"{data} \n"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return savename

def check_image_fmt():
    if test_cfg_exist("settings"):
        with open(".ini/settings.cfg", "r", encoding="utf-8") as fichier:
            exec(fichier.read())
    if ("biniou_global_img_fmt" in locals() and locals()['biniou_global_img_fmt'] != ""):
        extension = locals()['biniou_global_img_fmt']
    else:
        extension = "png"
    return extension

def check_metadata(value):
    if test_cfg_exist("settings"):
        with open(".ini/settings.cfg", "r", encoding="utf-8") as fichier:
            exec(fichier.read())
    if (value in locals() and locals()[value] != ""):
        exif = locals()[value]
    else:
        exif = True
    return exif

def name_seeded_image(seed):
    savename = f"outputs/{timestamper()}_{seed}.{check_image_fmt()}"
    return savename

def name_image():
    savename = f"outputs/{timestamper()}.{check_image_fmt()}"
    return savename

def name_idx_audio(idx):
    savename = f"outputs/{timestamper()}_{idx}"
    savename_final = savename+ ".wav" 
    return savename, savename_final

def name_seeded_audio(seed):
    savename = f"outputs/{timestamper()}_{seed}.wav"
    return savename

def name_audio():
    savename = f"outputs/{timestamper()}.wav"
    return savename

def name_seeded_video(seed):
    savename = f"outputs/{timestamper()}_{seed}.mp4"
    return savename

def name_seeded_gif(seed):
    savename = f"outputs/{timestamper()}_{seed}.gif"
    return savename

def name_seeded_shape(seed, ext):
    savename = f"outputs/{timestamper()}_{seed}.{ext}"
    return savename

def set_timestep_vid_ze(numstep, model) :
    if "turbo" not in model:
        factor = round(numstep/10)
        t1 = numstep-(factor+1)
        t0 = t1-factor
    else:
        t0 = numstep - 2
        t1 = numstep - 1
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
        data = f"{data}".replace("\n", "\\n")
        content += f"{data} \n"
        content = content.replace("False", "0")
        content = content.replace("True", "1")
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_auth(*args):
    savename = f".ini/auth.cfg"
    content = ""
    for idx, data in enumerate(args):
        content += f"{data} \n"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return 

def write_settings_ini(
    module,
    biniou_global_settings_lang_ui,
    biniou_global_settings_server_name,
    biniou_global_settings_server_port,
    biniou_global_settings_inbrowser,
    biniou_global_settings_auth,
    biniou_global_settings_auth_message,
    biniou_global_settings_share,
    biniou_global_settings_steps_max,
    biniou_global_settings_batch_size_max,
    biniou_global_settings_width_max_img_create,
    biniou_global_settings_height_max_img_create,
    biniou_global_settings_width_max_img_modify,
    biniou_global_settings_height_max_img_modify,
    biniou_global_settings_sd15_width,
    biniou_global_settings_sd15_height,
    biniou_global_settings_sdxl_width,
    biniou_global_settings_sdxl_height,
    biniou_global_settings_gfpgan,
    biniou_global_settings_tkme,
    biniou_global_settings_clipskip,
    biniou_global_settings_ays,
    biniou_global_settings_img_fmt,
    biniou_global_settings_text_metadatas,
    biniou_global_settings_img_exif,
    biniou_global_settings_gif_exif,
    biniou_global_settings_mp4_metadatas,
    biniou_global_settings_audio_metadatas,

):
    savename = f".ini/{module}.cfg"
    content = f"biniou_global_lang_ui = \"{biniou_global_settings_lang_ui}\"\n\
biniou_global_server_name = {biniou_global_settings_server_name}\n\
biniou_global_server_port = {biniou_global_settings_server_port}\n\
biniou_global_inbrowser = {biniou_global_settings_inbrowser}\n\
biniou_global_auth = {biniou_global_settings_auth}\n\
biniou_global_auth_message = \"{biniou_global_settings_auth_message}\"\n\
biniou_global_share = {biniou_global_settings_share}\n\
biniou_global_steps_max = {biniou_global_settings_steps_max}\n\
biniou_global_batch_size_max = {biniou_global_settings_batch_size_max}\n\
biniou_global_width_max_img_create = {biniou_global_settings_width_max_img_create}\n\
biniou_global_height_max_img_create = {biniou_global_settings_height_max_img_create}\n\
biniou_global_width_max_img_modify = {biniou_global_settings_width_max_img_modify}\n\
biniou_global_height_max_img_modify = {biniou_global_settings_height_max_img_modify}\n\
biniou_global_sd15_width = {biniou_global_settings_sd15_width}\n\
biniou_global_sd15_height = {biniou_global_settings_sd15_height}\n\
biniou_global_sdxl_width = {biniou_global_settings_sdxl_width}\n\
biniou_global_sdxl_height = {biniou_global_settings_sdxl_height}\n\
biniou_global_gfpgan = {biniou_global_settings_gfpgan}\n\
biniou_global_tkme = {biniou_global_settings_tkme}\n\
biniou_global_clipskip = {biniou_global_settings_clipskip}\n\
biniou_global_ays = {biniou_global_settings_ays}\n\
biniou_global_img_fmt = \"{biniou_global_settings_img_fmt}\"\n\
biniou_global_text_metadatas = {biniou_global_settings_text_metadatas}\n\
biniou_global_img_exif = {biniou_global_settings_img_exif}\n\
biniou_global_gif_exif = {biniou_global_settings_gif_exif}\n\
biniou_global_mp4_metadatas = {biniou_global_settings_mp4_metadatas}\n\
biniou_global_audio_metadatas = {biniou_global_settings_audio_metadatas}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def read_ini(module) :
    filename = f".ini/{module}.cfg"
    content = []
    with open(filename, "r", encoding="utf-8") as fichier :
        lignes = fichier.readlines()
        for ligne in lignes : 
            ligne = ligne.replace("\\n", "\n")
            ligne = ligne.strip(' \n')
            content.append(ligne)
    return content

def read_auth() :
    filename = f".ini/auth.cfg"
    content = []
    with open(filename, "r", encoding="utf-8") as fichier :
        lignes = fichier.readlines()
        for ligne in lignes : 
            ligne = ligne.strip(' \n')
            content.append(tuple(ligne.split(':')))
    return content

def test_cfg_exist(module) :
    if os.path.isfile(f".ini/{module}.cfg") :
        return True
    else :
        return False

def test_lang_exist(cfg_file) :
    if os.path.isfile(f"lang/{cfg_file}"):
        return True
    else :
        return False

def test_ini_exist(module) :
    if os.path.isfile(f".ini/{module}.ini") :
        return True
    else :
        return False

def del_cfg(module) :
    os.remove(f".ini/{module}.cfg")
    return 

def del_ini(module) :
    os.remove(f".ini/{module}.ini")
    return 

def detect_device():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32 
    else :
        device = "cpu"
        dtype = torch.float32
    return device, dtype

def metrics_decoration(func): 
    def wrap_func(progress=gr.Progress(track_tqdm=True), *args, **kwargs): 
        start_time = round(time.time()) 
        result = func(*args, **kwargs) 
        stop_time = round(time.time())
        timestamp = convert_seconds_to_timestamp((stop_time-start_time))
        print(f">>>[biniou 🧠]: Generation finished in {timestamp.split(',')[0]} ({(stop_time-start_time)} seconds)") 
        return result 
    return wrap_func 

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False 

def read_logs():
    sys.stdout.flush()
    with open("./.logs/output.log", "r", encoding="utf-8") as f:
        return f.read()

def convert_seconds_to_timestamp(seconds):
    reliquat0 = int(seconds)
    reliquat1 = int(seconds/60)
    reliquat2 = int(reliquat1/60)
    msecondes = round(seconds-(reliquat0), 3)
    msecondes_final = str(int(msecondes*1000)).ljust(3, '0')
    secondes = reliquat0-(reliquat1*60)
    minutes = int((reliquat0-((reliquat2*3600)+secondes))/60)
    heures = int((reliquat0-((minutes*60)+secondes))/3600)
    total = f"{str(heures).zfill(2)}:{str(minutes).zfill(2)}:{str(secondes).zfill(2)},{msecondes_final}"
    return total

def check_steps_strength (steps, strength, model, lora_model):
    model = model_cleaner_sd(model)
    lora_model = model_cleaner_lora(lora_model)
    if strength == 0:
        strength = 0.01

    if (model == "stabilityai/sdxl-turbo") or (model == "stabilityai/sd-turbo") or ("IDKIRO/SDXS-512" in model.upper()):
        steps = ceil(1/strength)
#    elif (model == "SG161222/RealVisXL_V4.0_Lightning"):
    elif ("LIGHTNING" in model.upper()):
        steps = ceil(4/strength)
    elif (model == "thibaud/sdxl_dpo_turbo"):
        steps = ceil(4/strength)
    elif (model == "RunDiffusion/Juggernaut-X-Hyper") or (model == "sd-community/sdxl-flash"):
        steps = ceil(6/strength)
    elif (is_flux(model) and lora_model == "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora"):
        steps = ceil(4/strength)
    elif (is_flux(model) and lora_model == "ByteDance/Hyper-SD"):
        steps = ceil(8/strength)
    else:
        if strength < 0.1:
            steps = ceil(1/strength)
        elif (model == "playgroundai/playground-v2.5-1024px-aesthetic"):
            steps = 15
        else:
            steps = 10
    return int(steps)

def which_os():
    return sys.platform

def timestamper():
	return str(time.time()).replace(".", "_")

def img_fmt_list():
    return IMAGES_FORMAT_LIST

def exif_writer_png(exif_datas, filename):
    if check_metadata("biniou_global_img_exif") == True:
        if (check_image_fmt() == "png"):
            datas = PngInfo()
            datas.add_text("UserComment", f"biniou settings: {exif_datas}")
            for j in range(len(filename)):
                with Image.open(filename[j]) as image:
                    image.save(filename[j], pnginfo=datas, encoding="utf-8")
        elif (check_image_fmt() == "jpg"):
            for j in range(len(filename)):
                image = exiv2.ImageFactory.open(filename[j])
                image.readMetadata()
                metadata = image.exifData()
                metadata['Exif.Image.ImageDescription'] = f"biniou settings: {exif_datas}"
                image.writeMetadata()
    else:
        pass
    return

def metadata_writer_txt(metadata, filename):
    if check_metadata("biniou_global_text_metadatas") == True:
        with open(filename, "a", encoding="utf-8") as fichier:
            fichier.write(f"\n\n---\n\nbiniou settings: {metadata}")
    return

def metadata_writer_gif(metadata, filename, fps):
    if check_metadata("biniou_global_gif_exif") == True:
        frametime = int((1000/fps))
        for j in range(len(filename)):
            shutil.move(filename[j], ".tmp/tmp.gif")
            with Image.open(".tmp/tmp.gif") as image:
                image.save(filename[j], save_all=True, duration=frametime, comment=f"biniou settings: {metadata}")
            os.remove(".tmp/tmp.gif")
    return

def metadata_writer_mp4(metadata, filename):
    if check_metadata("biniou_global_mp4_metadatas") == True:
        if type(filename) is str:
            filename_list = []
            filename_list.append(filename)
        else:
            filename_list = filename
        for j in range(len(filename_list)):
            shutil.move(filename_list[j], ".tmp/tmp.mp4")
            ffmpeg.input(".tmp/tmp.mp4").output(filename_list[j], metadata=f"comment=biniou settings: {metadata}", map=0, c="copy").overwrite_output().run()
            os.remove(".tmp/tmp.mp4")
    return

def metadata_writer_wav(metadata, filename):
    if check_metadata("biniou_global_audio_metadatas") == True:
        for i in range(len(filename)):
            f = music_tag.load_file(filename[i])
            f['comment'] = f'biniou settings: {metadata}'
            f.save()
    return

def schedulerer(pipe, scheduler, **kwargs):
    karras = False
    sde = False
    if ('Karras') in scheduler:
        karras = True
    if ('DPM++ 2M SDE' or 'DPM++ 2M SDE Karras') in scheduler:
        sde = True
    if karras and not sde:
        return get_scheduler(pipe=pipe, scheduler=scheduler, use_karras_sigmas=True, **kwargs)
    elif not karras and sde:
        return get_scheduler(pipe=pipe, scheduler=scheduler, algorithm_type="sde-dpmsolver++", **kwargs)
    elif karras and sde:
        return get_scheduler(pipe=pipe, scheduler=scheduler, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", **kwargs)
    elif not karras and not sde:
        return get_scheduler(pipe=pipe, scheduler=scheduler, **kwargs)

def nparse(string):
    return string.replace("\n", "\\n")

## Text modules default settings
def write_ini_llamacpp(
    module,
    model_llamacpp,
    quantization_llamacpp,
    max_tokens_llamacpp,
    seed_llamacpp,
    stream_llamacpp,
    n_ctx_llamacpp,
    repeat_penalty_llamacpp,
    temperature_llamacpp,
    top_p_llamacpp,
    top_k_llamacpp,
    force_prompt_template_llamacpp,
    prompt_template_llamacpp,
    system_template_llamacpp,
):
    savename = f".ini/{module}.ini"
    content = f"model_llamacpp.value = \"{model_llamacpp}\"\n\
quantization_llamacpp.value = \"{quantization_llamacpp}\"\n\
max_tokens_llamacpp.value = {max_tokens_llamacpp}\n\
seed_llamacpp.value = {seed_llamacpp}\n\
stream_llamacpp.value = {stream_llamacpp}\n\
n_ctx_llamacpp.value = {n_ctx_llamacpp}\n\
repeat_penalty_llamacpp.value = {repeat_penalty_llamacpp}\n\
temperature_llamacpp.value = {temperature_llamacpp}\n\
top_p_llamacpp.value = {top_p_llamacpp}\n\
top_k_llamacpp.value = {top_k_llamacpp}\n\
force_prompt_template_llamacpp.value = \"{force_prompt_template_llamacpp}\"\n\
prompt_template_llamacpp.value = \"{nparse(prompt_template_llamacpp)}\"\n\
system_template_llamacpp.value = \"{nparse(system_template_llamacpp)}\""
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_llava(
    module,
    model_llava,
    max_tokens_llava,
    seed_llava,
    stream_llava,
    n_ctx_llava,
    repeat_penalty_llava,
    temperature_llava,
    top_p_llava,
    top_k_llava,
    prompt_template_llava,
    system_template_llava,
):
    savename = f".ini/{module}.ini"
    content = f"model_llava.value = \"{model_llava}\"\n\
max_tokens_llava.value = {max_tokens_llava}\n\
seed_llava.value = {seed_llava}\n\
stream_llava.value = {stream_llava}\n\
n_ctx_llava.value = {n_ctx_llava}\n\
repeat_penalty_llava.value = {repeat_penalty_llava}\n\
temperature_llava.value = {temperature_llava}\n\
top_p_llava.value = {top_p_llava}\n\
top_k_llava.value = {top_k_llava}\n\
prompt_template_llava.value = \"{nparse(prompt_template_llava)}\"\n\
system_template_llava.value = \"{nparse(system_template_llava)}\""
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_img2txt_git(
    module,
    model_img2txt_git,
    min_tokens_img2txt_git,
    max_tokens_img2txt_git,
    num_beams_img2txt_git,
    num_beam_groups_img2txt_git,
    diversity_penalty_img2txt_git,
):
    savename = f".ini/{module}.ini"
    content = f"model_img2txt_git.value = \"{model_img2txt_git}\"\n\
min_tokens_img2txt_git.value = {min_tokens_img2txt_git}\n\
max_tokens_img2txt_git.value = {max_tokens_img2txt_git}\n\
num_beams_img2txt_git.value = {num_beams_img2txt_git}\n\
num_beam_groups_img2txt_git.value = {num_beam_groups_img2txt_git}\n\
diversity_penalty_img2txt_git.value = {diversity_penalty_img2txt_git}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_whisper(
    module,
    model_whisper,
    srt_output_whisper,
):
    savename = f".ini/{module}.ini"
    content = f"model_whisper.value = \"{model_whisper}\"\n\
srt_output_whisper.value = {srt_output_whisper}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_nllb(
    module,
    model_nllb,
    max_tokens_nllb,
):
    savename = f".ini/{module}.ini"
    content = f"model_nllb.value = \"{model_nllb}\"\n\
max_tokens_nllb.value = {max_tokens_nllb}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_txt2prompt(
    module,
    model_txt2prompt,
    max_tokens_txt2prompt,
    repetition_penalty_txt2prompt,
    seed_txt2prompt,
    num_prompt_txt2prompt,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2prompt.value = \"{model_txt2prompt}\"\n\
max_tokens_txt2prompt.value = {max_tokens_txt2prompt}\n\
repetition_penalty_txt2prompt.value = {repetition_penalty_txt2prompt}\n\
seed_txt2prompt.value = {seed_txt2prompt}\n\
num_prompt_txt2prompt.value = {num_prompt_txt2prompt}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

## Image modules default settings

def write_ini_txt2img_sd(
    module,
    model_txt2img_sd,
    num_inference_step_txt2img_sd,
    sampler_txt2img_sd,
    guidance_scale_txt2img_sd,
    num_images_per_prompt_txt2img_sd,
    num_prompt_txt2img_sd,
    width_txt2img_sd,
    height_txt2img_sd,
    seed_txt2img_sd,
    use_gfpgan_txt2img_sd,
    tkme_txt2img_sd,
    clipskip_txt2img_sd,
    use_ays_txt2img_sd,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2img_sd.value = \"{model_txt2img_sd}\"\n\
num_inference_step_txt2img_sd.value = {num_inference_step_txt2img_sd}\n\
sampler_txt2img_sd.value = \"{sampler_txt2img_sd}\"\n\
guidance_scale_txt2img_sd.value = {guidance_scale_txt2img_sd}\n\
num_images_per_prompt_txt2img_sd.value = {num_images_per_prompt_txt2img_sd}\n\
num_prompt_txt2img_sd.value = {num_prompt_txt2img_sd}\n\
width_txt2img_sd.value = {width_txt2img_sd}\n\
height_txt2img_sd.value = {height_txt2img_sd}\n\
seed_txt2img_sd.value = {seed_txt2img_sd}\n\
use_gfpgan_txt2img_sd.value = {use_gfpgan_txt2img_sd}\n\
tkme_txt2img_sd.value = {tkme_txt2img_sd}\n\
clipskip_txt2img_sd.value = {clipskip_txt2img_sd}\n\
use_ays_txt2img_sd.value = {use_ays_txt2img_sd}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_txt2img_kd(
    module,
    model_txt2img_kd,
    num_inference_step_txt2img_kd,
    sampler_txt2img_kd,
    guidance_scale_txt2img_kd,
    num_images_per_prompt_txt2img_kd,
    num_prompt_txt2img_kd,
    width_txt2img_kd,
    height_txt2img_kd,
    seed_txt2img_kd,
    use_gfpgan_txt2img_kd,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2img_kd.value = \"{model_txt2img_kd}\"\n\
num_inference_step_txt2img_kd.value = {num_inference_step_txt2img_kd}\n\
sampler_txt2img_kd.value = \"{sampler_txt2img_kd}\"\n\
guidance_scale_txt2img_kd.value = {guidance_scale_txt2img_kd}\n\
num_images_per_prompt_txt2img_kd.value = {num_images_per_prompt_txt2img_kd}\n\
num_prompt_txt2img_kd.value = {num_prompt_txt2img_kd}\n\
width_txt2img_kd.value = {width_txt2img_kd}\n\
height_txt2img_kd.value = {height_txt2img_kd}\n\
seed_txt2img_kd.value = {seed_txt2img_kd}\n\
use_gfpgan_txt2img_kd.value = {use_gfpgan_txt2img_kd}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_txt2img_lcm(
    module,
    model_txt2img_lcm,
    num_inference_step_txt2img_lcm,
    sampler_txt2img_lcm,
    guidance_scale_txt2img_lcm,
    lcm_origin_steps_txt2img_lcm,
    num_images_per_prompt_txt2img_lcm,
    num_prompt_txt2img_lcm,
    width_txt2img_lcm,
    height_txt2img_lcm,
    seed_txt2img_lcm,
    use_gfpgan_txt2img_lcm,
    tkme_txt2img_lcm,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2img_lcm.value = \"{model_txt2img_lcm}\"\n\
num_inference_step_txt2img_lcm.value = {num_inference_step_txt2img_lcm}\n\
sampler_txt2img_lcm.value = \"{sampler_txt2img_lcm}\"\n\
guidance_scale_txt2img_lcm.value = {guidance_scale_txt2img_lcm}\n\
lcm_origin_steps_txt2img_lcm.value = {lcm_origin_steps_txt2img_lcm}\n\
num_images_per_prompt_txt2img_lcm.value = {num_images_per_prompt_txt2img_lcm}\n\
num_prompt_txt2img_lcm.value = {num_prompt_txt2img_lcm}\n\
width_txt2img_lcm.value = {width_txt2img_lcm}\n\
height_txt2img_lcm.value = {height_txt2img_lcm}\n\
seed_txt2img_lcm.value = {seed_txt2img_lcm}\n\
use_gfpgan_txt2img_lcm.value = {use_gfpgan_txt2img_lcm}\n\
tkme_txt2img_lcm.value = {tkme_txt2img_lcm}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_txt2img_mjm(
    module,
    model_txt2img_mjm,
    num_inference_step_txt2img_mjm,
    sampler_txt2img_mjm,
    guidance_scale_txt2img_mjm,
    num_images_per_prompt_txt2img_mjm,
    num_prompt_txt2img_mjm,
    width_txt2img_mjm,
    height_txt2img_mjm,
    seed_txt2img_mjm,
    use_gfpgan_txt2img_mjm,
    tkme_txt2img_mjm,
    clipskip_txt2img_mjm,
    use_ays_txt2img_mjm,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2img_mjm.value = \"{model_txt2img_mjm}\"\n\
num_inference_step_txt2img_mjm.value = {num_inference_step_txt2img_mjm}\n\
sampler_txt2img_mjm.value = \"{sampler_txt2img_mjm}\"\n\
guidance_scale_txt2img_mjm.value = {guidance_scale_txt2img_mjm}\n\
num_images_per_prompt_txt2img_mjm.value = {num_images_per_prompt_txt2img_mjm}\n\
num_prompt_txt2img_mjm.value = {num_prompt_txt2img_mjm}\n\
width_txt2img_mjm.value = {width_txt2img_mjm}\n\
height_txt2img_mjm.value = {height_txt2img_mjm}\n\
seed_txt2img_mjm.value = {seed_txt2img_mjm}\n\
use_gfpgan_txt2img_mjm.value = {use_gfpgan_txt2img_mjm}\n\
tkme_txt2img_mjm.value = {tkme_txt2img_mjm}\n\
clipskip_txt2img_mjm.value = {clipskip_txt2img_mjm}\n\
use_ays_txt2img_mjm.value = {use_ays_txt2img_mjm}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_txt2img_paa(
    module,
    model_txt2img_paa,
    num_inference_step_txt2img_paa,
    sampler_txt2img_paa,
    guidance_scale_txt2img_paa,
    num_images_per_prompt_txt2img_paa,
    num_prompt_txt2img_paa,
    width_txt2img_paa,
    height_txt2img_paa,
    seed_txt2img_paa,
    use_gfpgan_txt2img_paa,
    tkme_txt2img_paa,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2img_paa.value = \"{model_txt2img_paa}\"\n\
num_inference_step_txt2img_paa.value = {num_inference_step_txt2img_paa}\n\
sampler_txt2img_paa.value = \"{sampler_txt2img_paa}\"\n\
guidance_scale_txt2img_paa.value = {guidance_scale_txt2img_paa}\n\
num_images_per_prompt_txt2img_paa.value = {num_images_per_prompt_txt2img_paa}\n\
num_prompt_txt2img_paa.value = {num_prompt_txt2img_paa}\n\
width_txt2img_paa.value = {width_txt2img_paa}\n\
height_txt2img_paa.value = {height_txt2img_paa}\n\
seed_txt2img_paa.value = {seed_txt2img_paa}\n\
use_gfpgan_txt2img_paa.value = {use_gfpgan_txt2img_paa}\n\
tkme_txt2img_paa.value = {tkme_txt2img_paa}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_img2img(
    module,
    model_img2img,
    num_inference_step_img2img,
    sampler_img2img,
    guidance_scale_img2img,
    num_images_per_prompt_img2img,
    num_prompt_img2img,
    width_img2img,
    height_img2img,
    seed_img2img,
    use_gfpgan_img2img,
    tkme_img2img,
    clipskip_img2img,
    use_ays_img2img,
):
    savename = f".ini/{module}.ini"
    content = f"model_img2img.value = \"{model_img2img}\"\n\
num_inference_step_img2img.value = {num_inference_step_img2img}\n\
sampler_img2img.value = \"{sampler_img2img}\"\n\
guidance_scale_img2img.value = {guidance_scale_img2img}\n\
num_images_per_prompt_img2img.value = {num_images_per_prompt_img2img}\n\
num_prompt_img2img.value = {num_prompt_img2img}\n\
width_img2img.value = {width_img2img}\n\
height_img2img.value = {height_img2img}\n\
seed_img2img.value = {seed_img2img}\n\
use_gfpgan_img2img.value = {use_gfpgan_img2img}\n\
tkme_img2img.value = {tkme_img2img}\n\
clipskip_img2img.value = {clipskip_img2img}\n\
use_ays_img2img.value = {use_ays_img2img}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_img2img_ip(
    module,
    model_img2img_ip,
    num_inference_step_img2img_ip,
    sampler_img2img_ip,
    guidance_scale_img2img_ip,
    num_images_per_prompt_img2img_ip,
    num_prompt_img2img_ip,
    width_img2img_ip,
    height_img2img_ip,
    seed_img2img_ip,
    use_gfpgan_img2img_ip,
    tkme_img2img_ip,
    clipskip_img2img_ip,
    use_ays_img2img_ip,
):
    savename = f".ini/{module}.ini"
    content = f"model_img2img_ip.value = \"{model_img2img_ip}\"\n\
num_inference_step_img2img_ip.value = {num_inference_step_img2img_ip}\n\
sampler_img2img_ip.value = \"{sampler_img2img_ip}\"\n\
guidance_scale_img2img_ip.value = {guidance_scale_img2img_ip}\n\
num_images_per_prompt_img2img_ip.value = {num_images_per_prompt_img2img_ip}\n\
num_prompt_img2img_ip.value = {num_prompt_img2img_ip}\n\
width_img2img_ip.value = {width_img2img_ip}\n\
height_img2img_ip.value = {height_img2img_ip}\n\
seed_img2img_ip.value = {seed_img2img_ip}\n\
use_gfpgan_img2img_ip.value = {use_gfpgan_img2img_ip}\n\
tkme_img2img_ip.value = {tkme_img2img_ip}\n\
clipskip_img2img_ip.value = {clipskip_img2img_ip}\n\
use_ays_img2img_ip.value = {use_ays_img2img_ip}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_img2var(
    module,
    model_img2var,
    num_inference_step_img2var,
    sampler_img2var,
    guidance_scale_img2var,
    num_images_per_prompt_img2var,
    num_prompt_img2var,
    width_img2var,
    height_img2var,
    seed_img2var,
    use_gfpgan_img2var,
    tkme_img2var,
):
    savename = f".ini/{module}.ini"
    content = f"model_img2var.value = \"{model_img2var}\"\n\
num_inference_step_img2var.value = {num_inference_step_img2var}\n\
sampler_img2var.value = \"{sampler_img2var}\"\n\
guidance_scale_img2var.value = {guidance_scale_img2var}\n\
num_images_per_prompt_img2var.value = {num_images_per_prompt_img2var}\n\
num_prompt_img2var.value = {num_prompt_img2var}\n\
width_img2var.value = {width_img2var}\n\
height_img2var.value = {height_img2var}\n\
seed_img2var.value = {seed_img2var}\n\
use_gfpgan_img2var.value = {use_gfpgan_img2var}\n\
tkme_img2var.value = {tkme_img2var}"

    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_pix2pix(
    module,
    model_pix2pix,
    num_inference_step_pix2pix,
    sampler_pix2pix,
    guidance_scale_pix2pix,
    image_guidance_scale_pix2pix,
    num_images_per_prompt_pix2pix,
    num_prompt_pix2pix,
    width_pix2pix,
    height_pix2pix,
    seed_pix2pix,
    use_gfpgan_pix2pix,
    tkme_pix2pix,
):
    savename = f".ini/{module}.ini"
    content = f"model_pix2pix.value = \"{model_pix2pix}\"\n\
num_inference_step_pix2pix.value = {num_inference_step_pix2pix}\n\
sampler_pix2pix.value = \"{sampler_pix2pix}\"\n\
guidance_scale_pix2pix.value = {guidance_scale_pix2pix}\n\
image_guidance_scale_pix2pix.value = {image_guidance_scale_pix2pix}\n\
num_images_per_prompt_pix2pix.value = {num_images_per_prompt_pix2pix}\n\
num_prompt_pix2pix.value = {num_prompt_pix2pix}\n\
width_pix2pix.value = {width_pix2pix}\n\
height_pix2pix.value = {height_pix2pix}\n\
seed_pix2pix.value = {seed_pix2pix}\n\
use_gfpgan_pix2pix.value = {use_gfpgan_pix2pix}\n\
tkme_pix2pix.value = {tkme_pix2pix}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_magicmix(
    module,
    model_magicmix,
    num_inference_step_magicmix,
    sampler_magicmix,
    guidance_scale_magicmix,
    kmin_magicmix,
    kmax_magicmix,
    num_prompt_magicmix,
    seed_magicmix,
    use_gfpgan_magicmix,
    tkme_magicmix,
):
    savename = f".ini/{module}.ini"
    content = f"model_magicmix.value = \"{model_magicmix}\"\n\
num_inference_step_magicmix.value = {num_inference_step_magicmix}\n\
sampler_magicmix.value = \"{sampler_magicmix}\"\n\
guidance_scale_magicmix.value = {guidance_scale_magicmix}\n\
kmin_magicmix.value = {kmin_magicmix}\n\
kmax_magicmix.value = {kmax_magicmix}\n\
num_prompt_magicmix.value = {num_prompt_magicmix}\n\
seed_magicmix.value = {seed_magicmix}\n\
use_gfpgan_magicmix.value = {use_gfpgan_magicmix}\n\
tkme_magicmix.value = {tkme_magicmix}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_inpaint(
    module,
    model_inpaint,
    num_inference_step_inpaint,
    sampler_inpaint,
    guidance_scale_inpaint,
    num_images_per_prompt_inpaint,
    num_prompt_inpaint,
    width_inpaint,
    height_inpaint,
    seed_inpaint,
    use_gfpgan_inpaint,
    tkme_inpaint,
    clipskip_inpaint,
    use_ays_inpaint,
):
    savename = f".ini/{module}.ini"
    content = f"model_inpaint.value = \"{model_inpaint}\"\n\
num_inference_step_inpaint.value = {num_inference_step_inpaint}\n\
sampler_inpaint.value = \"{sampler_inpaint}\"\n\
guidance_scale_inpaint.value = {guidance_scale_inpaint}\n\
num_images_per_prompt_inpaint.value = {num_images_per_prompt_inpaint}\n\
num_prompt_inpaint.value = {num_prompt_inpaint}\n\
width_inpaint.value = {width_inpaint}\n\
height_inpaint.value = {height_inpaint}\n\
seed_inpaint.value = {seed_inpaint}\n\
use_gfpgan_inpaint.value = {use_gfpgan_inpaint}\n\
tkme_inpaint.value = {tkme_inpaint}\n\
clipskip_inpaint.value = {clipskip_inpaint.value}\n\
use_ays_inpaint.value = {use_ays_inpaint.value}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_paintbyex(
    module,
    model_paintbyex,
    num_inference_step_paintbyex,
    sampler_paintbyex,
    guidance_scale_paintbyex,
    num_images_per_prompt_paintbyex,
    num_prompt_paintbyex,
    width_paintbyex,
    height_paintbyex,
    seed_paintbyex,
    use_gfpgan_paintbyex,
    tkme_paintbyex,
):
    savename = f".ini/{module}.ini"
    content = f"model_paintbyex.value = \"{model_paintbyex}\"\n\
num_inference_step_paintbyex.value = {num_inference_step_paintbyex}\n\
sampler_paintbyex.value = \"{sampler_paintbyex}\"\n\
guidance_scale_paintbyex.value = {guidance_scale_paintbyex}\n\
num_images_per_prompt_paintbyex.value = {num_images_per_prompt_paintbyex}\n\
num_prompt_paintbyex.value = {num_prompt_paintbyex}\n\
width_paintbyex.value = {width_paintbyex}\n\
height_paintbyex.value = {height_paintbyex}\n\
seed_paintbyex.value = {seed_paintbyex}\n\
use_gfpgan_paintbyex.value = {use_gfpgan_paintbyex}\n\
tkme_paintbyex.value = {tkme_paintbyex}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_outpaint(
    module,
    model_outpaint,
    num_inference_step_outpaint,
    sampler_outpaint,
    guidance_scale_outpaint,
    num_images_per_prompt_outpaint,
    num_prompt_outpaint,
    width_outpaint,
    height_outpaint,
    seed_outpaint,
    use_gfpgan_outpaint,
    tkme_outpaint,
    clipskip_outpaint,
    use_ays_outpaint,
):
    savename = f".ini/{module}.ini"
    content = f"model_outpaint.value = \"{model_outpaint}\"\n\
num_inference_step_outpaint.value = {num_inference_step_outpaint}\n\
sampler_outpaint.value = \"{sampler_outpaint}\"\n\
guidance_scale_outpaint.value = {guidance_scale_outpaint}\n\
num_images_per_prompt_outpaint.value = {num_images_per_prompt_outpaint}\n\
num_prompt_outpaint.value = {num_prompt_outpaint}\n\
width_outpaint.value = {width_outpaint}\n\
height_outpaint.value = {height_outpaint}\n\
seed_outpaint.value = {seed_outpaint}\n\
use_gfpgan_outpaint.value = {use_gfpgan_outpaint}\n\
tkme_outpaint.value = {tkme_outpaint}\n\
clipskip_outpaint.value = {clipskip_outpaint}\n\
use_ays_outpaint.value = {use_ays_outpaint}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_controlnet(
    module,
    model_controlnet,
    num_inference_step_controlnet,
    sampler_controlnet,
    guidance_scale_controlnet,
    num_images_per_prompt_controlnet,
    num_prompt_controlnet,
    width_controlnet,
    height_controlnet,
    seed_controlnet,
    low_threshold_controlnet,
    high_threshold_controlnet,
    strength_controlnet,
    start_controlnet,
    stop_controlnet,
    use_gfpgan_controlnet,
    tkme_controlnet,
    clipskip_controlnet,
    use_ays_controlnet,
):
    savename = f".ini/{module}.ini"
    content = f"model_controlnet.value = \"{model_controlnet}\"\n\
num_inference_step_controlnet.value = {num_inference_step_controlnet}\n\
sampler_controlnet.value = \"{sampler_controlnet}\"\n\
guidance_scale_controlnet.value = {guidance_scale_controlnet}\n\
num_images_per_prompt_controlnet.value = {num_images_per_prompt_controlnet}\n\
num_prompt_controlnet.value = {num_prompt_controlnet}\n\
width_controlnet.value = {width_controlnet}\n\
height_controlnet.value = {height_controlnet}\n\
seed_controlnet.value = {seed_controlnet}\n\
low_threshold_controlnet.value = {low_threshold_controlnet}\n\
high_threshold_controlnet.value = {high_threshold_controlnet}\n\
strength_controlnet.value = {strength_controlnet}\n\
start_controlnet.value = {start_controlnet}\n\
stop_controlnet.value = {stop_controlnet}\n\
use_gfpgan_controlnet.value = {use_gfpgan_controlnet}\n\
tkme_controlnet.value = {tkme_controlnet}\n\
clipskip_controlnet.value = {clipskip_controlnet}\n\
use_ays_controlnet.value = {use_ays_controlnet}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_faceid_ip(
    module,
    model_faceid_ip,
    num_inference_step_faceid_ip,
    sampler_faceid_ip,
    guidance_scale_faceid_ip,
    num_images_per_prompt_faceid_ip,
    num_prompt_faceid_ip,
    width_faceid_ip,
    height_faceid_ip,
    seed_faceid_ip,
    use_gfpgan_faceid_ip,
    tkme_faceid_ip,
    clipskip_faceid_ip,
):
    savename = f".ini/{module}.ini"
    content = f"model_faceid_ip.value = \"{model_faceid_ip}\"\n\
num_inference_step_faceid_ip.value = {num_inference_step_faceid_ip}\n\
sampler_faceid_ip.value = \"{sampler_faceid_ip}\"\n\
guidance_scale_faceid_ip.value = {guidance_scale_faceid_ip}\n\
num_images_per_prompt_faceid_ip.value = {num_images_per_prompt_faceid_ip}\n\
num_prompt_faceid_ip.value = {num_prompt_faceid_ip}\n\
width_faceid_ip.value = {width_faceid_ip}\n\
height_faceid_ip.value = {height_faceid_ip}\n\
seed_faceid_ip.value = {seed_faceid_ip}\n\
use_gfpgan_faceid_ip.value = {use_gfpgan_faceid_ip}\n\
tkme_faceid_ip.value = {tkme_faceid_ip}\n\
clipskip_faceid_ip.value = {clipskip_faceid_ip}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_faceswap(
    module,
    model_faceswap,
    width_faceswap,
    height_faceswap,
    use_gfpgan_faceswap,
):
    savename = f".ini/{module}.ini"
    content = f"model_faceswap.value = \"{model_faceswap}\"\n\
width_faceswap.value = {width_faceswap}\n\
height_faceswap.value = {height_faceswap}\n\
use_gfpgan_faceswap.value = {use_gfpgan_faceswap}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_resrgan(
    module,
    model_resrgan,
    scale_resrgan,
    width_resrgan,
    height_resrgan,
    use_gfpgan_resrgan,
):
    savename = f".ini/{module}.ini"
    content = f"model_resrgan.value = \"{model_resrgan}\"\n\
scale_resrgan.value = \"{scale_resrgan}\"\n\
width_resrgan.value = {width_resrgan}\n\
height_resrgan.value = {height_resrgan}\n\
use_gfpgan_resrgan.value = {use_gfpgan_resrgan}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_gfpgan(
    module,
    model_gfpgan,
    variant_gfpgan,
    width_gfpgan,
    height_gfpgan,
):
    savename = f".ini/{module}.ini"
    content = f"model_gfpgan.value = \"{model_gfpgan}\"\n\
variant_gfpgan.value = \"{variant_gfpgan}\"\n\
width_gfpgan.value = {width_gfpgan}\n\
height_gfpgan.value = {height_gfpgan}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

## Audio modules default settings

def write_ini_musicgen(
    module,
    model_musicgen,
    duration_musicgen,
    cfg_coef_musicgen,
    num_batch_musicgen,
    use_sampling_musicgen,
    temperature_musicgen,
    top_k_musicgen,
    top_p_musicgen,
):
    savename = f".ini/{module}.ini"
    content = f"model_musicgen.value = \"{model_musicgen}\"\n\
duration_musicgen.value = {duration_musicgen}\n\
cfg_coef_musicgen.value = {cfg_coef_musicgen}\n\
num_batch_musicgen.value = {num_batch_musicgen}\n\
use_sampling_musicgen.value = {use_sampling_musicgen}\n\
temperature_musicgen.value = {temperature_musicgen}\n\
top_k_musicgen.value = {top_k_musicgen}\n\
top_p_musicgen.value = {top_p_musicgen}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_musicgen_mel(
    module,
    model_musicgen_mel,
    duration_musicgen_mel,
    cfg_coef_musicgen_mel,
    num_batch_musicgen_mel,
    use_sampling_musicgen_mel,
    temperature_musicgen_mel,
    top_k_musicgen_mel,
    top_p_musicgen_mel,
):
    savename = f".ini/{module}.ini"
    content = f"model_musicgen_mel.value = \"{model_musicgen_mel}\"\n\
duration_musicgen_mel.value = {duration_musicgen_mel}\n\
cfg_coef_musicgen_mel.value = {cfg_coef_musicgen_mel}\n\
num_batch_musicgen_mel.value = {num_batch_musicgen_mel}\n\
use_sampling_musicgen_mel.value = {use_sampling_musicgen_mel}\n\
temperature_musicgen_mel.value = {temperature_musicgen_mel}\n\
top_k_musicgen_mel.value = {top_k_musicgen_mel}\n\
top_p_musicgen_mel.value = {top_p_musicgen_mel}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_musicldm(
    module,
    model_musicldm,
    num_inference_step_musicldm,
    sampler_musicldm,
    guidance_scale_musicldm,
    audio_length_musicldm,
    seed_musicldm,
    num_audio_per_prompt_musicldm,
    num_prompt_musicldm,
):
    savename = f".ini/{module}.ini"
    content = f"model_musicldm.value = \"{model_musicldm}\"\n\
num_inference_step_musicldm.value = {num_inference_step_musicldm}\n\
sampler_musicldm.value = \"{sampler_musicldm}\"\n\
guidance_scale_musicldm.value = {guidance_scale_musicldm}\n\
audio_length_musicldm.value = {audio_length_musicldm}\n\
seed_musicldm.value = {seed_musicldm}\n\
num_audio_per_prompt_musicldm.value = {num_audio_per_prompt_musicldm}\n\
num_prompt_musicldm.value = {num_prompt_musicldm}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_audiogen(
    module,
    model_audiogen,
    duration_audiogen,
    cfg_coef_audiogen,
    num_batch_audiogen,
    use_sampling_audiogen,
    temperature_audiogen,
    top_k_audiogen,
    top_p_audiogen,
):
    savename = f".ini/{module}.ini"
    content = f"model_audiogen.value = \"{model_audiogen}\"\n\
duration_audiogen.value = {duration_audiogen}\n\
cfg_coef_audiogen.value = {cfg_coef_audiogen}\n\
num_batch_audiogen.value = {num_batch_audiogen}\n\
use_sampling_audiogen.value = {use_sampling_audiogen}\n\
temperature_audiogen.value = {temperature_audiogen}\n\
top_k_audiogen.value = {top_k_audiogen}\n\
top_p_audiogen.value = {top_p_audiogen}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_harmonai(
    module,
    model_harmonai,
    steps_harmonai,
    seed_harmonai,
    length_harmonai,
    batch_size_harmonai,
    batch_repeat_harmonai,
):
    savename = f".ini/{module}.ini"
    content = f"model_harmonai.value = \"{model_harmonai}\"\n\
steps_harmonai.value = {steps_harmonai}\n\
seed_harmonai.value = {seed_harmonai}\n\
length_harmonai.value = {length_harmonai}\n\
batch_size_harmonai.value = {batch_size_harmonai}\n\
batch_repeat_harmonai.value = {batch_repeat_harmonai}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_bark(
    module,
    model_bark,
    voice_preset_bark,
):
    savename = f".ini/{module}.ini"
    content = f"model_bark.value = \"{model_bark}\"\n\
voice_preset_bark.value = \"{voice_preset_bark}\""
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

## Video modules default settings

def write_ini_txt2vid_ms(
    module,
    model_txt2vid_ms,
    num_inference_step_txt2vid_ms,
    sampler_txt2vid_ms,
    guidance_scale_txt2vid_ms,
    num_frames_txt2vid_ms,
    num_fps_txt2vid_ms,
    num_prompt_txt2vid_ms,
    width_txt2vid_ms,
    height_txt2vid_ms,
    seed_txt2vid_ms,
    use_gfpgan_txt2vid_ms,

):
    savename = f".ini/{module}.ini"
    content = f"model_txt2vid_ms.value = \"{model_txt2vid_ms}\"\n\
num_inference_step_txt2vid_ms.value = {num_inference_step_txt2vid_ms}\n\
sampler_txt2vid_ms.value = \"{sampler_txt2vid_ms}\"\n\
guidance_scale_txt2vid_ms.value = {guidance_scale_txt2vid_ms}\n\
num_frames_txt2vid_ms.value = {num_frames_txt2vid_ms}\n\
num_fps_txt2vid_ms.value = {num_fps_txt2vid_ms}\n\
num_prompt_txt2vid_ms.value = {num_prompt_txt2vid_ms}\n\
width_txt2vid_ms.value = {width_txt2vid_ms}\n\
height_txt2vid_ms.value = {height_txt2vid_ms}\n\
seed_txt2vid_ms.value = {seed_txt2vid_ms}\n\
use_gfpgan_txt2vid_ms.value = {use_gfpgan_txt2vid_ms}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_txt2vid_ze(
    module,
    model_txt2vid_ze,
    num_inference_step_txt2vid_ze,
    sampler_txt2vid_ze,
    guidance_scale_txt2vid_ze,
    seed_txt2vid_ze,
    num_frames_txt2vid_ze,
    num_fps_txt2vid_ze,
    num_chunks_txt2vid_ze,
    width_txt2vid_ze,
    height_txt2vid_ze,
    num_videos_per_prompt_txt2vid_ze,
    num_prompt_txt2vid_ze,
    motion_field_strength_x_txt2vid_ze,
    motion_field_strength_y_txt2vid_ze,
    timestep_t0_txt2vid_ze,
    timestep_t1_txt2vid_ze,
    use_gfpgan_txt2vid_ze,
    tkme_txt2vid_ze,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2vid_ze.value = \"{model_txt2vid_ze}\"\n\
num_inference_step_txt2vid_ze.value = {num_inference_step_txt2vid_ze}\n\
sampler_txt2vid_ze.value = \"{sampler_txt2vid_ze}\"\n\
guidance_scale_txt2vid_ze.value = {guidance_scale_txt2vid_ze}\n\
seed_txt2vid_ze.value = {seed_txt2vid_ze}\n\
num_frames_txt2vid_ze.value = {num_frames_txt2vid_ze}\n\
num_fps_txt2vid_ze.value = {num_fps_txt2vid_ze}\n\
num_chunks_txt2vid_ze.value = {num_chunks_txt2vid_ze}\n\
width_txt2vid_ze.value = {width_txt2vid_ze}\n\
height_txt2vid_ze.value = {height_txt2vid_ze}\n\
num_videos_per_prompt_txt2vid_ze.value = {num_videos_per_prompt_txt2vid_ze}\n\
num_prompt_txt2vid_ze.value = {num_prompt_txt2vid_ze}\n\
motion_field_strength_x_txt2vid_ze.value = {motion_field_strength_x_txt2vid_ze}\n\
motion_field_strength_y_txt2vid_ze.value = {motion_field_strength_y_txt2vid_ze}\n\
timestep_t0_txt2vid_ze.value = {timestep_t0_txt2vid_ze}\n\
timestep_t1_txt2vid_ze.value = {timestep_t1_txt2vid_ze}\n\
use_gfpgan_txt2vid_ze.value = {use_gfpgan_txt2vid_ze}\n\
tkme_txt2vid_ze.value = {tkme_txt2vid_ze}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_animatediff_lcm(
    module,
    model_animatediff_lcm,
    model_adapters_animatediff_lcm,
    num_inference_step_animatediff_lcm,
    sampler_animatediff_lcm,
    guidance_scale_animatediff_lcm,
    seed_animatediff_lcm,
    num_frames_animatediff_lcm,
    num_fps_animatediff_lcm,
    width_animatediff_lcm,
    height_animatediff_lcm,
    num_videos_per_prompt_animatediff_lcm,
    num_prompt_animatediff_lcm,
    use_gfpgan_animatediff_lcm,
    tkme_animatediff_lcm,
    clipskip_animatediff_lcm,
):
    savename = f".ini/{module}.ini"
    content = f"model_animatediff_lcm.value = \"{model_animatediff_lcm}\"\n\
model_adapters_animatediff_lcm.value = \"{model_adapters_animatediff_lcm}\"\n\
num_inference_step_animatediff_lcm.value = {num_inference_step_animatediff_lcm}\n\
sampler_animatediff_lcm.value = \"{sampler_animatediff_lcm}\"\n\
guidance_scale_animatediff_lcm.value = {guidance_scale_animatediff_lcm}\n\
seed_animatediff_lcm.value = {seed_animatediff_lcm}\n\
num_frames_animatediff_lcm.value = {num_frames_animatediff_lcm}\n\
num_fps_animatediff_lcm.value = {num_fps_animatediff_lcm}\n\
width_animatediff_lcm.value = {width_animatediff_lcm}\n\
height_animatediff_lcm.value = {height_animatediff_lcm}\n\
num_videos_per_prompt_animatediff_lcm.value = {num_videos_per_prompt_animatediff_lcm}\n\
num_prompt_animatediff_lcm.value = {num_prompt_animatediff_lcm}\n\
use_gfpgan_animatediff_lcm.value = {use_gfpgan_animatediff_lcm}\n\
tkme_animatediff_lcm.value = {tkme_animatediff_lcm}\n\
clipskip_animatediff_lcm.value = {clipskip_animatediff_lcm}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_img2vid(
    module,
    model_img2vid,
    num_inference_steps_img2vid,
    sampler_img2vid,
    min_guidance_scale_img2vid,
    max_guidance_scale_img2vid,
    seed_img2vid,
    num_frames_img2vid,
    num_fps_img2vid,
    decode_chunk_size_img2vid,
    width_img2vid,
    height_img2vid,
    num_prompt_img2vid,
    num_videos_per_prompt_img2vid,
    motion_bucket_id_img2vid,
    noise_aug_strength_img2vid,
    use_gfpgan_img2vid,
    tkme_img2vid,
):
    savename = f".ini/{module}.ini"
    content = f"model_img2vid.value = \"{model_img2vid}\"\n\
num_inference_steps_img2vid.value = {num_inference_steps_img2vid}\n\
sampler_img2vid.value = \"{sampler_img2vid}\"\n\
min_guidance_scale_img2vid.value = {min_guidance_scale_img2vid}\n\
max_guidance_scale_img2vid.value = {max_guidance_scale_img2vid}\n\
seed_img2vid.value = {seed_img2vid}\n\
num_frames_img2vid.value = {num_frames_img2vid}\n\
num_fps_img2vid.value = {num_fps_img2vid}\n\
decode_chunk_size_img2vid.value = {decode_chunk_size_img2vid}\n\
width_img2vid.value = {width_img2vid}\n\
height_img2vid.value = {height_img2vid}\n\
num_prompt_img2vid.value = {num_prompt_img2vid}\n\
num_videos_per_prompt_img2vid.value = {num_videos_per_prompt_img2vid}\n\
motion_bucket_id_img2vid.value = {motion_bucket_id_img2vid}\n\
noise_aug_strength_img2vid.value = {noise_aug_strength_img2vid}\n\
use_gfpgan_img2vid.value = {use_gfpgan_img2vid}\n\
tkme_img2vid.value = {tkme_img2vid}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_vid2vid_ze(
    module,
    model_vid2vid_ze,
    num_inference_step_vid2vid_ze,
    sampler_vid2vid_ze,
    guidance_scale_vid2vid_ze,
    image_guidance_scale_vid2vid_ze,
    num_images_per_prompt_vid2vid_ze,
    num_prompt_vid2vid_ze,
    width_vid2vid_ze,
    height_vid2vid_ze,
    seed_vid2vid_ze,
    num_frames_vid2vid_ze,
    num_fps_vid2vid_ze,
    use_gfpgan_vid2vid_ze,
    tkme_vid2vid_ze,
):
    savename = f".ini/{module}.ini"
    content = f"model_vid2vid_ze.value = \"{model_vid2vid_ze}\"\n\
num_inference_step_vid2vid_ze.value = {num_inference_step_vid2vid_ze}\n\
sampler_vid2vid_ze.value = \"{sampler_vid2vid_ze}\"\n\
guidance_scale_vid2vid_ze.value = {guidance_scale_vid2vid_ze}\n\
image_guidance_scale_vid2vid_ze.value = {image_guidance_scale_vid2vid_ze}\n\
num_images_per_prompt_vid2vid_ze.value = {num_images_per_prompt_vid2vid_ze}\n\
num_prompt_vid2vid_ze.value = {num_prompt_vid2vid_ze}\n\
width_vid2vid_ze.value = {width_vid2vid_ze}\n\
height_vid2vid_ze.value = {height_vid2vid_ze}\n\
seed_vid2vid_ze.value = {seed_vid2vid_ze}\n\
num_frames_vid2vid_ze.value = {num_frames_vid2vid_ze}\n\
num_fps_vid2vid_ze.value = {num_fps_vid2vid_ze}\n\
use_gfpgan_vid2vid_ze.value = {use_gfpgan_vid2vid_ze}\n\
tkme_vid2vid_ze.value = {tkme_vid2vid_ze}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

## 3D modules default settings

def write_ini_txt2shape(
    module,
    model_txt2shape,
    num_inference_step_txt2shape,
    sampler_txt2shape,
    guidance_scale_txt2shape,
    num_images_per_prompt_txt2shape,
    num_prompt_txt2shape,
    frame_size_txt2shape,
    seed_txt2shape,
):
    savename = f".ini/{module}.ini"
    content = f"model_txt2shape.value = \"{model_txt2shape}\"\n\
num_inference_step_txt2shape.value = {num_inference_step_txt2shape}\n\
sampler_txt2shape.value = \"{sampler_txt2shape}\"\n\
guidance_scale_txt2shape.value = {guidance_scale_txt2shape}\n\
num_images_per_prompt_txt2shape.value = {num_images_per_prompt_txt2shape}\n\
num_prompt_txt2shape.value = {num_prompt_txt2shape}\n\
frame_size_txt2shape.value = {frame_size_txt2shape}\n\
seed_txt2shape.value = {seed_txt2shape}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def write_ini_img2shape(
    module,
    model_img2shape,
    num_inference_step_img2shape,
    sampler_img2shape,
    guidance_scale_img2shape,
    num_images_per_prompt_img2shape,
    num_prompt_img2shape,
    frame_size_img2shape,
    seed_img2shape,
):
    savename = f".ini/{module}.ini"
    content = f"model_img2shape.value = \"{model_img2shape}\"\n\
num_inference_step_img2shape.value = {num_inference_step_img2shape}\n\
sampler_img2shape.value = \"{sampler_img2shape}\"\n\
guidance_scale_img2shape.value = {guidance_scale_img2shape}\n\
num_images_per_prompt_img2shape.value = {num_images_per_prompt_img2shape}\n\
num_prompt_img2shape.value = {num_prompt_img2shape}\n\
frame_size_img2shape.value = {frame_size_img2shape}\n\
seed_img2shape.value = {seed_img2shape}"
    with open(savename, 'w', encoding="utf-8") as savefile:
        savefile.write(content)
    return

def autodoc(toparse):
    content = ""
    for item in toparse:
        if (item[:2] != "-[") and (item[:2] != "./"):
            content += f"<a href='https://huggingface.co/{item}' target='_blank'>{item}</a>, "
        elif (item[:2] == "-["):
            content += f"<br />{item}: "
        elif (item[:2] == "./"):
            content += f"{item}, "
    return content

def is_fast_lora(model):
    model = model_cleaner_sd(model)
    if (\
(model == "ByteDance/SDXL-Lightning") or \
(model == "GraydientPlatformAPI/lightning-faster-lora") or \
(model == "ByteDance/Hyper-SD") or \
(model == "openskyml/lcm-lora-sdxl-turbo") or \
("H1T/TCD-SD" in model.upper()) or \
(model == "wangfuyun/PCM_Weights") or \
(model == "tianweiy/DMD2") or \
(model == "jasperai/flash-sdxl") or \
(model == "jasperai/flash-sd") or \
(model == "sd-community/sdxl-flash-lora") or \
(model == "mann-e/Mann-E_Turbo") or \
(model == "alimama-creative/slam-lora-sdxl") or \
(model == "jasperai/flash-sd3") or \
(model == "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora") or \
(model == "RED-AIGC/TDD") or \
(model == "alimama-creative/FLUX.1-Turbo-Alpha") or \
(model == "sunhaha123/stable-diffusion-3.5-medium-turbo") or \
(model == "tensorart/stable-diffusion-3.5-medium-turbo") or \
(model == "ostris/fluxdev2schnell-lora")\
):
        is_fast_lora_value = True
    else:
        is_fast_lora_value = False
    return is_fast_lora_value

def is_fp16(model):
    model = model_cleaner_sd(model)
    if (\
(model == "yodayo-ai/clandestine-xl-1.0")\
):
        is_fp16_value = True
    else :
        is_fp16_value = False
    return is_fp16_value

def is_bin(model):
    model = model_cleaner_sd(model)
    if (\
("dataautogpt3/ProteusV0.4" in model) or \
(model == "RunDiffusion/Juggernaut-XL-Lightning") or \
(model == "RunDiffusion/Juggernaut-X-Hyper") or \
(model == "nitrosocke/classic-anim-diffusion") or \
(model == "sd-dreambooth-library/dndcoverart-v1") or \
(model == "stablediffusionapi/dream-diffusion-lightning")\
):
        is_bin_value = True
    else :
        is_bin_value = False
    return is_bin_value

def is_sdxl(model):
    model = model_cleaner_sd(model)
    if (\
("XL" in model.upper()) or \
("LIGHTNING" in model.upper()) or \
("ETRI-VILAB/KOALA-" in model.upper()) or \
("PLAYGROUNDAI/PLAYGROUND-V2" in model.upper()) or \
("SSD-1B" in model.upper()) or \
("SEGMIND-VEGA" in model.upper()) or \
(model == "ehristoforu/Visionix-alpha") or \
(model == "Corcelio/mobius") or \
(model == "mann-e/Mann-E_Dreams") or \
(model == "mann-e/Mann-E_Art") or \
(model == "cutycat2000x/InterDiffusion-4.0") or \
(model == "RunDiffusion/Juggernaut-X-Hyper") or \
(model == "comin/IterComp") or \
(model == "aipicasso/emi-2") or \
(model == "stablediffusionapi/anime-journey-v2") or \
(model == "dataautogpt3/OpenDalleV1.1") or \
(model == "dataautogpt3/ProteusV0.4") or \
(model == "dataautogpt3/ProteusV0.5") or \
(model == "dataautogpt3/Proteus-v0.6") or \
(model == "dataautogpt3/PrometheusV1") or \
(model == "dataautogpt3/ProteusSigma")\
):
        is_sdxl_value = True
    else:
        is_sdxl_value = False
    return is_sdxl_value

def is_sd3(model):
    model = model_cleaner_sd(model)
    if (\
(model == "v2ray/stable-diffusion-3-medium-diffusers") or \
(model == "ptx0/sd3-reality-mix")\
):
        is_sd3_value = True
    else:
        is_sd3_value = False
    return is_sd3_value

def is_sd35(model):
    model = model_cleaner_sd(model)
    if (\
(model == "adamo1139/stable-diffusion-3.5-large-turbo-ungated") or \
(model == "ariG23498/sd-3.5-merged") or \
(model == "aipicasso/emi-3")\
):
        is_sd35_value = True
    else:
        is_sd35_value = False
    return is_sd35_value

def is_sd35m(model):
    model = model_cleaner_sd(model)
    if (\
(model == "adamo1139/stable-diffusion-3.5-medium-ungated") or \
(model == "tensorart/stable-diffusion-3.5-medium-turbo") or \
(model == "SG161222/RealVis_Medium_1.0b")\
):
        is_sd35m_value = True
    else:
        is_sd35m_value = False
    return is_sd35m_value

def is_flux(model):
    model = model_cleaner_sd(model)
    if (\
(model == "Freepik/flux.1-lite-8B") or \
(model == "black-forest-labs/FLUX.1-schnell") or \
(model == "sayakpaul/FLUX.1-merged") or \
(model == "ChuckMcSneed/FLUX.1-dev") or \
(model == "enhanceaiteam/Mystic") or \
(model == "AlekseyCalvin/AuraFlux_merge_diffusers") or \
(model == "ostris/Flex.1-alpha") or \
(model == "shuttleai/shuttle-jaguar") or \
(model == "Shakker-Labs/AWPortrait-FL") or \
(model == "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16") or \
(model == "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers")\
):
        is_flux_value = True
    else:
        is_flux_value = False
    return is_flux_value

def is_noloras(model):
    model = model_cleaner_sd(model)
    if (\
("IDKIRO/SDXS-512" in model.upper()) or \
(model == "stabilityai/sd-turbo") or \
("ETRI-VILAB/KOALA-" in model.upper()) or \
("SSD-1B" in model.upper()) or \
("SEGMIND-VEGA" in model.upper()) or \
("PLAYGROUNDAI/PLAYGROUND-V2" in model.upper())\
):
        is_noloras_value = True
    else:
        is_noloras_value = False
    return is_noloras_value

def model_cleaner_sd(model):
    model_replacement = {
        "-[ 👍 🚀 Fast SD15 ]-": "IDKiro/sdxs-512-0.9",
        "-[ 👍 🇯🇵 Anime SD15 ]-": "gsdf/Counterfeit-V2.5",
        "-[ 👍 SD15 ]-": "SG161222/Realistic_Vision_V3.0_VAE",
        "-[ 👌 🚀 Fast SDXL ]-": "sd-community/sdxl-flash",
        "-[ 👌 🇯🇵 Anime SDXL ]-": "GraydientPlatformAPI/geekpower-cellshade-xl",
        "-[ 👌 🐢 SDXL ]-": "fluently/Fluently-XL-Final",
        "-[ 👏 🐢 SD3 ]-": "v2ray/stable-diffusion-3-medium-diffusers",
        "-[ 👏 🐢 SD3.5 Large ]-": "adamo1139/stable-diffusion-3.5-large-turbo-ungated",
        "-[ 👏 🐢 SD3.5 Medium ]-": "adamo1139/stable-diffusion-3.5-medium-ungated",
        "-[ 🏆 🐢 Flux ]-": "Freepik/flux.1-lite-8B",
    }
    for clean_model_key, clean_model_value in model_replacement.items():
        model = model.replace(clean_model_key, clean_model_value)
    return model

def model_cleaner_llamacpp(model):
    model_replacement = {
        "-[ 👌 Standard ]-": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "-[ 👍 🚀 Tiny ]-":	"bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        "-[ 👏 🐢 High-end ]-":	"bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "-[ 🧑‍💻 Code ]-": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "-[ 🧑‍⚕️ Specialized ]-": "AdityaXPV/Llama-3-8b-Instruct-law-sage-v0.1-GGUF",
        "-[ 📜 Legacy ]-": "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF",
    }
    for clean_model_key, clean_model_value in model_replacement.items():
        model = model.replace(clean_model_key, clean_model_value)
    return model

def model_cleaner_lora(model):
    model_replacement = {
        "-[ 👍 🚀 Fast SD15 LoRA ]-": "ByteDance/Hyper-SD",
        "-[ 👍 🚀 Fast SDXL LoRA ]-": "ByteDance/SDXL-Lightning",
        "-[ 👍 🚀 Fast SD3 LoRA ]-": "ByteDance/Hyper-SD",
        "-[ 👍 🚀 Fast Flux LoRA ]-": "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora",
        "-[ 👍 🚀 Fast SD3.5 Medium LoRA ]-": "sunhaha123/stable-diffusion-3.5-medium-turbo",
        "-[ 👏 🎚️ Sliders SDXL ]-": "ntc-ai/SDXL-LoRA-slider.extremely-detailed",
        "-[ 👏 🎚️ Sliders SD15 ]-": "color_temperature_slider_v1.safetensors",
        "-[ 👍 SD15 LoRA ]-": "Kvikontent/midjourney-v6",
        "-[ 👏 🐢 SD3 LoRA ]-": "adbrasi/jujutsuKaisen-style-sd3",
        "-[ 👏 🐢 SD3.5 Large LoRA ]-": "TDN-M/vietnamese-paint-art",
        "-[ 👏 🐢 SD3.5 Medium LoRA ]-": "linoyts/yarn-art-30-37-max-grad-norm-medium",
        "-[ 🏆 🐢 Flux LoRA ]-": "alvdansen/pola-photo-flux",
        "-[ 👏 🔎 Enhancement SDXL LoRA ]-": "KingNish/Better-SDXL-Lora",
        "-[ 🏆 🐢 🔎 Enhancement Flux LoRA ]-": "prithivMLmods/Ton618-Epic-Realism-Flux-LoRA",
        "-[ 👌 🎨 Style SDXL LoRA ]-": "goofyai/3d_render_style_xl",
        "-[ 🏆 🐢 🎨 Style Flux LoRA ]-": "alvdansen/flux-koda",
        "-[ 👌 🪧 Posters SDXL LoRA ]-": "Norod78/SDXL-Caricaturized-Lora",
        "-[ 👌 📦 Objects SDXL LoRA ]-": "KappaNeuro/1987-action-figure-playset-packaging",
        "-[ 🏆 🐢 📦 Objects Flux LoRA ]-": "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
        "-[ 👌 🇯🇵 Anime SDXL LoRA ]-": "KappaNeuro/studio-ghibli-style",
        "-[ 🏆 🐢 🇯🇵 Anime Flux LoRA ]-": "alvarobartt/ghibli-characters-flux-lora",
    }
    for clean_model_key, clean_model_value in model_replacement.items():
        model = model.replace(clean_model_key, clean_model_value)
    return model

def lora_model_list(model, *args):
    secondary_lora = False
    for idx, data in enumerate(args):
        if bool(data):
            secondary_lora = True
    model = model_cleaner_sd(model)
    if is_sdxl(model):
        model_path_lora = model_path_lora_sdxl
        model_list_lora_builtin_fast = {
            "-[ 👍 🚀 Fast SDXL LoRA ]-":("sdxl_lightning_4step_lora.safetensors", ""),
            "ByteDance/SDXL-Lightning":("sdxl_lightning_4step_lora.safetensors", ""),
            "tianweiy/DMD2":("dmd2_sdxl_4step_lora.safetensors", ""),
            "wangfuyun/PCM_Weights":("sdxl/pcm_sdxl_lcmlike_lora_converted.safetensors", ""),
            "mann-e/Mann-E_Turbo":("manne_turbo.safetensors", ""),
            "sd-community/sdxl-flash-lora":("sdxl-flash-lora.safetensors", "sdxl-flash-lora:0.55"),
            "jasperai/flash-sdxl":("pytorch_lora_weights.safetensors", ""),
            "openskyml/lcm-lora-sdxl-turbo":("lcm-lora-sdxl-turbo.safetensors", ""),
            "ByteDance/Hyper-SD":("Hyper-SDXL-2steps-lora.safetensors", ""),
#            "ByteDance/Hyper-SD":("Hyper-SDXL-1step-lora.safetensors", ""),
            "h1t/TCD-SDXL-LoRA":("pytorch_lora_weights.safetensors", ""),
            "GraydientPlatformAPI/lightning-faster-lora":("PAseer-SDXL-AcceleratorLighting.safetensors", ""),
            "alimama-creative/slam-lora-sdxl":("pytorch_lora_weights.safetensors", ""),
        }
        model_list_lora_builtin = {
            "-[ 👏 🔎 Enhancement SDXL LoRA ]-":("example-03.safetensors", ""),
            "KingNish/Better-SDXL-Lora":("example-03.safetensors", ""),
            "ehristoforu/dalle-3-xl-v2":("dalle-3-xl-lora-v2.safetensors", ""),
            "ostris/face-helper-sdxl-lora":("face_xl_v0_1.safetensors", ""),
            "-[ 👏 🎚️ Sliders SDXL ]-":("extremely detailed.safetensors", "extremely detailed"),
            "ntc-ai/SDXL-LoRA-slider.extremely-detailed":("extremely detailed.safetensors", "extremely detailed"),
            "LyliaEngine/add-detail-xl":("add-detail-xl.safetensors", ""),
            "ostris/photorealistic-slider-sdxl-lora":("sdxl_photorealistic_slider_v1-0.safetensors", "more realistic"),
            "ntc-ai/SDXL-LoRA-slider.2000s-indie-comic-art-style":("2000s indie comic art style.safetensors", "2000s indie comic art style"),
            "ntc-ai/SDXL-LoRA-slider.psychedelic-trip":("psychedelic trip.safetensors", "psychedelic trip"),
            "ntc-ai/SDXL-LoRA-slider.pixar-style":("pixar-style.safetensors", "pixar-style"),
            "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style":("Studio Ghibli style.safetensors", "Studio Ghibli style"),
            "AiWise/sdxl-faetastic-details_v24":("SDXLFaeTastic2400.safetensors", "fantasy, details, sdxl style lora"),
            "-[ 👌 🎨 Style SDXL LoRA ]-":("3d_render_style_xl.safetensors", "3d style, 3d render"),
            "goofyai/3d_render_style_xl":("3d_render_style_xl.safetensors", "3d style, 3d render"),
            "goofyai/disney_style_xl":("disney_style_xl.safetensors", "disney style"),
            "goofyai/cyborg_style_xl":("cyborg_style_xl-off.safetensors", "cyborg style, cyborg, android"),
            "goofyai/Leonardo_Ai_Style_Illustration":("leonardo_illustration.safetensors", "leonardo style, illustration, vector art"),
            "GraydientPlatformAPI/spiderman-sdxl":("Spider-Man.safetensors", "Spider-Man"),
            "nerijs/pixel-art-xl":("pixel-art-xl.safetensors", "pixel, pixel art"),
            "artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl":("PS1Redmond-PS1Game-Playstation1Graphics.safetensors", "Playstation 1 Graphics, PS1 Game"),
            "SvenN/sdxl-emoji":("lora.safetensors", "emoji"),
            "artificialguybr/IconsRedmond-IconsLoraForSDXL":("IconsRedmond.safetensors", "icredm"),
            "Norod78/SDXL-YarnArtStyle-LoRA":("SDXL_Yarn_Art_Style.safetensors", "Yarn art style"),
            "KappaNeuro/director-tim-burton-style":("Director Tim Burton style.safetensors", "Director Tim Burton style - "),
            "thwri/dark-gothic-fantasy-xl":("dark_gothic_fantasy_xl_3.01.safetensors", "dark fantasy, dark gothic fantasy"),
            "TheLastBen/William_Eggleston_Style_SDXL":("wegg.safetensors", "by william eggleston"),
            "KappaNeuro/cute-animals":("Cute Animals.safetensors", "Cute Animals - "),
            "Senetor/Voxel_style-2":("ral-vxl.safetensors", "ral-vxl"),
            "itsVilen/Mspaint":("SDXL_MSPaint_Portrait.safetensors", "MSPaint portrait, MSPaint drawing"),
            "alvdansen/popartanime":("Pop_Art_Illustration-000012.safetensors", "daiton style"),
            "alvdansen/frosting_lane_redux":("araminta_k_frosting_lane_redux.safetensors", ""),
            "IcelosAI/Spacephoto_LORA_XL_Base":("spacephoto-10.safetensors", "spacephoto"),
            "KappaNeuro/moebius-jean-giraud-style":("Moebius (Jean Giraud) Style.safetensors", "Moebius (Jean Giraud) Style - "),
            "prithivMLmods/Pegasi-Beta-GTA-LoRA":("Pegasi-Beta-GTA-LoRA.safetensors", "Gta"),
            "inventwithdean/vangogh-SDXL-LoRA":("vangogh-SDXL-LoRA.safetensors", "<s0><s1>"),
            "Fictiverse/Voxel_XL_Lora":("VoxelXL_v1.safetensors", "voxel style"),
            "-[ 👌 🪧 Posters SDXL LoRA ]-":("SDXL-Caricaturized-Lora.safetensors", "Caricaturized"),
            "Norod78/SDXL-Caricaturized-Lora":("SDXL-Caricaturized-Lora.safetensors", "Caricaturized"),
            "artificialguybr/movie-poster-redmond-for-sd-xl-create-movie-poster-images":("MoviePosterRedmond-MoviePoster-MoviePosterRedAF.safetensors", "Movie Poster, MoviePosterAF"),
            "KappaNeuro/movie-poster":("Movie Poster.safetensors", "Movie Poster - "),
            "openskyml/soviet-diffusion-xl":("Soviet-poster.safetensors", "soviet poster"),
            "HarroweD/HarrlogosXL":("Harrlogos_v2.0.safetensors", "text logo"),
            "KappaNeuro/vintage-postage-stamps":("Vintage Postage Stamps.safetensors", "Vintage Postage Stamps - "),
            "-[ 👌 📦 Objects SDXL LoRA ]-":("1987 Action Figure Playset Packaging.safetensors", "1987 Action Figure Playset Packaging - "),
            "KappaNeuro/1987-action-figure-playset-packaging":("1987 Action Figure Playset Packaging.safetensors", "1987 Action Figure Playset Packaging - "),
            "lordjia/lelo-lego-lora-for-xl-sd1-5":("Lego_XL_v2.1.safetensors", "LEGO Creator"),
            "KappaNeuro/diorama":("Diorama.safetensors", "Diorama - "),
            "DoctorDiffusion/doctor-diffusion-s-claymation-style-lora":("DD-made-of-clay-XL-v2.safetensors", "made-of-clay"),
            "DoctorDiffusion/doctor-diffusion-s-toy-xl-style-lora":("DD-toy-v2.safetensors", "toy"),
            "twn39/blindbox-popmart-xl":("blindbox_popmart_xl.safetensors", "popmart, blindbox"),
            "artificialguybr/StickersRedmond":("StickersRedmond.safetensors", "Stickers, Sticker"),
            "Norod78/SDXL-StickerSheet-Lora":("SDXL-StickerSheet-Lora.safetensors", "StickerSheet"),
            "DoctorDiffusion/doctor-diffusion-s-tarot-card-crafter":("DD-trtcrd-v4.safetensors", "trtcrd artstyle"),
            "DoctorDiffusion/doctor-diffusion-s-xray-xl-lora":("DD-xray-v1.safetensors", "xray"),
            "ostris/ikea-instructions-lora-sdxl":("ikea_instructions_xl_v1_5.safetensors", ""),
            "ostris/super-cereal-sdxl-lora":("cereal_box_sdxl_v1.safetensors", ""),
            "TheLastBen/Papercut_SDXL":("papercut.safetensors", "papercut - "),
            "pookienumnums/DpictClassicalIllustration":("DPICT_Classic_Sketches.safetensors", "dpict, monochrome, classical illustration"),
            "-[ 👌 🇯🇵 Anime SDXL LoRA ]-":("Studio Ghibli Style.safetensors", "Studio Ghibli Style - "),
            "KappaNeuro/studio-ghibli-style":("Studio Ghibli Style.safetensors", "Studio Ghibli Style - "),
            "mnemic/MinionStyleXL-SDXL-LoRA":("MinionStyleXL.safetensors", "MinionStyle"),
            "Norod78/SDXL-JojosoStyle-Lora-v2":("SDXL-JojosoStyle-Lora-v2-r16.safetensors", "JojoSostyle"),
            "alvdansen/BandW-Manga":("BW-000014.safetensors", ""),
            "Norod78/SDXL-simpstyle-Lora-v2":("SDXL-Simpstyle-Lora-v2-r16.safetensors", "simpstyle"),
            "ProomptEngineer/pe-old-school-cartoon-style":("PE_OldCartoonStyle.safetensors", "old school cartoon style"),
            "Pclanglais/Mickey-1928":("pytorch_lora_weights.safetensors", "Mickey|Minnie|Pete"),
            "ivolegrey/Sci-fi_Sketch_Style_SDXL":("Sci-fi_Sketch_Style_SDXL_V1.safetensors", "rough sketch, messy lineart, monochromatic, flat color"),
            "artificialguybr/StudioGhibliRedmond":("StudioGhibliRedmond-StdGBRedmAF.safetensors", "StdGBRedmAF"),
            "artificialguybr/StudioGhibli.Redmond-V2":("StudioGhibli.Redmond-StdGBRRedmAF-StudioGhibli.safetensors", "Studio Ghibli"),
            "alvdansen/phantasma-anime":("araminta_k_phantasma_anime.safetensors", ""),
            "-[ 🏠 Local models ]-":("", ""),
        }
    elif is_sd3(model):
        model_path_lora = model_path_lora_sd3
        model_list_lora_builtin_fast = {
            "-[ 👍 🚀 Fast SD3 LoRA ]-":("Hyper-SD3-4steps-CFG-lora.safetensors", ""),
#            "jasperai/flash-sd3":("adapter_model.safetensors", ""),
            "ByteDance/Hyper-SD":("Hyper-SD3-4steps-CFG-lora.safetensors", ""),
            "":("", ""),
        }
        model_list_lora_builtin = {
            "-[ 👏 🐢 SD3 LoRA ]-":("pytorch_lora_weights.safetensors", "anime in jks style"),
            "adbrasi/jujutsuKaisen-style-sd3":("pytorch_lora_weights.safetensors", "anime in jks style"),
            "linoyts/Yarn_art_SD3_LoRA":("pytorch_lora_weights.safetensors", "yarn art style"),
            "gdvstd/dessin-style-sketch-sd3-lora":("pytorch_lora_weights.safetensors", "sks sketch"),
            "nerijs/pixel-art-medium-128-v0.1":("pixel-art-medium-128-v0.1.safetensors", "pixel art style"),
#            "darknoon/symbols-sd3-lora":("pytorch_lora_weights.safetensors", "in the style of SF"),
            "-[ 🏠 Local models ]-":("", ""),
    }
    elif is_sd35(model):
        model_path_lora = model_path_lora_sd35
        model_list_lora_builtin_fast = {
            "":("", ""),
        }
        model_list_lora_builtin = {
            "-[ 👏 🐢 SD3.5 Large LoRA ]-":("vietnamese-painting-art.safetensors", "viet-art"),
            "TDN-M/vietnamese-paint-art":("vietnamese-painting-art.safetensors", "viet-art"),
            "Shakker-Labs/SD3.5-LoRA-Chinese-Line-Art":("SD35-lora-Chinese-Line-Art.safetensors", "Chinese line art"),
            "alvarobartt/ghibli-characters-sd3.5-lora":("pytorch_lora_weights.safetensors", "Ghibli style"),
            "reverentelusarca/ancient-style-sd35":("sd35-ancient_style_v1.safetensors", "ancientstyle"),
            "Wadaka/NewYorkerComic_Style":("NYS_Lora.safetensors", "nyrkr style"),
            "nerijs/pixel-art-3.5L":("pixel-art-3.5L-v2_000000500.safetensors", "pixel art"),
            "-[ 🏠 Local models ]-":("", ""),
    }

    elif is_sd35m(model):
        model_path_lora = model_path_lora_sd35
        model_list_lora_builtin_fast = {
            "-[ 👍 🚀 Fast SD3.5 Medium LoRA ]-":("sd3.5m_turbo_8steps.safetensors", ""),
            "sunhaha123/stable-diffusion-3.5-medium-turbo":("sd3.5m_turbo_8steps.safetensors", ""),
        }
        model_list_lora_builtin = {
            "-[ 👏 🐢 SD3.5 Medium LoRA ]-":("pytorch_lora_weights.safetensors", "yarn art style"),
            "linoyts/yarn-art-30-37-max-grad-norm-medium":("pytorch_lora_weights.safetensors", "yarn art style"),
#            "JINTAIRE/anime_pose":("anime pose(final).safetensors", "anime pose ,"),
#            "raaedk/subliminal":("pytorch_lora_weights.safetensors", "ps2 graphics, fog filled space, subliminal"),
#            "Clybius/mspaint-style-sd3.5medium":("mspaint-style-sd3-v1.safetensors", "mspaint style"),
            "-[ 🏠 Local models ]-":("", ""),
    }

    elif is_flux(model):
        model_path_lora = model_path_lora_flux
        model_list_lora_builtin_fast = {
            "-[ 👍 🚀 Fast Flux LoRA ]-":("Hyper-Flux.1-Dev 4-step-Lora.safetensors", ""),
            "Lingyuzhou/Hyper_Flux.1_Dev_4_step_Lora":("Hyper-Flux.1-Dev 4-step-Lora.safetensors", ""),
            "ostris/fluxdev2schnell-lora":("dev2schnell.safetensors", ""),
            "ByteDance/Hyper-SD":("Hyper-FLUX.1-dev-8steps-lora.safetensors", ""),
            "RED-AIGC/TDD":("TDD-FLUX.1-dev-lora-beta.safetensors", ""),
            "alimama-creative/FLUX.1-Turbo-Alpha":("diffusion_pytorch_model.safetensors", ""),
        }
        model_list_lora_builtin = {
#            "-[ 🏆 🐢 Flux LoRA ]-":("pola_photo_araminta_k.safetensors", "polaroid style"),
            "-[ 🏆 🐢 🔎 Enhancement Flux LoRA ]-":("Epic-Realism-Unpruned.safetensors", "Epic Realism"),
            "prithivMLmods/Ton618-Epic-Realism-Flux-LoRA":("Epic-Realism-Unpruned.safetensors", "Epic Realism"),
            "XLabs-AI/flux-RealismLora":("lora.safetensors", ""),
            "strangerzonehf/Flux-Super-Realism-LoRA":("super-realism.safetensors", "Super Realism"),
            "Shakker-Labs/FLUX.1-dev-LoRA-add-details":("FLUX-dev-lora-add_details.safetensors", ""),
            "prithivMLmods/Flux-Realism-FineDetailed":("Flux-Realism-FineDetailed.safetensors", "Fine Detailed"),
            "glif-loradex-trainer/insectagon_Architect_render1":("Architect_render1.safetensors", "R3nderd"),
            "strangerzonehf/Realism-v3-Flux":("Realism-v3.safetensors", "Realism v3"),
            "strangerzonehf/cinematicShot-Pics-Flux":("cinematic-shot.safetensors", "cinematic shot"),
            "strangerzonehf/Realism-H6-Flux":("Realism H6.safetensors", "Realism H6"),
            "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur":("FLUX-dev-lora-AntiBlur.safetensors", ""),
            "-[ 🏆 🐢 🎨 Style Flux LoRA ]-":("araminta_k_flux_koda.safetensors", "flmft kodachrome style"),
            "alvdansen/flux-koda":("araminta_k_flux_koda.safetensors", "flmft kodachrome style"),
            "alvdansen/pola-photo-flux":("pola_photo_araminta_k.safetensors", "polaroid style"),
            "dvyio/flux-lora-seventies-photograph":("1d21ceb96d254a2699df6a681c02780c_pytorch_lora_weights.safetensors", "photograph in the style of SVNTS"),
            "Jovie/Midjourney":("mj.safetensors", ""),
            "Jovie/Midjourney_Schnell":("mj5.safetensors", ""),
#            "kudzueye/Boreal":("boreal-flux-dev-lora-v04_1000_steps.safetensors", "photo"),
            "kudzueye/boreal-flux-dev-v2":("boreal-v2.safetensors", "photo"),
            "alvdansen/frosting_lane_flux":("flux_dev_frostinglane_araminta_k.safetensors", "frstingln illustration"),
            "gokaygokay/Flux-Double-Exposure-LoRA":("double_exposure.safetensors", "dblxpsr"),
            "glif-loradex-trainer/shipley_flux_dev_AlicesAdventuresInWonderland_v2":("flux_dev_AlicesAdventuresInWonderland_v2.safetensors", "wonderlandstyle"),
            "markury/surrealidescent":("pytorch_lora_weights.safetensors", "A surreal photo"),
            "FounderFeed/gta-style-lora":("aidmaGTA6-FLUX-V0.1.safetensors", "aidmaGTA5"),
            "FounderFeed/MidjourneyMeetFlux":("aidmaMJ6.1-FLUX-v0.4.safetensors", "aidmaMJ6.1"),
            "strangerzonehf/Flux-Midjourney-Mix2-LoRA":("mjV6.safetensors", "MJ v6"),
            "prithivMLmods/Flux-Dalle-Mix-LoRA":("dalle-mix.safetensors", "dalle-mix"),
            "Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch":("FLUX-dev-lora-children-simple-sketch.safetensors", "sketched style"),
            "renderartist/retrocomicflux":("Retro_Comic_Flux_v2_renderartist.safetensors", "c0m1c, comic book panel"),
            "alvdansen/haunted_linework_flux":("hauntedlinework_flux_araminta_k.safetensors", "hntdlnwrk style"),
            "Grohv/randart2_lora":("randart2-lora.safetensors", "randart2_lora"),
            "prithivMLmods/Flux.1-Dev-Realtime-Toon-Mix":("toon-mix.safetensors", "toon mix"),
            "Shakker-Labs/FLUX.1-dev-LoRA-Vector-Journey":("FLUX-dev-lora-Vector-Journey.safetensors", "artistic style blends reality and illustration elements."),
            "strangerzonehf/Flux-Isometric-3D-Cinematography":("Isometric-3D-Cinematography.safetensors", "Isometric 3D Cinematography"),
            "swsqy/ue5flux":("3d.safetensors", "ue5"),
            "glif/Brain-Melt-Acid-Art":("Brain_Melt.safetensors", "in an acid surrealism style, maximalism"),
            "goofyai/3D_Render_for_Flux":("3D_render_flux.safetensors", "3D render"),
            "dvyio/flux-lora-simple-illustration":("c262b2a947bb43c18218813436f9401f_pytorch_lora_weights.safetensors", "illustration in the style of SMPL"),
            "Keltezaa/Fire_Spirit":("Fire Spirit-000008.safetensors", "bailing_fire"),
            "UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art":("ume_modern_pixelart.safetensors", "umempart"),
            "glif-loradex-trainer/insectagon_Eightball_comics":("Eightball_comics.safetensors", "8Ball"),
            "Norod78/chalk-board-drawing-flux":("Chalk_Board_Drawing_FLUX.safetensors", "ChalkBoardDrawing"),
            "Norod78/humeow-flux":("HuMeow_FLUX.safetensors", "HuMeow"),
            "Datou1111/Yoji_Shinkawa":("Yoji_Shinkawa.safetensors", "Yoji_Shinkawa"),
            "youknownothing/flux_dreamscape":("flux_dreamscape.safetensors", "in the style of BSstyle004"),
            "Datou1111/shou_xin":("shou_xin.safetensors", "shou_xin, pencil sketch"),
            "gokaygokay/Flux-Engrave-LoRA":("engrave.safetensors", "NGRVNG, engrave"),
            "gokaygokay/Flux-Mosaic-LoRA":("mosaic_lora.safetensors", "MZCMZC, mosaic"),
            "leonel4rd/FluxDisney":("Flux_retro_Disney.safetensors", "egdisney"),
            "Weiii722/SouthParkVibe":("South_Park_style_flux.safetensors", "SouthParkVibe"),
            "aixonlab/FLUX.1-dev-LoRA-Cinematic-1940s":("cinematic-1940s.safetensors", "cinematic_1940s"),
            "fofr/flux-80s-cyberpunk":("lora.safetensors", "80s cyberpunk"),
            "veryVANYA/ps1-style-flux":("ps1_style_flux_v1.safetensors", "ps1"),
            "alvdansen/the-point-flux":("thepoint_flux_araminta_k.safetensors", "pnt style"),
            "mgwr/Cine-Aesthetic":("MGWR_Cine.safetensors", "mgwr/cine"),
            "keturn/woodcut-illustrations-Trousset-LoRA":("woodcut-illustration.epoch8.safetensors", "woodcut illustration"),
            "glif-loradex-trainer/AP123_flux_dev_cutaway_style":("flux_dev_cutaway_style.safetensors", "cut4w4y illustration"),
            "glif-loradex-trainer/i12bp8_i12bp8_povshots_v1":("i12bp8_povshots_v1.safetensors", "PVSHTS_PPLSNSM first person POV shot"),
            "glif-loradex-trainer/fabian3000_bosch":("bosch.safetensors", "boschstyle"),
            "ludocomito/flux-lora-caravaggio":("lora.safetensors", "In the style of CARAVAGGIO"),
            "strangerzonehf/Flux-Claude-Art":("claude-art.safetensors", "claude art"),
            "WiroAI/GTA6-style-flux-lora":("gta6_style.safetensors", "gta6style"),
            "strangerzonehf/Flux-Sketch-Ep-LoRA":("ep-sketch.safetensors", "ep sketch"),
            "alvdansen/flux_film_foto":("araminta_k_flux_film_foto.safetensors", "flmft photo style"),
            "Jovie/Comics":("comics.safetensors", ""),
            "den123/squidgame":("squidgame.safetensors", "squidgame"),
            "leonel4rd/Comicfx":("the-farish-side_v20-000035.safetensors", "a difrsd comic"),
            "glif-loradex-trainer/fabian3000_pixelfighters":("pixelfighters.safetensors", "pixelfighters style"),
            "prithivMLmods/Retro-Pixel-Flux-LoRA":("Retro-Pixel.safetensors", "Retro Pixel"),
            "Shakker-Labs/FLUX.1-dev-LoRA-Dark-Fantasy":("FLUX.1-dev-lora-Dark-Fantasy.safetensors", ""),
            "memyself/synthwaveglitch":("SynthWaveGlitchyAnalogVideoSynthesisFluxV1.safetensors", "vidsynthsynth, video synth"),
            "batchku/storai-persian-miniature":("storai-persian-miniature.safetensors", "ali_persian-miniature"),
            "strangerzonehf/Flux-Super-Paint-LoRA":("Super-Paint.safetensors", "Super Paint"),
            "fffiloni/oniric-750":("oniric-750.safetensors", "in the style of TOK"),
            "mujibanget/vector-illustration":("lora-000002.TA_trained.safetensors", "mujibvector, vector"),
            "prithivMLmods/Castor-Red-Dead-Redemption-2-Flux-LoRA":("Castor-Red-Dead-Redemption-2-Flux-LoRA.safetensors", "Red Dead Redemption 2"),
            "fffiloni/cozy-book-800":("cozy-book-800.safetensors", "in the style of TOK"),
            "Jonjew/DPMadeofSoap":("DP_Made_Of_Soap_Bubble_FLUX_madeOfSoapBubble.safetensors", "madeOfSoapBubble"),
            "Jonjew/TransformersStyle":("dvr-transformers-flux.safetensors", "mechanical, robot, cybernetic"),
            "Shakker-Labs/Lumatales-FL":("Lumatales.safetensors", "gushirensheng\(style\)"),
            "nerijs/dark-fantasy-movie-flux":("darkfantasy_movie_v1.safetensors", ""),
            "martintomov/retrofuturism-flux-v2":("retrofuturism_flux_lora_martintomov_v2.safetensors", "retrofuturism"),
            "fffiloni/greyscale-tiny-town":("greyscale-tiny-town.safetensors", "greyscale drawing"),
            "AlekseyCalvin/HSTcolorFlexAlpha":("HSToricColorFlex_6000steps_LoRAforFluxOrFlex_32dim64alpha.safetensors", "HST style autochrome photograph"),
            "fofr/flux-mona-lisa":("lora.safetensors", "MNALSA"),
            "ms2stationthis/dutchbaroqueflux":("dutchbaroqueflux.safetensors", "dutchbaroque"),
            "Anashel/rpg":("RPGv6-beta3.safetensors", "RPG Style"),
            "strangerzonehf/Flux-Cute-3D-Kawaii-LoRA":("Cute-3d-Kawaii.safetensors", "Cute 3d Kawaii"),
            "Hectore/80s_commercial_scan":("80s_commercial_scan_2.safetensors", "80s commercial"),
            "-[ 🏆 🐢 📦 Objects Flux LoRA ]-":("FLUX-dev-lora-Logo-Design.safetensors", "wablogo, logo, Minimalist"),
            "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design":("FLUX-dev-lora-Logo-Design.safetensors", "wablogo, logo, Minimalist"),
            "prithivMLmods/Ton618-Tarot-Cards-Flux-LoRA":("Tarot-card.safetensors", "Tarot card"),
            "Purz/vhs-box":("purz-vhs_box.safetensors", "vhs_box"),
            "multimodalart/flux-tarot-v1":("flux_tarot_v1_lora.safetensors", "in the style of TOK a trtcrd tarot style"),
            "prithivMLmods/EBook-Creative-Cover-Flux-LoRA":("EBook-Cover.safetensors", "EBook Cover"),
            "prithivMLmods/Coloring-Book-Flux-LoRA":("coloring-book.safetensors", "Coloring Book"),
            "Shakker-Labs/FLUX.1-dev-LoRA-Text-Poster":("FLUX-dev-lora-Text-Poster.safetensors", "text poster"),
            "prithivMLmods/Ton618-Only-Stickers-Flux-LoRA":("only-stickers.safetensors", "Only Sticker"),
            "dooart/flux-lora-vintage-tarot":("flux-lora-vintage-tarot.safetensors", "tarot card"),
            "Keltezaa/movie-poster-ce-sdxl-flux":("MoviePoster03-02_CE_FLUX_128AIT.safetensors", "mvpstrCE style"),
            "Norod78/Flux_1_Dev_LoRA_Paper-Cutout-Style":("Flux_1_Dev_LoRA_Paper-Cutout-Style.safetensors", "Paper Cutout Style"),
            "prithivMLmods/Knitted-Character-Flux-LoRA":("Knitted-Character.safetensors", "Knitted Character"),
            "prithivMLmods/Seamless-Pattern-Design-Flux-LoRA":("Seamless-Pattern-Design.safetensors", "Seamless Pattern Design"),
            "prithivMLmods/Flux.1-Dev-Poster-HQ-LoRA":("poster-foss.safetensors", "poster foss"),
            "renderartist/retroadflux":("Retro_Ad_Flux_v1_renderartist.safetensors", "m1dc3ntury, vintage advertisement"),
            "prithivMLmods/Flux-Product-Ad-Backdrop":("Prod-Ad.safetensors", "Product Ad"),
            "mozobozo/icon45":("icon45.safetensors", "style of icon45"),
            "prithivMLmods/Flux.1-Dev-Quote-LoRA":("quoter001.safetensors", "quoter"),
            "gokaygokay/Flux-White-Background-LoRA":("80cfbf52faf541d49c6abfe1ac571112_lora.safetensors", "in the middle, white background"),
            "multimodalart/vintage-ads-flux":("vintage-ads-flux-1350.safetensors", "vintage ad of"),
            "prithivMLmods/Flux-Lego-Ref-LoRA":("Lego.safetensors", "lego --fref --89890"),
            "strangerzonehf/Flux-Icon-Kit-LoRA":("Icon-Kit.safetensors", "Icon Kit"),
            "prithivMLmods/Flux.1-Dev-Stamp-Art-LoRA":("stam9.safetensors", "stam9"),
            "strangerzonehf/Flux-Xmas-3D-LoRA":("Flux-Xmas-3D-LoRA.safetensors", "Xmas 3D"),
            "SouthbayJay/t-shirt-vector-style-flux":("vector_shirt_flux_v1.safetensors", "tshirt design, in the style of 5h1rt"),
            "fofr/flux-handwriting":("lora.safetensors", "HWRIT handwriting"),
            "Norod78/ms-fluentui-style-emoji-flux":("ms-fluentui_emoji_flux_lora_000001750.safetensors", "emoji"),
            "strangerzonehf/Flux-3D-Emojies-LoRA":("Flux-3D-Emojies-Mation.safetensors", "3D Emojies"),
            "glif-loradex-trainer/swapagrawal14_tattoo_artist_v1":("tattoo_artist_v1.safetensors", "tattooz_v1_swap"),
            "glif-loradex-trainer/quitters_flux_dev_gc_boxart":("flux_dev_gc_boxart.safetensors", "in the style of gc_boxart"),
            "glif-loradex-trainer/x_bulbul_x_Playstation_2_Game_Covers":("Playstation_2_Game_Covers.safetensors", "ps2 cover"),
            "WizWhite/Wizards_vintage_romance_novel-FLUX":("Wiz-Vintage_Romance-FLUX.safetensors", "Harlequin Romance Book Cover"),
            "WizWhite/wizard-s-paper-model-universe":("Wiz-Paper_Model_Universe.safetensors", "A paper model"),
            "renderartist/toyboxflux":("Toy_Box_Flux_v2_renderartist.safetensors", "t0yb0x 3d render of a toy design"),
            "EvanZhouDev/open-genmoji":("flux-dev.safetensors", "emoji"),
            "glif-loradex-trainer/shipley_flux_dev_bookFoldArt_v1":("flux_dev_bookFoldArt_v1.safetensors", "bookfoldart"),
            "glif-loradex-trainer/i12bp8_i12bp8_greeksculptures_v1":("i12bp8_greeksculptures_v1.safetensors", "GRKSCLPTRS_PPLSNSM"),
            "strangerzonehf/Flux-Cardboard-Art-LoRA":("cardboard# art.safetensors", "cardboard# art"),
            "glif-loradex-trainer/i12bp8_appelsiensam_vintagesign_v1":("appelsiensam_vintagesign_v1.safetensors", "VNTGSGN_PPLSNSM"),
            "prithivMLmods/Logo-Design-Flux-LoRA":("Logo-design.safetensors", "Logo Design"),
            "saurabhswami/Vibrant-tech-3D":("VibrantTech3D_v1.safetensors", "VibrantTech3D style"),
            "noahyoungs/icon-generator":("pytorch_lora_weights.safetensors", "Minimalist icon"),
            "multimodalart/product-design":("product-design.safetensors", "product designed by prdsgn"),
            "Shakker-Labs/FLUX.1-dev-LoRA-Miniature-World":("FLUX-dev-lora-Miniature-World.safetensors", "a meticulously crafted miniature scene"),
            "strangerzonehf/Real-Claymation":("Real-Claymation.safetensors", "Real Claymation"),
            "glif-loradex-trainer/fab1an_1970sbookcovers":("1970sbookcovers.safetensors", "1970s sci-fi book cover"),
            "elikoy/storyboard":("Storyboard_sketch--FLUX.safetensors", "storyboard sketch of"),
            "fffiloni/wooly-play-doh":("wooly-play-doh.safetensors", "in the style of TOK"),
            "iliketoasters/miniature-people":("mini-people-v2-13.safetensors", "miniature people"),
            "AlekseyCalvin/Deni_Style_Soviet_Agitprop_FluxLoRA_BySilverAgePoets":("Deni_Style_Agitprop_FluxLoRA.safetensors", "Art by Deni. Soviet Revolutionary agitprop illustration"),
            "elikoy/actionherofigurator":("Action_Figurator_F1D-000015.safetensors", "boxed action figure"),
            "AlekseyCalvin/Propaganda_Poster_Schnell_by_doctor_diffusion":("propaganda_schnell_v1.safetensors", "propaganda poster"),
            "strangerzonehf/Flux-Master-Claymation":("Master-Claymation.safetensors", "Master Claymation"),
            "SebastianBodza/flux_lora_retro_linedrawing_style_v1":("lora.safetensors", "Retro TOK illustration style of"),
            "Borcherding/LogoIconEmojiMOE-v0.2-FLUX.1-dev-LoRA":("logoiconemojimoe-v0-2-flux-1-dev-lora.safetensors", "LogoIconEmojiMOE"),
            "strangerzonehf/NFT-Plus-InsideOut-Perspective":("NFT-Plus-InsideOut-Perspective.safetensors", "NFT ++, a glass milk carton with inside a "),
            "h0x91b/clay-vorona":("Clay-Vorona-Flux-Lora-V1-Strong.safetensors", "a clay painting of"),
            "-[ 🏆 🐢 🇯🇵 Anime Flux LoRA ]-":("ghibli-characters-flux-lora.safetensors", "Ghibli style"),
            "alvarobartt/ghibli-characters-flux-lora":("ghibli-characters-flux-lora.safetensors", "Ghibli style"),
            "dataautogpt3/FLUX-AestheticAnime":("Flux_1_Dev_LoRA_AestheticAnime.safetensors", ""),
            "Bootoshi/retroanime":("RetroAnimeFluxV1.safetensors", "retro anime"),
            "dataautogpt3/FLUX-SyntheticAnime":("Flux_1_Dev_LoRA_syntheticanime.safetensors", "1980s anime screengrab, VHS quality, syntheticanime"),
            "prithivMLmods/Canopus-Pixar-3D-Flux-LoRA":("Canopus-Pixar-3D-FluxDev-LoRA.safetensors", "Pixar 3D"),
            "prithivMLmods/Flux-Dev-Real-Anime-LoRA":("Flux-Dev-Real-Anime-LoRA.safetensors", "Real Anime"),
            "Nishitbaria/LoRa-Flux-Anime-Style":("lora.safetensors", "ANMCH"),
            "prithivMLmods/Flux-Long-Toon-LoRA":("Long-Toon.safetensors", "Long toons"),
            "nerijs/animation2k-flux":("animation2k_v1.safetensors", ""),
            "brushpenbob/flux-midjourney-anime":("FLUX_MidJourney_Anime.safetensors", "egmid"),
            "glif/90s-anime-art":("flux_dev_anime.safetensors", ""),
            "strangerzonehf/Flux-Ghibli-Art-LoRA":("Ghibli-Art.safetensors", "Ghibli Art"),
            "leonel4rd/Shogunflux":("UENO (1).safetensors", "A black and white manga illustration of"),
            "Nishitbaria/Anime-style-flux-lora-Large":("lora.safetensors", "anm"),
            "leonel4rd/DBZFLUX":("Akira_Toriyama_Style_FX-000001.safetensors", "toriyama_style"),
            "glif-loradex-trainer/x_bulbul_x_90s_anime":("90s_anime.safetensors", "90s anime"),
            "strangerzonehf/Ctoon-Plus-Plus":("Ctoon++.safetensors", "Ctoon++"),
            "cagrison/cyberpunk-anime":("CPA.safetensors", ""),
            "WiroAI/Batman-Animated-Flux-LoRA":("batman_flux_lora.safetensors", "btmnwiro"),
            "alvdansen/softserve_anime":("flux_dev_softstyle_araminta_k.safetensors", "sftsrv style illustration"),
            "glif-loradex-trainer/R4Z0R1337_rubberhose_inspired":("rubberhose_inspired.safetensors", "rubberhose"),
            "FounderFeed/3dAnime-Style-flux-dev-lora":("1800861.safetensors", "3dstyle style"),
            "openfree/flux-chatgpt-ghibli-lora":("flux-chatgpt-ghibli-lora.safetensors", "ghibli"),
            "fffiloni/cute-comic-800":("cute-comic-800.safetensors", "in the style of TOK"),
            "InstantX/FLUX.1-dev-LoRA-Ghibli":("ghibli_style.safetensors", "ghibli style"),
            "strangerzonehf/Ghibli-Flux-Cartoon-LoRA":("Ghibili-Cartoon-Art.safetensors", "Ghibli Art"),
            "aleksa-codes/flux-ghibsky-illustration":("lora_v2.safetensors", "GHIBSKY style painting"),
            "InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai":("Makoto_Shinkai_style.safetensors", "Makoto Shinkai style"),
            "AIWarper/RubberCore1920sCartoonStyle":("RU883R_AIWARPER_v1.0.safetensors", "RU883R"),
            "AIGCDuckBoss/fluxLora_handDrawAnime":("fluxLora_handdrawAnime.safetensors", "A simple color hand-drawn anime style illustration"),
            "-[ 🏠 Local models ]-":("", ""),
    }

    else :
        model_path_lora = model_path_lora_sd
        model_list_lora_builtin_fast = {
            "-[ 👍 🚀 Fast SD15 LoRA ]-":("Hyper-SD15-1step-lora.safetensors", ""),
            "ByteDance/Hyper-SD":("Hyper-SD15-1step-lora.safetensors", ""),
            "h1t/TCD-SD15-LoRA":("pytorch_lora_weights.safetensors", ""),
            "wangfuyun/PCM_Weights":("sd15/pcm_sd15_lcmlike_lora_converted.safetensors", ""),
            "jasperai/flash-sd":("pytorch_lora_weights.safetensors", ""),
        }
        model_list_lora_builtin = {
            "-[ 👏 🎚️ Sliders SD15 ]-":("color_temperature_slider_v1.safetensors", ""),
            "ostris/color-temperature-slider-lora":("color_temperature_slider_v1.safetensors", ""),
            "-[ 👍 SD15 LoRA ]-":("mj6-10.safetensors", ""),
            "Kvikontent/midjourney-v6":("mj6-10.safetensors", ""),
            "artificialguybr/icons-redmond-1-5v-app-icons-lora-for-sd-liberteredmond-sd-1-5":("IconsRedmond15V-Icons.safetensors", "icons, ios icon app, icons app"),
            "Norod78/SD15-IllusionDiffusionPattern-LoRA":("SD15-IllusionDiffusionPattern-LoRA.safetensors","IllusionDiffusionPattern"),
            "mnemic/MinionStyle-SD1.5-LoRA":("MinionStyle.safetensors", "MinionStyle"),
            "-[ 🏠 Local models ]-":("", ""),
        }

    os.makedirs(model_path_lora, exist_ok=True)
    model_list_lora = {
        "":("", ""),
    }
    model_list_lora_local = {}
    for filename in os.listdir(model_path_lora):
        f = os.path.join(model_path_lora, filename)
        if os.path.isfile(f) and filename.endswith('.safetensors') :
            final_f = {f:(f.split("/")[-1], "")}
            model_list_lora_local.update(final_f)

    if secondary_lora == False:
        model_list_lora.update(model_list_lora_builtin_fast)
    model_list_lora.update(model_list_lora_builtin)
    model_list_lora.update(model_list_lora_local)

    return model_list_lora

def txtinv_list(model):
    model = model_cleaner_sd(model)
    if is_sdxl(model):
        model_path_txtinv = "./models/TextualInversion/SDXL"
        model_list_txtinv_builtin = {
            "SalahZaidi/textual_inversion_cat_sdxl":("learned_embeds-steps-15000.safetensors", ""),
        }

    else:
        model_path_txtinv = "./models/TextualInversion/SD"
        model_list_txtinv_builtin = {
            "embed/EasyNegative":("EasyNegative.safetensors", "EasyNegative"),
            "sd-concepts-library/gta5-artwork":("learned_embeds.bin", "gta5-artwork"),
        }

    os.makedirs(model_path_txtinv, exist_ok=True)
    model_list_txtinv = {
        "":("", ""),
    }

    model_list_txtinv_local = {}
    for filename in os.listdir(model_path_txtinv):
        f = os.path.join(model_path_txtinv, filename)
        if os.path.isfile(f) and filename.endswith('.safetensors'):
            final_f = {f:(f.split("/")[-1], "")}
            model_list_txtinv_local.update(final_f)

    model_list_txtinv.update(model_list_txtinv_builtin)
    model_list_txtinv.update(model_list_txtinv_local)
    return model_list_txtinv
